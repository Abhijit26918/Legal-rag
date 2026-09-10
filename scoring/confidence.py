"""Confidence score (TECH_DESIGN.md module 7): combines retrieval
similarity, the entailment verdict, and precedent-currency status into one
trust signal per claim, then the whole answer.

**Combination method, stated explicitly (not just "we combined three
signals" — TECH_DESIGN is explicit that the method itself needs to be
defensible):**

    base = sigmoid(W0 + W1 * retrieval_similarity + W2 * entailment_score)
    claim_confidence = base * (CURRENCY_PENALTY_FACTOR if any case this
                                claim cites has a confirmed stale-precedent
                                warning, else 1.0)
    answer_confidence = min(claim_confidence over all claims)

Weakest-link aggregation, not the mean: an answer is only as trustworthy as
its least-supported claim. Averaging would let one fabricated claim hide
behind several well-supported ones — exactly the failure mode this whole
project exists to catch (see the live hallucination example in
[[generation-module-journey]]).

**Only "factual" claims set the floor, not "synthesis" ones.** First version
of this scored every uncited claim as maximally untrustworthy (entailment
score 0), including ordinary framing/transition/conclusion sentences that
don't assert anything new — since almost every real answer has a few of
those, the overall score always bottomed out near 0 regardless of how well-
supported the actual factual content was, making the metric useless (a
score that's always ~0.05 can't distinguish a good answer from a bad one).
Fixed by having `decompose_claims` tag each claim "factual" or "synthesis"
(see its docstring) — synthesis claims are still shown in the breakdown but
don't drag the answer-level floor down for lacking a citation they were
never supposed to need.

Entailment dominates the logistic (W2 >> W1) because module 5's validation
found the LLM-judge entailment check dramatically more reliable than raw
retrieval similarity as a correctness signal (50% agreement with the NLI
baseline, LLM consistently right on inspection) — retrieval similarity on
its own only says a chunk is topically related, not that it actually
supports the claim.

**Explicit caveat, per TECH_DESIGN's own instruction that weights should be
"fit or at least sanity-checked against the hand-labeled eval set":** the
weights below are a reasoned default, NOT fit against real labels — module
8's eval set doesn't exist yet. Revisit once it does; don't treat these
numbers as final.
"""
import math

# See module docstring — reasoned defaults, not yet fit/validated against
# a hand-labeled eval set (module 8 doesn't exist yet).
W0 = -3.0  # bias — keeps confidence low absent positive signal
W1 = 1.0   # retrieval similarity weight
W2 = 4.0   # entailment weight — dominant, see docstring
CURRENCY_PENALTY_FACTOR = 0.15

ENTAILMENT_SCORES = {
    "supported": 1.0,
    "not_supported": 0.2,
    "contradicted": 0.0,
    "uncited": 0.0,
    "marker_not_found": 0.0,
}


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def score_claim(checked_claim, marker_map, currency_warnings):
    markers = checked_claim["markers"]
    sims = [marker_map[m]["score"] for m in markers if m in marker_map]
    retrieval_sim = sum(sims) / len(sims) if sims else 0.0

    entailment_score = ENTAILMENT_SCORES.get(checked_claim["verdict"], 0.0)

    base = _sigmoid(W0 + W1 * retrieval_sim + W2 * entailment_score)

    cited_ids = {marker_map[m]["opinion_id"] for m in markers if m in marker_map}
    stale_ids = {w["cited_opinion_id"] for w in currency_warnings}
    has_stale = bool(cited_ids & stale_ids)

    confidence = base * (CURRENCY_PENALTY_FACTOR if has_stale else 1.0)

    return {
        **checked_claim,
        "retrieval_sim": retrieval_sim,
        "entailment_score": entailment_score,
        "has_stale_precedent": has_stale,
        "confidence": confidence,
    }


def score_answer(checked_claims, marker_map, currency_warnings):
    """Every claim gets scored and reported, but only "factual" claims (see
    decompose_claims' docstring) set the answer-level floor — a "synthesis"
    claim (framing/transition/conclusion) isn't supposed to carry its own
    citation, so it shouldn't be able to drag the score down for lacking
    one. Falls back to all claims if none are tagged "factual" (defensive —
    shouldn't happen, but a hallucinated answer that's ALL synthesis claims
    with no factual content is itself suspicious and worth a low score)."""
    scored = [score_claim(c, marker_map, currency_warnings) for c in checked_claims]
    if not scored:
        return {"answer_confidence": 0.0, "mean_claim_confidence": 0.0, "weakest_claim": None, "claims": []}

    factual = [c for c in scored if c.get("claim_type", "factual") == "factual"]
    floor_pool = factual or scored

    weakest = min(floor_pool, key=lambda c: c["confidence"])
    mean_confidence = sum(c["confidence"] for c in scored) / len(scored)
    mean_factual_confidence = sum(c["confidence"] for c in floor_pool) / len(floor_pool)

    return {
        "answer_confidence": weakest["confidence"],
        "mean_claim_confidence": mean_confidence,
        "mean_factual_confidence": mean_factual_confidence,
        "weakest_claim": weakest,
        "claims": scored,
    }


if __name__ == "__main__":
    import sys

    from generation.generate import generate_answer
    from attribution.decompose_claims import decompose_claims
    from attribution.entailment import check_answer
    from currency.check_precedent import check_precedent_currency

    query = " ".join(sys.argv[1:]) or "Can police use a taser on a suspect who is already handcuffed and restrained?"
    print(f"Query: {query}\n")

    result = generate_answer(query)
    print(f"Answer:\n{result['answer']}\n")

    claims = decompose_claims(result["answer"])
    checked = check_answer(claims, result["sources"])

    cited_opinion_ids = [result["sources"][m]["opinion_id"] for m in result["markers_used"]]
    warnings = check_precedent_currency(cited_opinion_ids)

    scored = score_answer(checked, result["sources"], warnings)

    print(f"{'='*70}")
    print(f"ANSWER CONFIDENCE: {scored['answer_confidence']:.2f}  "
          f"(mean over factual claims: {scored['mean_factual_confidence']:.2f}, "
          f"mean over all claims: {scored['mean_claim_confidence']:.2f})")
    print(f"{'='*70}\n")

    if scored["weakest_claim"]:
        w = scored["weakest_claim"]
        print(f"Weakest FACTUAL claim (sets the overall score): [{w['confidence']:.2f}] {w['claim']}")
        print(f"  verdict={w['verdict']}  retrieval_sim={w['retrieval_sim']:.2f}  "
              f"entailment_score={w['entailment_score']:.2f}  stale_precedent={w['has_stale_precedent']}\n")

    print("Per-claim breakdown:")
    for c in sorted(scored["claims"], key=lambda c: c["confidence"]):
        flag = " [STALE PRECEDENT]" if c["has_stale_precedent"] else ""
        print(f"  [{c['confidence']:.2f}] ({c['claim_type']}/{c['verdict']}){flag} {c['claim']}")
