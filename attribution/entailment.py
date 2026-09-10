"""Claim vs. cited-passage entailment check (TECH_DESIGN.md module 5). Two
methods, kept side by side rather than one replacing the other, per
TECH_DESIGN's own instruction that comparing them is itself a reportable
result:

- NLI cross-encoder (`check_entailment_nli`): fast, free, local. Hand-
  validated on real pipeline output (see memory: entailment-nli-validation)
  and found unreliable on this data — 2/2 deep-checked "not_supported"
  verdicts turned out to be confident (~99%) false negatives on near-
  verbatim matches. Working theory: it was trained on short SNLI/MNLI-style
  sentence pairs, not ~1,300-character legal paragraphs with the actual
  supporting sentence buried inside a lot of surrounding boilerplate.
- LLM-as-judge (`check_entailment_llm`): slower, small per-call cost via
  Groq, but reads the full passage with actual comprehension rather than a
  fixed-length embedding of it. Promoted to the primary method after the
  NLI validation failure above — not just an optional fallback.
"""
import json
import re

import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder

import config
from llm_utils import with_retry

_nli_model = None
_groq_client = None
_LABELS = ["contradiction", "entailment", "neutral"]

LLM_JUDGE_SYSTEM = """You are fact-checking one claim from a legal answer against \
the single source passage it was cited to support.

Read the ENTIRE passage carefully before deciding — the supporting text is \
often not in the first sentence.

Respond with ONLY a JSON object, no other text, no markdown fence: \
{"verdict": "supported", "reason": "<one sentence>"}

verdict must be exactly one of:
- "supported": the passage directly states or clearly implies the claim
- "contradicted": the passage states something that conflicts with the claim
- "not_supported": the passage does not address the claim either way"""

LLM_JUDGE_USER = """Claim: {claim}

Source passage:
{passage}

Does the source passage support this claim?"""


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder(config.NLI_MODEL)
    return _nli_model


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise SystemExit("GROQ_API_KEY not set — add it to .env.")
        _groq_client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _groq_client


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _extract_json_object(text):
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return json.loads(text)


def check_entailment_nli(claim, passage):
    """premise=passage (the source text), hypothesis=claim — asks "does the
    cited passage support this claim," the correct orientation for fact-
    checking a claim against its source."""
    scores = _get_nli_model().predict([(passage, claim)])[0]
    probs = _softmax(scores)
    label_idx = int(np.argmax(probs))
    verdict = {"entailment": "supported", "contradiction": "contradicted", "neutral": "not_supported"}[_LABELS[label_idx]]
    return {"verdict": verdict, "label": _LABELS[label_idx], "probs": dict(zip(_LABELS, probs.tolist()))}


def check_entailment_llm(claim, passage, model=None, max_tokens=6000):
    """LLM-as-judge — max_tokens=6000 for the same reason as
    decompose_claims: openai/gpt-oss-120b spends a variable, sometimes
    large, amount on hidden reasoning before the visible JSON verdict.
    2000 (the original setting here) wasn't actually generous enough —
    ran completely dry (empty content, unparseable) on a real claim check.
    Input is small either way, so there's ample TPM headroom to go this
    high."""
    client = _get_groq_client()

    def _call_and_parse():
        response = client.chat.completions.create(
            model=model or config.GENERATION_MODEL_GROQ_PRIMARY,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM},
                {"role": "user", "content": LLM_JUDGE_USER.format(claim=claim, passage=passage)},
            ],
        )
        raw = response.choices[0].message.content
        try:
            return _extract_json_object(raw)
        except (json.JSONDecodeError, AttributeError) as e:
            # Retriable, not fatal: gpt-oss-120b's hidden-reasoning token
            # spend is random per call, so a fresh attempt often succeeds
            # even at the same max_tokens — retrying only the API call (not
            # this parse step too) would keep retrying a call that already
            # succeeded-but-empty, never actually getting a new sample.
            raise ValueError(f"Could not parse LLM-judge verdict as JSON. Raw output:\n{raw}") from e

    result = with_retry(_call_and_parse)
    return {"verdict": result["verdict"], "reason": result.get("reason", "")}


def check_claim(claim_obj, marker_map, method="llm"):
    """Full check for one decomposed claim: no markers -> flagged
    immediately as unsupported (this is exactly the failure mode seen in
    the live hallucination example — an LLM asserting something with no
    citation at all). method: "nli" or "llm" (default — see module
    docstring for why LLM is primary, not NLI)."""
    markers = claim_obj["markers"]
    if not markers:
        return {**claim_obj, "verdict": "uncited"}

    passages = [marker_map[m]["text"] for m in markers if m in marker_map]
    if not passages:
        return {**claim_obj, "verdict": "marker_not_found"}

    combined_passage = "\n\n".join(passages)
    if method == "nli":
        result = check_entailment_nli(claim_obj["claim"], combined_passage)
    elif method == "llm":
        result = check_entailment_llm(claim_obj["claim"], combined_passage)
    else:
        raise ValueError(f"Unknown method {method!r} — expected 'nli' or 'llm'")
    return {**claim_obj, **result}


def check_answer(claims, marker_map, method="llm"):
    return [check_claim(c, marker_map, method=method) for c in claims]


if __name__ == "__main__":
    examples = [
        {"claim": "The officers used a taser on a handcuffed suspect.",
         "markers": ["S1"]},
        {"claim": "The Supreme Court banned all taser use nationwide.",
         "markers": ["S1"]},
        {"claim": "Casey v. City of Fed. Heights held that active resistance is required.",
         "markers": []},
    ]
    fake_marker_map = {
        "S1": {"text": "The officers deployed a taser against the suspect twice while he "
                        "was already handcuffed and offering no resistance."}
    }
    for method in ("nli", "llm"):
        print(f"=== method={method} ===")
        for result in check_answer(examples, fake_marker_map, method=method):
            print(f"[{result['verdict']:14s}] {result['claim']}")
            if result.get("probs"):
                print(f"    probs: {result['probs']}")
            if result.get("reason"):
                print(f"    reason: {result['reason']}")
