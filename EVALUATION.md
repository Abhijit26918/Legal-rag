# Evaluation Protocol — leg_RAG

This is the part that turns the project from a demo into a defensible
result. Treat it with the same rigor as the code — a weak eval set undermines
every metric downstream of it.

## 1. Eval set construction

**Target: 75–100 hand-verified Q&A pairs**, all within the scoped
circuit/doctrine, each with:
- the question
- a ground-truth answer
- ground-truth supporting case(s) and, ideally, the specific passage/holding
- (for trap questions) the overruling/distinguishing case and treatment type

**Ground-truth sourcing — don't freehand it.** Anchor each answer to
something citable, not just personal judgment:
- Case syllabi (where available from the ingestion pull) state the holding
  in the court's own words — use as primary anchor.
- Cross-check against a secondary summary (e.g. a case brief or headnote)
  before finalizing, to catch mis-readings.
- If a second labeler (classmate, advisor, or even a careful second pass by
  you a few days later) reviews a subset (~20%), report agreement — this is
  a standard eval-set-quality signal that makes the "hand-labeled" claim more
  defensible in the write-up, and costs little.

**Question types to include (not just trap cases):**
- Direct holding lookup ("what did case X hold on issue Y")
- Multi-hop ("what case overturned X, and what did that later affect")
- **Trap questions (≥10, per the project spec):** the naive-obvious answer
  cites a precedent that has since been overruled or distinguished. Pick
  these deliberately from doctrine areas with a known, documented overruling
  event in the scoped circuit/era — identify the overruling events *first*
  (from the citation graph's `overruled`-labeled edges, module 2), then write
  trap questions that would tempt a naive system into citing the stale case.
  This order (find real overruling events, then write questions around them)
  is more reliable than writing questions first and hoping traps exist in
  the corpus.
- A few "no good answer in corpus" questions (out-of-scope or genuinely
  unsettled law) — tests whether the system fabricates confidence it
  shouldn't have. Not in the original spec but cheap to add and strengthens
  the hallucination-rate metric.

## 2. Metrics — precise definitions

Define each metric as a formula before implementing `run_eval.py`, so there's
no ambiguity when writing up results later.

- **Answer accuracy**: fraction of questions where the generated answer's
  core claim matches ground truth (exact-match is too strict for free text —
  use LLM-judge-graded match against the ground-truth answer, with the
  rubric fixed in advance, or manual grading if the set is small enough).
- **Attribution precision** = (claims with a cited source that actually
  entails the claim) / (all claims with a citation marker).
- **Attribution recall** = (ground-truth-supportable claims that received a
  correct citation) / (all ground-truth-supportable claims in the answer).
- **Hallucination rate** = (claims with no supporting passage in the
  retrieved set, or contradicted by it) / (all claims in the answer).
- **Stale-precedent catch rate** (headline metric) = (trap questions where
  the system surfaced the precedent-currency warning) / (total trap
  questions). Report this **separately** from overall accuracy — a system
  can get the trap question's answer "wrong" in a generic sense but still
  succeed at the one thing being tested (flagging staleness), and that
  distinction matters for the write-up.
- **NLI/entailment validation accuracy**: agreement between the off-the-shelf
  entailment model's judgment and hand labels on the validation subset (see
  TECH_DESIGN.md module 5) — report this as a limitation/validity check on
  the attribution metrics above, not as a headline result itself.
- **Treatment-classifier precision/recall** on `overruled`/`distinguished`
  classes (see TECH_DESIGN.md module 2 step 3) — same purpose, reported as a
  validity check underpinning the stale-precedent metric.

## 3. Comparison design

Run the **same eval set** through two configurations:
1. **Baseline**: retrieve + generate, no attribution layer, no
   precedent-currency check.
2. **Full system**: baseline + attribution + precedent-currency +
   confidence score.

Optional third configuration (per original milestone 6): swap the strong
API model for the small local model in generation, holding retrieval and
verification constant — isolates how much the generation model itself
matters vs. the verification layer.

Report one comparison table: rows = metrics above, columns = configurations.
This table is the report's central empirical result.

## 4. Statistical honesty

With n=75–100, point estimates alone overstate precision. Report a
bootstrap confidence interval (or simple binomial CI for rate metrics like
stale-precedent catch rate) alongside each number, and say so explicitly in
the write-up's limitations section rather than presenting single numbers as
if they were exact. This is a small addition that meaningfully improves how
the report reads to an academic audience.

## 5. Cost/latency (industry-relevant, easy to add)

Log per-question: retrieval latency, generation latency, total tokens,
approximate cost (if using a paid API). Not a research contribution, but for
the "industry project" framing this is exactly the kind of table a
production-minded reviewer looks for and the original plan doesn't mention
it at all.

## 6. Write-up structure (suggested)

1. Problem + gap (from the project overview — the hallucinated/stale
   citation failure mode)
2. System architecture (summarize TECH_DESIGN.md)
3. Corpus + eval set construction (this doc, section 1) — include the
   labeler-agreement number
4. Results: the comparison table (section 3), broken out by metric
5. Validity checks: NLI validation accuracy, treatment-classifier
   precision/recall (section 2) — presented as "here's why you should trust
   the numbers above," which is what separates this from a demo write-up
6. Limitations: confidence intervals, single-circuit/doctrine scope,
   embedding model domain mismatch, eval set size
7. Cost/latency table
8. Future work
