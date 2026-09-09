# leg_RAG — Case Law Precedent Lookup with Attribution & Precedent-Currency Verification

## Overview

A legal-domain RAG system that answers case law questions and, unlike a typical RAG demo, **proves its answers are trustworthy**: every claim in the generated answer is traced back to a specific source passage (verified, not just cited), and every cited case is checked for whether it's still good law or has since been overturned/distinguished.

This directly targets a real, well-publicized failure mode: lawyers (and now increasingly other LLM users) citing overturned or fabricated case law because the model doesn't actually check precedent currency. Most RAG systems don't verify this at all — that's the gap this project fills.

Deliverable: a working demo **plus** a small empirical comparison study (baseline RAG vs RAG+attribution+verification), backed by a hand-built evaluation set — written up like a compact research report, not just a demo app.

## Why this scope

- **Legal over medical/finance**: open, license-free data (CourtListener / Caselaw Access Project) — no HIPAA or licensed-data friction blocking solo work.
- **Precedent lookup over contract Q&A**: requires multi-hop reasoning over a citation graph, which is technically meatier and more interesting than flat document Q&A (e.g. CUAD-style contract clause lookup).
- **Attribution + verification layer over plain RAG**: plain RAG is a solved demo pattern at this point. Adding claim-level entailment checking and precedent-currency verification turns this into something with a genuine evaluation contribution, not just a wrapper around an LLM.
- **No training from scratch**: ruled out early — not a good use of effort for what this project needs to demonstrate. Everything here is retrieval + verification engineering, with an optional small fine-tuned model as a comparison point later.

## Architecture

1. **Corpus ingestion**
   Pull a scoped slice of case law from the CourtListener API / Caselaw Access Project: one circuit + one legal doctrine (e.g. qualified immunity / 4th Amendment), spanning ~20 years. Target ~5,000+ opinions — large enough to be a real corpus, small enough to stay tractable solo.

2. **Citation graph**
   Parse citation relationships between opinions, including treatment where available (followed / distinguished / overruled). Build with `networkx` — no need for a dedicated graph database at this scale.

3. **Retrieval (hybrid)**
   - Vector similarity search (sentence-transformer embeddings, e.g. bge/e5, stored in Chroma) for the initial candidate set.
   - Graph traversal from those hits to pull in citing/cited cases — this is what enables multi-hop questions like "what precedent overturned X, and what did that later affect?"

4. **Generation**
   LLM produces an answer with inline citations to specific retrieved passages.

5. **Attribution layer**
   The generated answer is decomposed into individual claims. Each claim is checked against its cited source passage using an off-the-shelf NLI/entailment model — this is what catches the LLM asserting something its own citation doesn't actually support.

6. **Precedent-currency check** *(the standout feature)*
   For every case cited in the answer, walk the citation graph forward: has any later case overruled or distinguished it? If so, surface a warning instead of presenting it as settled law.

7. **Confidence score**
   A single faithfulness indicator combining retrieval similarity, entailment score, and precedent-currency status, shown alongside the answer.

## Evaluation design (the part that makes this defensible)

- Hand-label 75–100 question/answer pairs within the scoped domain: ground-truth answer + ground-truth supporting case(s)/passage(s).
- Deliberately include **at least 10 "trap" questions** — cases where the obviously relevant precedent has since been overturned, to test whether the system catches it or naively cites stale law.
- **Metrics:**
  - Answer accuracy vs. ground truth
  - Attribution precision/recall (does the cited passage actually support the claim?)
  - Hallucination rate (claims with no support in retrieved material)
  - **Stale-precedent catch rate** — the headline metric. Rarely measured in existing RAG literature; this is the project's real contribution.
- **Comparison**: baseline RAG (retrieve + generate, no verification) vs. full system (RAG + attribution + precedent-currency). Optionally add a small fine-tuned model as a third comparison point later.

## Objectives / milestones

1. Corpus: ≥5,000 opinions ingested, citation graph built with treatment labels
2. Eval set: 75–100 hand-verified Q&A pairs, incl. ≥10 trap cases
3. Baseline RAG working end-to-end, accuracy measured against eval set
4. Attribution layer live, with a measurable hallucination-rate reduction vs. baseline
5. Stale-precedent catch rate ≥80% on trap cases
6. Small-model vs. strong-model generation quality compared
7. Final deliverable: demo (Streamlit) + written report with the comparison table

## Rough plan of attack

1. **Scope** — pick the exact circuit + doctrine, confirm data is available and sized right via CourtListener
2. **Ingest** — pull opinions, parse citation metadata, build the graph
3. **Baseline RAG** — chunking, embedding, retrieval, generation with citations, end to end
4. **Attribution + verification layer** — entailment checking, precedent-currency check, confidence scoring
5. **Eval set** — hand-label the Q&A pairs (including trap cases) — don't rush this step, it's the differentiator
6. **Comparison study** — run baseline vs. full system through the eval set, report metrics
7. **Polish** — demo UI, README, write-up

## Tech stack

- Python, hand-rolled pipeline (deliberately not LangChain/LlamaIndex — demonstrates understanding of internals rather than framework plumbing)
- Sentence-transformer embeddings (bge/e5) + Chroma for vector retrieval
- `networkx` for the citation graph
- Off-the-shelf HF NLI/entailment model for claim verification
- LLM for generation: an API model and/or a local small open model, compared side by side
- Streamlit for the demo UI
- Data source: CourtListener API / Caselaw Access Project (free, open)

## Immediate next step

Pick the specific circuit + legal doctrine to scope the corpus to, confirm data availability/volume via the CourtListener API, then set up the project structure and start ingestion.
