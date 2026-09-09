# Technical Design — leg_RAG

Implementation-ready spec. Follow this module-by-module; each section states the
interface, the concrete libraries/models to use, and the specific risk it's
addressing from the project review. Where an external API's exact field names
are involved, verify against current docs before coding — the shapes described
here are correct in structure but library/API surfaces drift.

## 0. Repo layout

```
leg_rag/
  ingest/
    fetch_opinions.py        # CourtListener API pull -> raw JSON
    clean_text.py            # strip OCR/editorial artifacts
    build_store.py            # raw JSON -> normalized SQLite table
  citations/
    extract_citations.py     # eyecite -> citation mentions
    classify_treatment.py    # signal-phrase + LLM classifier -> treatment labels
    build_graph.py            # networkx DiGraph construction
  retrieval/
    chunk.py                  # opinion -> chunks with metadata
    embed.py                  # chunks -> Chroma collection
    hybrid_retrieve.py        # vector search + graph expansion
  generation/
    prompts.py
    generate.py                # LLM call, inline citation markers
  attribution/
    decompose_claims.py       # answer -> atomic claims (LLM-prompted)
    entailment.py              # claim vs passage -> NLI score
  currency/
    check_precedent.py        # forward graph traversal for overruling
  scoring/
    confidence.py              # combine signals -> single score
  eval/
    labeled_set.jsonl          # hand-built eval set (see EVALUATION.md)
    run_eval.py                 # baseline vs full system, compute metrics
  demo/
    app.py                      # Streamlit UI
  data/                          # gitignored: raw pulls, sqlite db, chroma store
  config.py                       # model names, paths, thresholds — one place
  requirements.txt
```

Build order matches the numbered modules below — each one is independently
testable before the next depends on it.

## 1. Ingestion (`ingest/`)

**Source: CourtListener, via quarterly bulk CSV snapshots — not the live
REST API, and not the Caselaw Access Project.** CAP's bulk data is frozen
around 2018; the project's headline metric (stale-precedent catch rate)
needs recent overruling events, which only a live-maintained source
captures, so CAP is out. The live REST API is also out for the *ingestion*
pull specifically: free-tier accounts are capped at 125 requests/day (5/min,
50/hour), which makes pulling ~5,000 opinions through the paginated search
endpoint impractical (confirmed against courtlistener.com's account
dashboard and API docs — this is a hard account-level limit, not something
`fetch_opinions.py` can page around faster). The live API is still fine, and
still used, for anything low-volume later (e.g. spot-checking, small
lookups) — it just isn't the ingestion path.

**Use CourtListener's bulk data instead** — quarterly PostgreSQL `COPY TO`
CSV exports, published with headers, hosted publicly on S3
(`com-courtlistener-storage/bulk-data/`), no auth token and no rate limit.
Confirmed table sizes as of the last refresh: `courts` (~80KB), `dockets`
(~5GB compressed), `opinion-clusters` (~2.4GB compressed), `citation-map`
(~526MB compressed — the citation *graph edge* table; confusingly, the file
literally named `citations-*.csv.bz2` is a different table, reporter
citation strings per cluster like "530 U.S. 466" — not what module 2 needs,
verified by downloading its header), `opinions` (**~54GB compressed** — full
text of every opinion CourtListener has, across every US court, ever). That
last number is the one real cost in this pipeline; everything else here is
designed to avoid paying it more than once.

**Ingestion sequence — small tables first, confirm scope, then the big one:**

1. Download `courts.csv.bz2`, `dockets.csv.bz2`, `opinion-clusters.csv.bz2`
   (~7.4GB total, one-time). These already carry `court_id` (on dockets) and
   `date_filed` + `syllabus` (on opinion-clusters), which is enough to get an
   **exact count** of opinions in the scoped circuit + date window before
   touching any full opinion text — this replaces the API-based "volume
   check" from earlier plans entirely, and needs no API token.
2. Filter `dockets` to the target `court_id` → set of matching `docket_id`s.
3. Filter `opinion-clusters` to `docket_id` in that set AND `date_filed`
   within the target range → set of matching `cluster_id`s, with metadata
   (`case_name`, `syllabus`, `precedential_status`) saved locally. **This
   count is the volume check.** If it's off target, adjust the date range or
   circuit before going further — don't touch the 54GB file until scope is
   confirmed.
4. Only then: stream-filter `opinions.csv.bz2` for rows whose `cluster_id`
   is in the confirmed set, writing matches to a local file as you go and
   discarding the rest. This still requires downloading the full 54GB
   stream once (bulk data isn't server-side filterable, and rows aren't
   sorted by court), but never requires storing the full decompressed
   dataset — peak local disk usage stays close to just the ~5,000 matched
   rows. Run this as a one-time background job.
5. Filter `citation-map.csv.bz2` (~526MB — small enough to download whole)
   to edges where either `citing_opinion_id` or `cited_opinion_id` is in the
   matched opinion set — this is the raw input to the citation graph
   (module 2). The table is exactly `{id, depth, cited_opinion_id,
   citing_opinion_id}` — `depth` is a citation-frequency count, **not** a
   treatment/polarity signal. Confirms there is no free structured
   "overruled by" data anywhere in CourtListener — the treatment classifier
   in module 2 is doing real work, not filling in a gap that could've been
   avoided.

A `search_opinion` row (the actual opinion, vs. `search_opinioncluster` the
case) carries `type` (majority/dissent/concurrence — exact values are
CourtListener's internal codes, check a sample), `plain_text`, `cluster_id`,
`per_curiam`. Store majority opinion text as primary; keep concurrence/
dissent as separate rows flagged by type — don't merge them into one blob,
they carry different precedential weight.

**Normalized schema (SQLite table `opinions`):**

| column | type | notes |
|---|---|---|
| opinion_id | text PK | CourtListener ID |
| cluster_id | text | groups majority/concurrence/dissent |
| case_name | text | |
| court | text | |
| date_filed | date | |
| opinion_type | text | majority / concurrence / dissent |
| citation_string | text | canonical citation, e.g. "410 U.S. 113" |
| raw_text | text | uncleaned |
| clean_text | text | after `clean_text.py` |
| syllabus | text | if available — useful later as ground-truth anchor |

**Text cleaning (`clean_text.py`):** strip headnote/editorial content if any
publisher metadata leaked in, normalize whitespace, but **do not strip inline
citations from `clean_text`** — the citation extraction step (module 2) needs
them. Produce a *separate* `embedding_text` field later in the chunking step
that has citations replaced with a placeholder token, so embeddings aren't
dominated by string-cite noise. Keep both.

## 2. Citation graph (`citations/`)

This is the highest-risk module in the project — budget the most time here.

**Step 1 — extract citation mentions.** Use `eyecite` (Free Law Project's own
open-source citation parser, built for exactly this corpus). It finds every
citation string in `clean_text` and resolves it to a normalized citation
object. This solves citation *detection*; it does not tell you treatment.

**Step 2 — resolve mentions to opinion_ids.** Match each extracted citation
against your `opinions` table (and, for citations pointing outside your
scoped corpus, keep them as external nodes with metadata-only, no full text —
your graph should include edges to cases outside the 5k-opinion scope, since
"has this been overruled" often points to a later case you didn't ingest).

**Step 3 — classify treatment polarity.** No free structured Shepard's-style
data exists at scale, so build this as a small weakly-supervised classifier:

1. For each citation mention, extract a text window around it (the citing
   sentence + 1 sentence before/after).
2. Rule-based first pass: signal phrases map directly to labels —
   "overruled by", "abrogated by", "superseded by" → `overruled`;
   "distinguished" → `distinguished`; "following", "consistent with" →
   `followed`; no signal phrase → `neutral_cite`.
3. For windows the rules don't confidently classify, prompt an LLM
   classifier (few-shot, fixed label set) and record its confidence.
4. **Validate this classifier against a hand-labeled subset (~100 citation
   instances) before trusting it anywhere downstream.** Report precision/
   recall for the `overruled`/`distinguished` classes specifically — these
   are the ones the whole project's headline metric depends on. This
   validation is worth reporting in the writeup as its own small result.

**Graph construction (`build_graph.py`):** `networkx.DiGraph`. Node =
opinion (with metadata). Edge citing→cited, attributes: `treatment` (from
step 3), `confidence`, `date`. This graph is what both hybrid retrieval
(module 3) and precedent-currency checking (module 6) query — build it once,
persist to disk (pickle or graphml), don't rebuild per query.

## 3. Retrieval (`retrieval/`)

**Chunking:** split by paragraph, not fixed token windows — legal opinions
have real paragraph structure and fixed windows cut across holdings
mid-sentence. Target ~200–400 tokens per chunk with small overlap. For each
chunk, also produce `embedding_text` = clean_text with inline citations
replaced by a `[CITATION]` placeholder (keep the real citation string in
chunk metadata) — this stops the embedding model from over-weighting on
string cites that are dense but semantically flat.

Chunk metadata to retain: `opinion_id`, `chunk_id`, `case_name`, `court`,
`date_filed`, `opinion_type`, `char_span` (for highlighting the source
passage in the demo UI later).

**Embeddings:** `bge-large-en-v1.5` or `e5-large-v2` (both open,
sentence-transformers compatible). Store in a persistent Chroma collection,
one entry per chunk, metadata as above. Note as a stated limitation: these
are general-domain embedding models, not legal-domain-tuned — if time
allows, a legal-domain embedding model is a good ablation to compare, but
don't block the main pipeline on finding one.

**Hybrid retrieval (`hybrid_retrieve.py`):**
1. Vector top-k (k≈10) against the query.
2. Graph expansion: for each hit's `opinion_id`, pull 1-hop citing + cited
   neighbors from the citation graph (module 2's output).
3. Merge, dedupe by opinion, cap total candidate chunks (e.g. 15–20) before
   generation — this is what enables multi-hop questions ("what overturned
   X, and what did that affect") without blowing up context size.

## 4. Generation (`generation/`)

Prompt template: system instructions require inline citation markers tied
to chunk IDs (e.g. `[S1]`, `[S2]`) for every factual claim, and forbid
asserting anything not traceable to a marker. Pass the retrieved chunks with
their IDs in context.

**Model choice — decide concretely, don't leave "API model and/or local
model" open:** use one API model (e.g. Claude) as the primary generation
model, and one local open-weights instruct model (e.g. Llama 3.1 8B
Instruct or Mistral 7B Instruct, run via `transformers` or `ollama`) as the
comparison point for the "small vs strong model" milestone. Pick both before
starting module 4 so prompts can be tuned against both, not retrofitted.

## 5. Attribution layer (`attribution/`)

**Claim decomposition:** don't use sentence-splitting — legal sentences
bundle multiple assertions. Use an LLM-prompted decomposition: given the
generated answer with its citation markers, prompt for a JSON list of atomic
claims, each retaining which marker(s) it depends on. This is a well-defined,
testable subroutine — write a handful of unit-test cases (answer text in,
expected claim list out) before wiring it into the full pipeline.

**Entailment check:** off-the-shelf NLI model (e.g.
`cross-encoder/nli-deberta-v3-base`) scoring each (claim, cited passage)
pair as entailment/neutral/contradiction. **State the domain-mismatch risk
explicitly and validate it**: hand-label ~50 (claim, passage) pairs from
your own pipeline's output and check the NLI model's agreement before
trusting its scores in the final metrics. If agreement is poor, an
LLM-as-judge entailment check (more expensive, likely more accurate on legal
text) is the fallback — worth having both implemented so the eval can
compare them, which itself is a reportable result.

## 6. Precedent-currency check (`currency/`)

Given the set of cases cited in the final answer: for each, traverse the
citation graph *forward* (find nodes with an edge pointing *to* this case,
i.e. later cases citing it) filtered to `treatment in {overruled,
superseded, abrogated}`. Do this to depth 2 (an overruling case can itself
later be the subject of further treatment — catches indirect chains, e.g.
"X was distinguished by Y, and Y was overruled by Z" still means X's status
is unsettled). Output a structured warning: `{cited_case, overruling_case,
treatment, date}` for each hit, surfaced instead of presenting the citation
as settled law.

## 7. Confidence score (`scoring/`)

State the combination method explicitly rather than an ad hoc weighted sum
picked by feel: e.g. logistic combination of (retrieval similarity,
entailment score, precedent-currency flag as binary penalty), with weights
fit or at least sanity-checked against the hand-labeled eval set — compare
the score's ranking against human judgment of answer trustworthiness on a
subset. Document whatever method is chosen and why; "we combined three
signals" isn't defensible on its own in the write-up, showing the
combination was validated is.

## 8. Eval harness (`eval/`)

See EVALUATION.md for the full protocol. `run_eval.py` should run both
pipeline configurations (baseline RAG; full system) over the same labeled
set and emit one metrics table — this is the artifact that becomes the
report's comparison table.

## 9. Demo (`demo/`)

Streamlit. Show: the answer with citation markers, the source passage for
each marker (highlighted via `char_span`), entailment score per claim,
precedent-currency warnings inline (not buried), and the combined confidence
score. This is also the fastest way to sanity-check the pipeline manually
during development — don't leave it until the end.

## Config & secrets

One `config.py` (or `.env` + loader) for: API keys (CourtListener, LLM
provider), model names, chunk size, retrieval k, graph traversal depth,
confidence-score weights. Nothing hardcoded inline in module files — every
threshold above should be adjustable from one place, since you'll be tuning
several of them against the eval set.

## Suggested implementation order

1. Ingestion, with the volume-check gate before writing anything else.
2. Citation extraction + graph (start the treatment-classifier validation
   early — it's the biggest unknown, don't discover it's weak in week 6).
3. Chunking + embedding + baseline vector retrieval.
4. Baseline generation (no attribution/currency yet) — get an ugly
   end-to-end pipeline working before layering verification on top.
5. Hybrid retrieval (add graph expansion).
6. Attribution layer (decomposition + entailment), validated against hand
   labels.
7. Precedent-currency check.
8. Confidence scoring.
9. Eval set construction (can start in parallel with 3–4, doesn't block on
   the full pipeline).
10. Run comparison study, then demo UI, then write-up.
