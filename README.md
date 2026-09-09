# leg_RAG

Case law precedent lookup with claim-level attribution and precedent-currency
verification.

## What this is

A legal-domain RAG system that doesn't just retrieve-and-generate — it
proves its answers are trustworthy. Every claim in a generated answer is
checked against its cited source passage (entailment, not just citation),
and every cited case is checked against a citation graph to see whether it's
still good law or has since been overruled or distinguished. This targets a
real, well-publicized failure mode (lawyers and LLM users citing overturned
or fabricated precedent) that most RAG demos don't check for at all.

Scope: one federal circuit, one legal doctrine, ~20 years of opinions
(~5,000+), sourced from the CourtListener API.

Full design: [`leg_RAG_project.md`](leg_RAG_project.md) (original project
plan) · [`TECH_DESIGN.md`](TECH_DESIGN.md) (implementation spec) ·
[`EVALUATION.md`](EVALUATION.md) (eval protocol).

## Status

Pre-implementation. See TECH_DESIGN.md's "suggested implementation order"
for the build sequence.

## Architecture (summary)

```
CourtListener API
      │
      ▼
  Ingestion ──► Citation extraction (eyecite) ──► Treatment classifier
      │                                                    │
      ▼                                                    ▼
  Chunking + embedding (Chroma)                    Citation graph (networkx)
      │                                                    │
      └───────────────► Hybrid retrieval ◄─────────────────┘
                               │
                               ▼
                          Generation (LLM)
                               │
                               ▼
                    Claim decomposition + entailment check
                               │
                               ▼
                   Precedent-currency check (graph traversal)
                               │
                               ▼
                       Confidence score ──► Streamlit demo
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requires a free CourtListener API token and an API key for the chosen LLM
provider — see `config.py`.

## Data attribution

Case law data via the [CourtListener](https://www.courtlistener.com/) API
(Free Law Project). This project is a research/academic prototype — its
output is not legal advice and has not been validated for professional use.

## Repo layout

See TECH_DESIGN.md §0 for the full module breakdown (`ingest/`,
`citations/`, `retrieval/`, `generation/`, `attribution/`, `currency/`,
`scoring/`, `eval/`, `demo/`).
