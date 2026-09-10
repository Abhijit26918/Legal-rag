"""Generates a batch of candidate cases for hand-labeling the eval set
(EVALUATION.md section 1) — solves the "ask Claude for one case at a time,
some turn out to be duds" problem by pre-filtering and pre-extracting a
likely-holding preview for a whole batch at once, so a human can skim many
quickly and only request full text for the ones worth writing up.

Publication status: Published opinions carry more precedential weight, but
in this corpus they run far longer than average (median ~45,700 chars vs.
~9,700 corpus-wide — the short, quick dispositions tend to be Unpublished).
Restricting to "short AND Published" leaves almost nothing to pick from, so
this includes both, sorted Published-first, with status shown clearly in
each preview so you can weigh that when choosing.

Also excludes:
- Text outside a reasonable length range (too short = often a procedural
  order/fee dispute with no real merits holding, like the withdrawal-order
  dud found by hand; too long = multi-issue sprawling opinion, harder to
  extract one clean question from)
- Known non-opinion patterns spotted by hand so far (e.g. "is withdrawn" —
  a docket-management order, not a ruling)
"""
import csv
import random
import re
import sys

import config

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

MIN_LEN = 2500
MAX_LEN = 15000
EXCLUDE_PATTERNS = [
    r"\bis withdrawn\b",
    r"petition for rehearing en banc is moot",
]
DISPOSITION_WORDS = ["AFFIRMED", "REVERSED", "VACATED", "REMANDED", "DISMISSED", "DENIED", "GRANTED"]


def _load_opinions_with_metadata():
    with open(config.PROCESSED_DIR / "opinions_scoped_loose.csv", encoding="utf-8", newline="") as f:
        opinions = list(csv.DictReader(f))
    cluster_ids = {o["cluster_id"] for o in opinions}
    meta = {}
    with open(config.PROCESSED_DIR / "opinion_clusters_scoped.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["id"] in cluster_ids:
                meta[row["id"]] = row
    for o in opinions:
        o["case_name"] = meta.get(o["cluster_id"], {}).get("case_name", "")
        o["date_filed"] = meta.get(o["cluster_id"], {}).get("date_filed", "")
        o["precedential_status"] = meta.get(o["cluster_id"], {}).get("precedential_status", "")
    return opinions


def _extract_preview(text):
    """Heuristic holding-preview extractor, following the same reading
    recipe in eval/LABELING_GUIDE.md: find the concluding disposition line,
    grab the paragraph before it (the reasoning), plus the opening
    paragraph (the case summary)."""
    opening = text[:500].strip()

    last_disposition_idx = -1
    for word in DISPOSITION_WORDS:
        idx = text.rfind(word)
        if idx > last_disposition_idx:
            last_disposition_idx = idx
    if last_disposition_idx == -1:
        return opening, "(no clear disposition line found — inspect manually)"

    context_start = max(0, last_disposition_idx - 700)
    context_end = min(len(text), last_disposition_idx + 100)
    conclusion = text[context_start:context_end].strip()
    return opening, conclusion


def pick_candidates(n=20, seed=42, exclude_ids=None):
    exclude_ids = exclude_ids or set()
    opinions = _load_opinions_with_metadata()

    filtered = []
    for o in opinions:
        if o["id"] in exclude_ids:
            continue
        text = o["plain_text"]
        if not (MIN_LEN <= len(text) <= MAX_LEN):
            continue
        if any(re.search(p, text) for p in EXCLUDE_PATTERNS):
            continue
        filtered.append(o)

    rng = random.Random(seed)
    rng.shuffle(filtered)
    # Published-first, but not published-only — see module docstring.
    filtered.sort(key=lambda o: o["precedential_status"] != "Published")
    return filtered[:n]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    candidates = pick_candidates(n=n, seed=seed)
    print(f"{len(candidates)} candidates (seed={seed} — pass a different number as the 2nd "
          f"argument to get a fresh batch, e.g. `python -m eval.candidate_picker 15 43`)\n")

    for o in candidates:
        opening, conclusion = _extract_preview(o["plain_text"])
        print(f"{'='*90}")
        print(f"{o['case_name']}  ({o['date_filed']})  [{o['precedential_status']}]  [opinion_id={o['id']}]")
        print(f"{'='*90}")
        print("OPENING:")
        print(f"  {opening}")
        print("LIKELY CONCLUSION/REASONING:")
        print(f"  ...{conclusion}...")
        print()
