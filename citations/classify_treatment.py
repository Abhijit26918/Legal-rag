"""Treatment classifier — rule-based first pass (TECH_DESIGN.md module 2,
step 3.2), v2.

v1 (fixed 300-char window around a plain substring match) was hand-validated
and found to have a serious precision problem: legal citations commonly
appear back-to-back, each with its own short trailing treatment note ("Case
A, cite, overruled by Case B, cite; see also Case C, cite"), and a wide
window around Case C's own citation routinely swept in the neighboring
"overruled by" note that actually belongs to Case A or B, not C. A second
failure mode: generic English uses of "following" ("the following facts")
and "distinguish" ("distinguished between two claims") got misread as legal
treatment language.

v2 fixes both by using eyecite to find the exact character span of every
citation in the text, then only looking for a signal phrase in the short
span of text immediately after a *specific* citation (stopping at the next
citation, or the next sentence break, whichever comes first) — matching how
these parentheticals are actually written, attached to the citation they
immediately follow.
"""
import csv
import re
import sys

import networkx as nx
from eyecite import get_citations

import config

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

MAX_TRAIL_CHARS = 150

# Checked in order — first match wins. Phrases are deliberately specific
# (not bare "following", which is common ordinary English) since they only
# get checked in the few characters immediately after a citation.
SIGNAL_RULES = [
    ("overruled", ["overruled by", "overruled on", "abrogated by", "abrogated on",
                   "superseded by", "vacated by", "no longer good law"]),
    ("distinguished", ["distinguished by", "distinguished on", "distinguishing"]),
    ("followed", ["we follow", "following our decision in", "is controlled by",
                  "we are bound by", "in accord with"]),
]

_SENTENCE_BREAK = re.compile(r"[.;]")


def _load_plain_text(mode):
    from ingest.bulk_ingest import DOCTRINE_MODES
    _, opinions_name = DOCTRINE_MODES[mode]
    path = config.PROCESSED_DIR / opinions_name
    texts = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            texts[row["id"]] = row.get("plain_text", "") or ""
    return texts


def _load_citation_tuples(mode):
    """cluster_id -> set of (volume, reporter, page) tuples, matching eyecite's c.groups."""
    name = "citation_strings_scoped.csv" if mode == "strict" else "citation_strings_scoped_loose.csv"
    path = config.PROCESSED_DIR / name
    by_cluster = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_cluster.setdefault(row["cluster_id"], set()).add(
                (row["volume"], row["reporter"], row["page"])
            )
    return by_cluster


def _classify_phrase(snippet):
    lower = snippet.lower()
    for label, phrases in SIGNAL_RULES:
        for phrase in phrases:
            if phrase in lower:
                return label
    return None


def _citations_with_treatment(text):
    """Runs eyecite once on `text`, returns a dict (volume, reporter, page) ->
    (label, confidence, evidence) — the best treatment label found for each
    distinct citation actually mentioned in this text (a case cited more than
    once keeps whichever mention carried the strongest signal)."""
    cites = get_citations(text)
    spans = []
    for c in cites:
        groups = getattr(c, "groups", None)
        if not groups or not all(k in groups for k in ("volume", "reporter", "page")):
            continue
        start, end = c.full_span()
        key = (groups["volume"], groups["reporter"], groups["page"])
        parenthetical = getattr(c.metadata, "parenthetical", None) or ""
        spans.append((start, end, key, parenthetical))
    spans.sort(key=lambda s: s[0])

    priority = {"overruled": 3, "distinguished": 2, "followed": 1, "neutral_cite": 0}
    results = {}
    for i, (start, end, key, parenthetical) in enumerate(spans):
        next_start = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        trail_end = min(end + MAX_TRAIL_CHARS, next_start)
        trail = text[end:trail_end]
        m = _SENTENCE_BREAK.search(trail)
        if m:
            trail = trail[:m.start()]

        label = _classify_phrase(parenthetical) or _classify_phrase(trail)
        confidence = 0.75 if _classify_phrase(parenthetical) else (0.6 if label else 0.3)
        label = label or "neutral_cite"
        evidence = parenthetical if _classify_phrase(parenthetical) else trail

        prev = results.get(key)
        if prev is None or priority[label] > priority[prev[0]]:
            results[key] = (label, confidence, evidence.strip())

    return results


def classify(mode="loose"):
    G = nx.read_graphml(config.GRAPH_PATH)
    plain_text = _load_plain_text(mode)
    citation_tuples = _load_citation_tuples(mode)

    text_citations_cache = {}

    counts = {"overruled": 0, "distinguished": 0, "followed": 0, "neutral_cite": 0,
              "not_located": 0, "no_reporter_citation": 0, "skipped": 0}
    samples = {"overruled": [], "distinguished": [], "followed": []}

    for citing, cited, data in G.edges(data=True):
        if G.nodes[citing]["is_external"] == 1 or G.nodes[cited]["is_external"] == 1:
            counts["skipped"] += 1
            continue

        cited_cluster = G.nodes[cited]["cluster_id"]
        candidates = citation_tuples.get(cited_cluster)
        if not candidates:
            data["treatment"] = "no_reporter_citation"
            data["confidence"] = 0.0
            counts["no_reporter_citation"] += 1
            continue

        if citing not in text_citations_cache:
            text_citations_cache[citing] = _citations_with_treatment(plain_text.get(citing, ""))
        found = text_citations_cache[citing]

        match = None
        for key in candidates:
            if key in found:
                match = found[key]
                break

        if match is None:
            data["treatment"] = "not_located"
            data["confidence"] = 0.0
            counts["not_located"] += 1
            continue

        label, confidence, evidence = match
        data["treatment"] = label
        data["confidence"] = confidence
        counts[label] += 1
        if label in samples and len(samples[label]) < 8:
            samples[label].append({
                "citing_case": G.nodes[citing]["case_name"],
                "cited_case": G.nodes[cited]["case_name"],
                "evidence": re.sub(r"\s+", " ", evidence).strip(),
            })

    print("Treatment classification v2 (internal -> internal edges only):")
    for label, n in counts.items():
        print(f"  {label}: {n}")

    nx.write_graphml(G, config.GRAPH_PATH)
    print(f"Updated graph saved to {config.GRAPH_PATH}")

    return counts, samples


if __name__ == "__main__":
    counts, samples = classify(mode="loose")
    print("\n--- Samples for spot-check ---")
    for label, examples in samples.items():
        print(f"\n[{label}]")
        for ex in examples:
            print(f"  {ex['citing_case']}  -->  {ex['cited_case']}")
            print(f"    evidence: ...{ex['evidence']}...")
