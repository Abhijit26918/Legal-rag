"""Precedent-currency check (TECH_DESIGN.md module 6) — the project's
standout feature: for each case actually cited in a generated answer, walk
the citation graph *forward* (who cites INTO this case) looking for
negative treatment (overruled/superseded/abrogated), to depth 2 so an
indirect chain ("X was distinguished by Y, and Y was overruled by Z" — X's
status is unsettled) still surfaces.

Deliberately query-time, not bulk: per [[treatment-classifier-journey]] the
rule-based graph-wide classifier (citations/classify_treatment.py) proved
too unreliable to trust directly, and per module 5's entailment validation,
an LLM-judge is dramatically more accurate than any cheap heuristic on this
kind of legal-nuance question. So this module re-checks treatment live,
with the LLM, only for the small number of cases that actually appear in
one answer — not the whole 56,113-edge graph.

Case targeted by this check is always internal (one of our 1,696 ingested
opinions) — that's what generation ever cites, since it only draws from
retrieved chunks. Forward-citers (who cites the target) are frequently
*external* (54,010 of 56,113 edges touch a case outside our corpus) — those
need a live CourtListener text fetch, capped per config.CURRENCY_MAX_EXTERNAL_LOOKUPS.
"""
import csv
import re
import sys

import networkx as nx
from eyecite import get_citations
from openai import OpenAI

import config
from ingest.fetch_opinions import get_opinion
from llm_utils import with_retry

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TREATMENT_SYSTEM = """You are checking how one court opinion treats a specific \
earlier case it cites.

Read the excerpt below (it surrounds a citation to the earlier case) and \
classify the treatment. Respond with ONLY a JSON object, no other text, no \
markdown fence: {"treatment": "overruled", "reason": "<one sentence>"}

treatment must be exactly one of:
- "overruled": the excerpt says the earlier case has been overruled, abrogated, or superseded — it is no longer good law
- "distinguished": the excerpt says the earlier case doesn't apply here due to different facts, without saying it's wrong
- "followed": the excerpt applies or agrees with the earlier case's holding
- "no_clear_signal": just a citation with no explicit treatment either way"""

TREATMENT_USER = """Earlier case being discussed: {target_case_name}

Excerpt from the citing opinion:
{window}

How does this excerpt treat {target_case_name}?"""

_graph = None
_internal_texts = None
_citation_strings = None
_external_text_cache = {}
_groq_client = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = nx.read_graphml(config.GRAPH_PATH)
    return _graph


def _get_internal_texts():
    global _internal_texts
    if _internal_texts is None:
        from ingest.bulk_ingest import DOCTRINE_MODES
        _, opinions_name = DOCTRINE_MODES["loose"]
        path = config.PROCESSED_DIR / opinions_name
        _internal_texts = {}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                _internal_texts[row["id"]] = row.get("plain_text", "") or ""
    return _internal_texts


def _get_citation_strings():
    global _citation_strings
    if _citation_strings is None:
        path = config.PROCESSED_DIR / "citation_strings_scoped_loose.csv"
        _citation_strings = {}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                _citation_strings.setdefault(row["cluster_id"], []).append(
                    (row["volume"], row["reporter"], row["page"])
                )
    return _citation_strings


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise SystemExit("GROQ_API_KEY not set — add it to .env.")
        _groq_client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _groq_client


def _get_text_for_node(node_id, graph):
    """Internal nodes: text is already local, free. External nodes: one
    live CourtListener fetch, cached for the life of this process so the
    same external opinion is never fetched twice in one run."""
    if graph.nodes[node_id]["is_external"] == 0:
        return _get_internal_texts().get(node_id, "")
    if node_id in _external_text_cache:
        return _external_text_cache[node_id]
    try:
        data = get_opinion(node_id)
        text = data.get("plain_text") or ""
    except Exception:
        text = ""
    _external_text_cache[node_id] = text
    return text


def _display_name(node_id, graph):
    node = graph.nodes[node_id]
    if node.get("case_name"):
        return node["case_name"], node.get("date_filed", "")
    if node_id in _external_text_cache or graph.nodes[node_id]["is_external"] == 1:
        try:
            data = get_opinion(node_id)
            slug = data.get("absolute_url", "").strip("/").split("/")[-1]
            name = slug.replace("-", " ").title() if slug else f"opinion #{node_id}"
            return name, ""
        except Exception:
            pass
    return f"opinion #{node_id}", ""


def _find_windows(citing_text, target_citation_tuples, window_chars=400):
    """Locates every mention of the target case in the citing opinion's
    text via eyecite's exact citation spans (not plain substring search —
    see citations/classify_treatment.py's v1/v2 history for why that
    matters), returns a generous bidirectional window around each — wide
    enough for the LLM to find the relevant clause itself, unlike the
    narrower trailing-only window the old rule-based classifier used."""
    cites = get_citations(citing_text)
    windows = []
    for c in cites:
        groups = getattr(c, "groups", None)
        if not groups:
            continue
        key = (groups.get("volume"), groups.get("reporter"), groups.get("page"))
        if key in target_citation_tuples:
            start, end = c.full_span()
            w_start = max(0, start - window_chars)
            w_end = min(len(citing_text), end + window_chars)
            windows.append(citing_text[w_start:w_end])
    return windows


def _check_treatment_llm(window, target_case_name, model=None, max_tokens=6000):
    # 6000 not 1500 — see attribution/entailment.py's check_entailment_llm
    # docstring: openai/gpt-oss-120b's variable hidden-reasoning consumption
    # has emptied out real responses at lower budgets on a small-input call
    # just like this one. Input here is small, so there's plenty of TPM
    # headroom to go this high.
    client = _get_groq_client()

    def _call_and_parse():
        response = client.chat.completions.create(
            model=model or config.GENERATION_MODEL_GROQ_PRIMARY,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": TREATMENT_SYSTEM},
                {"role": "user", "content": TREATMENT_USER.format(target_case_name=target_case_name, window=window)},
            ],
        )
        raw = response.choices[0].message.content
        match = re.search(r'"treatment"\s*:\s*"(\w+)"', raw or "")
        if not match:
            # Retriable, not a silent "nothing found": an empty/unparseable
            # response (gpt-oss-120b's reasoning-token consumption is random
            # per call) must not be allowed to masquerade as a confirmed
            # "no_clear_signal" verdict — that would make the currency check
            # falsely report "still good law" when it actually just got no
            # usable response. See attribution/entailment.py's same pattern.
            raise ValueError(f"Could not parse treatment verdict. Raw output:\n{raw}")
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
        return match, reason_match

    match, reason_match = with_retry(_call_and_parse)
    return {"treatment": match.group(1), "reason": reason_match.group(1) if reason_match else ""}


def _check_one_edge(citer_id, target_id, target_case_name, graph, external_budget):
    """Checks how `citer_id` treats `target_id`. Returns (result_dict or
    None if no mention found / budget exhausted, external_calls_used)."""
    is_external = graph.nodes[citer_id]["is_external"] == 1
    if is_external and external_budget <= 0:
        return None, 0

    citing_text = _get_text_for_node(citer_id, graph)
    used = 1 if (is_external and citing_text) else 0
    if not citing_text:
        return None, used

    target_cluster = graph.nodes[target_id].get("cluster_id", "")
    target_cites = set(_get_citation_strings().get(target_cluster, []))
    if not target_cites:
        return None, used

    windows = _find_windows(citing_text, target_cites)
    if not windows:
        return None, used

    result = _check_treatment_llm(windows[0], target_case_name)
    return result, used


def check_precedent_currency(cited_opinion_ids, depth=None):
    """Main entry point. `cited_opinion_ids` — the internal opinion IDs
    actually cited in a generated answer (e.g. generate_answer()'s
    result["sources"][m]["opinion_id"] for each m in markers_used).
    Returns a list of structured warnings, one per negative-treatment hit
    found, each with the citation chain that surfaced it."""
    depth = depth or config.CURRENCY_TRAVERSAL_DEPTH
    graph = _get_graph()
    warnings = []
    external_budget = config.CURRENCY_MAX_EXTERNAL_LOOKUPS

    for target_id in cited_opinion_ids:
        if target_id not in graph:
            continue
        target_name = graph.nodes[target_id].get("case_name") or f"opinion #{target_id}"
        target_date = graph.nodes[target_id].get("date_filed", "")

        hop1_citers = list(graph.predecessors(target_id))
        print(f"[currency] {target_name}: checking {len(hop1_citers)} hop-1 citers "
              f"(external budget remaining: {external_budget})", file=sys.stderr)
        for i, citer_id in enumerate(hop1_citers):
            if (i + 1) % 10 == 0:
                print(f"[currency]   ...{i+1}/{len(hop1_citers)} checked, "
                      f"external budget remaining: {external_budget}", file=sys.stderr)
            result, used = _check_one_edge(citer_id, target_id, target_name, graph, external_budget)
            external_budget -= used
            if result is None or result["treatment"] not in config.STALE_TREATMENTS:
                continue

            citer_name, citer_date = _display_name(citer_id, graph)
            warnings.append({
                "cited_opinion_id": target_id,
                "cited_case": target_name,
                "cited_date": target_date,
                "overruling_case": citer_name,
                "overruling_date": citer_date,
                "treatment": result["treatment"],
                "reason": result["reason"],
                "chain": [target_name, citer_name],
            })

            if depth >= 2:
                hop2_citers = list(graph.predecessors(citer_id))
                for citer2_id in hop2_citers:
                    result2, used2 = _check_one_edge(citer2_id, citer_id, citer_name, graph, external_budget)
                    external_budget -= used2
                    if result2 is None or result2["treatment"] not in config.STALE_TREATMENTS:
                        continue
                    citer2_name, citer2_date = _display_name(citer2_id, graph)
                    warnings.append({
                        "cited_opinion_id": target_id,
                        "cited_case": target_name,
                        "cited_date": target_date,
                        "overruling_case": citer2_name,
                        "overruling_date": citer2_date,
                        "treatment": result2["treatment"],
                        "reason": f"(indirect, via {citer_name}) {result2['reason']}",
                        "chain": [target_name, citer_name, citer2_name],
                    })

    return warnings


if __name__ == "__main__":
    from generation.generate import generate_answer

    query = " ".join(sys.argv[1:]) or "Can police use a taser on a suspect who is already handcuffed and restrained?"
    result = generate_answer(query)
    cited_opinion_ids = [result["sources"][m]["opinion_id"] for m in result["markers_used"]]

    print(f"Query: {query}")
    print(f"Cases actually cited in the answer: {len(cited_opinion_ids)}\n")

    warnings = check_precedent_currency(cited_opinion_ids)
    if not warnings:
        print("No stale-precedent warnings found among the cited cases.")
    for w in warnings:
        print(f"[{w['treatment'].upper()}] {' -> '.join(w['chain'])}")
        print(f"  {w['reason']}")
        print()
