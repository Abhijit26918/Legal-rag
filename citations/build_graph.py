"""Builds the citation graph (networkx.DiGraph) from the scoped ingestion
output. Nodes = opinions (internal, with metadata, or external stubs — see
below). Edges = citing -> cited, with placeholder treatment/confidence
attributes filled in later by classify_treatment.py.

"External" nodes: citations_scoped_loose.csv keeps an edge if EITHER end is
in our 1,696-opinion set, so some edges point to opinions outside that set
(e.g. a later case, possibly outside our court/doctrine scope, that cites or
overrules one of ours). We don't have full text or metadata for those — only
their bare opinion_id — so they're added as minimal stub nodes. This is
deliberate (see TECH_DESIGN.md module 2): precedent-currency checking needs
exactly these edges to know if a case has been overruled by something outside
the ingested corpus.
"""
import csv
import sys

import networkx as nx

import config

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _load_internal_opinions(mode):
    from ingest.bulk_ingest import DOCTRINE_MODES
    _, opinions_name = DOCTRINE_MODES[mode]
    path = config.PROCESSED_DIR / opinions_name
    opinions = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            opinions[row["id"]] = {"cluster_id": row["cluster_id"], "type": row["type"]}
    return opinions


def _load_cluster_metadata(cluster_ids):
    path = config.PROCESSED_DIR / "opinion_clusters_scoped.csv"
    meta = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["id"] in cluster_ids:
                meta[row["id"]] = {
                    "case_name": row.get("case_name", ""),
                    "date_filed": row.get("date_filed", ""),
                    "precedential_status": row.get("precedential_status", ""),
                }
    return meta


def _citations_path(mode):
    return config.PROCESSED_DIR / ("citations_scoped.csv" if mode == "strict" else "citations_scoped_loose.csv")


def build_graph(mode="loose"):
    internal_opinions = _load_internal_opinions(mode)
    cluster_ids = {o["cluster_id"] for o in internal_opinions.values()}
    cluster_meta = _load_cluster_metadata(cluster_ids)

    G = nx.DiGraph()
    for opinion_id, o in internal_opinions.items():
        cm = cluster_meta.get(o["cluster_id"], {})
        G.add_node(
            opinion_id,
            cluster_id=o["cluster_id"],
            opinion_type=o["type"],
            case_name=cm.get("case_name", ""),
            date_filed=cm.get("date_filed", ""),
            precedential_status=cm.get("precedential_status", ""),
            is_external=0,
        )

    edges_path = _citations_path(mode)
    n_edges = 0
    n_external_nodes = 0
    with open(edges_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            citing, cited = row["citing_opinion_id"], row["cited_opinion_id"]
            for node_id in (citing, cited):
                if node_id not in G:
                    G.add_node(node_id, cluster_id="", opinion_type="", case_name="",
                               date_filed="", precedential_status="", is_external=1)
                    n_external_nodes += 1
            G.add_edge(citing, cited, depth=int(row.get("depth") or 0),
                       treatment="unclassified", confidence=0.0)
            n_edges += 1

    internal_internal = sum(
        1 for u, v in G.edges if G.nodes[u]["is_external"] == 0 and G.nodes[v]["is_external"] == 0
    )
    internal_external = sum(
        1 for u, v in G.edges if G.nodes[u]["is_external"] == 0 and G.nodes[v]["is_external"] == 1
    )
    external_internal = sum(
        1 for u, v in G.edges if G.nodes[u]["is_external"] == 1 and G.nodes[v]["is_external"] == 0
    )

    print(f"Graph built: {G.number_of_nodes()} nodes "
          f"({len(internal_opinions)} internal, {n_external_nodes} external stub), "
          f"{n_edges} edges.")
    print(f"  internal -> internal: {internal_internal} (classifiable now — text available on both ends' cluster)")
    print(f"  internal -> external: {internal_external} (our case cites something outside scope — "
          f"cited case's citation string usually unknown, mostly unclassifiable for now)")
    print(f"  external -> internal: {external_internal} (something outside scope cites one of ours — "
          f"the direction precedent-currency checks care about most; needs the external opinion's "
          f"text, which we don't have in bulk — left unclassified, resolved per-case at query time instead)")

    out_path = config.GRAPH_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, out_path)
    print(f"Saved to {out_path}")
    return G


if __name__ == "__main__":
    build_graph(mode="loose")
