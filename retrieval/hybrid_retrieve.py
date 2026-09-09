"""Hybrid retrieval (TECH_DESIGN.md module 3): vector top-k against the
query, then 1-hop graph expansion (citing + cited neighbors of each hit) so
multi-hop questions ("what overturned X, and what did that affect") can pull
in cases that wouldn't rank highly on text similarity alone. Merge, dedupe by
opinion (best-scoring chunk per opinion kept), cap total candidates.
"""
import chromadb
import networkx as nx
from openai import OpenAI

import config

_client = None
_collection = None
_graph = None


def _get_openai_client():
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise SystemExit("OPENAI_API_KEY not set — add it to .env.")
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = chroma_client.get_collection("opinions")
    return _collection


def _get_graph():
    global _graph
    if _graph is None:
        _graph = nx.read_graphml(config.GRAPH_PATH)
    return _graph


def _embed_query(text):
    resp = _get_openai_client().embeddings.create(model=config.EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def _best_chunk_for_opinion(query_embedding, opinion_id):
    res = _get_collection().query(
        query_embeddings=[query_embedding], n_results=1, where={"opinion_id": opinion_id}
    )
    if not res["ids"][0]:
        return None
    return {
        "chunk_id": res["ids"][0][0],
        "text": res["documents"][0][0],
        "metadata": res["metadatas"][0][0],
        "score": 1 - res["distances"][0][0],
    }


def hybrid_retrieve(query, top_k=None, graph_hops=None, max_candidates=None):
    top_k = top_k or config.RETRIEVAL_TOP_K
    graph_hops = graph_hops or config.GRAPH_EXPANSION_HOPS
    max_candidates = max_candidates or config.MAX_CANDIDATE_CHUNKS

    query_embedding = _embed_query(query)
    collection = _get_collection()
    G = _get_graph()

    # Step 1: vector top-k, dedupe to best chunk per opinion.
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    candidates = {}
    for chunk_id, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        opinion_id = meta["opinion_id"]
        score = 1 - dist
        if opinion_id not in candidates or score > candidates[opinion_id]["score"]:
            candidates[opinion_id] = {"chunk_id": chunk_id, "text": doc, "metadata": meta,
                                       "score": score, "source": "vector"}

    # Step 2: graph expansion — citing + cited neighbors of each seed hit,
    # `graph_hops` deep, restricted to internal opinions (external stub
    # nodes have no text/embedding to search).
    frontier = set(candidates.keys())
    seen = set(frontier)
    for _ in range(graph_hops):
        next_frontier = set()
        for opinion_id in frontier:
            if opinion_id not in G:
                continue
            for neighbor in set(G.predecessors(opinion_id)) | set(G.successors(opinion_id)):
                if G.nodes[neighbor].get("is_external") == 0 and neighbor not in seen:
                    next_frontier.add(neighbor)
        seen |= next_frontier
        frontier = next_frontier

    for opinion_id in seen - set(candidates.keys()):
        best = _best_chunk_for_opinion(query_embedding, opinion_id)
        if best:
            candidates[opinion_id] = {**best, "source": "graph"}

    # Step 3: cap.
    ranked = sorted(candidates.values(), key=lambda c: c["score"], reverse=True)[:max_candidates]
    return ranked


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Can police use a taser on a suspect who is already restrained?"
    print(f"Query: {query}\n")
    hits = hybrid_retrieve(query)
    n_vector = sum(1 for h in hits if h["source"] == "vector")
    n_graph = sum(1 for h in hits if h["source"] == "graph")
    print(f"{len(hits)} candidates ({n_vector} from vector search, {n_graph} from graph expansion)\n")
    for h in hits:
        m = h["metadata"]
        print(f"[{h['source']:6s}] score={h['score']:.3f}  {m['case_name']}  ({m['date_filed']})")
        print(f"    {h['text'][:200]!r}")
        print()
