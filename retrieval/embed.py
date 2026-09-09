"""Chunks -> Chroma collection (TECH_DESIGN.md module 3), embedded via
OpenAI's API (config.EMBEDDING_MODEL) rather than a local sentence-transformer
— cheap enough at this corpus size (~$0.20-0.30 total for text-embedding-3-small)
and avoids downloading/running a local model on CPU.
"""
import json
import sys

import chromadb
from openai import OpenAI
from tqdm import tqdm

import config

BATCH_SIZE = 200


def _load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _embed_batch(client, texts):
    resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_chunks(chunks_path=None, collection_name="opinions"):
    chunks_path = chunks_path or (config.PROCESSED_DIR / "chunks_loose.jsonl")
    chunks = _load_chunks(chunks_path)
    print(f"{len(chunks)} chunks to embed.")

    if not config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY not set — add it to .env.")
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), unit="batch"):
        batch = chunks[i:i + BATCH_SIZE]
        embeddings = _embed_batch(client, [c["embedding_text"] for c in batch])
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[{
                "opinion_id": c["opinion_id"],
                "cluster_id": c["cluster_id"],
                "case_name": c["case_name"],
                "court": c["court"],
                "date_filed": c["date_filed"],
                "opinion_type": c["opinion_type"],
                "precedential_status": c["precedential_status"],
                "char_start": c["char_span"][0],
                "char_end": c["char_span"][1],
            } for c in batch],
        )

    print(f"Collection '{collection_name}' now has {collection.count()} chunks. Stored at {config.CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    embed_chunks()
