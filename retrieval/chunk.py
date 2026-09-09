"""Opinion -> chunks with metadata (TECH_DESIGN.md module 3).

Data reality check before chunking: CourtListener's `plain_text` for this
corpus is PDF-derived and uses a double-newline as a generic *line* break —
it shows up mid-sentence just as often as between real paragraphs — plus
page-number lines and form-feed (\\x0c) page-break characters. There's no
reliable paragraph signal to split on, so "split by paragraph" (as originally
planned) isn't workable on this data. Deliberate deviation: clean the text
into continuous prose, then chunk by real token count — using `tiktoken`,
matching how the actual embedding model (OpenAI text-embedding-3-small)
counts tokens, not a word-count approximation — with overlap, snapping chunk
boundaries so they never land inside a citation.
"""
import csv
import re
import sys

import tiktoken
from eyecite import get_citations

import config

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_PAGE_NUM_LINE = re.compile(r"\n[ \t]*\d{1,4}[ \t]*\n")
_FORM_FEED = "\x0c"

_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.encoding_for_model(config.EMBEDDING_MODEL)
    return _encoding


def clean_text(text):
    text = text.replace(_FORM_FEED, "\n")
    text = _PAGE_NUM_LINE.sub("\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_opinions(mode="loose"):
    from ingest.bulk_ingest import DOCTRINE_MODES
    _, opinions_name = DOCTRINE_MODES[mode]
    path = config.PROCESSED_DIR / opinions_name
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _snap_to_citation_boundary(pos, citation_spans):
    """If `pos` falls inside a citation's span, push it out to the span's
    end — never cut a chunk boundary through the middle of a case citation."""
    for start, end in citation_spans:
        if start < pos < end:
            return end
    return pos


def _chunk_one_opinion(cleaned_text, citation_spans, encoding, token_size, token_overlap):
    token_ids = encoding.encode(cleaned_text)
    # tiktoken doesn't give per-token char offsets directly, so decode
    # incrementally to recover them — fine at this corpus size.
    offsets = []
    cursor = 0
    for tok_bytes in encoding.decode_tokens_bytes(token_ids):
        tok_str = tok_bytes.decode("utf-8", errors="ignore")
        start = cleaned_text.find(tok_str, cursor) if tok_str else cursor
        if start == -1:
            start = cursor
        end = start + len(tok_str)
        offsets.append((start, end))
        cursor = end

    if not offsets:
        return []

    chunks = []
    n_tokens = len(offsets)
    tok_start = 0
    while tok_start < n_tokens:
        tok_end = min(tok_start + token_size, n_tokens)
        char_start = offsets[tok_start][0]
        char_end = offsets[tok_end - 1][1]
        char_end = _snap_to_citation_boundary(char_end, citation_spans)
        chunks.append((char_start, char_end))
        if tok_end >= n_tokens:
            break
        tok_start = tok_end - token_overlap
    return chunks


def _make_embedding_text(chunk_text, chunk_start, citation_spans):
    """Replaces each citation's text with a [CITATION] placeholder so the
    embedding isn't dominated by dense-but-semantically-flat string cites."""
    pieces = []
    cursor = 0
    for start, end in citation_spans:
        rel_start, rel_end = start - chunk_start, end - chunk_start
        if rel_start < 0 or rel_end > len(chunk_text):
            continue
        pieces.append(chunk_text[cursor:rel_start])
        pieces.append("[CITATION]")
        cursor = rel_end
    pieces.append(chunk_text[cursor:])
    return "".join(pieces)


def build_chunks(mode="loose", token_size=None, token_overlap=None):
    token_size = token_size or config.CHUNK_TOKEN_SIZE
    token_overlap = token_overlap or config.CHUNK_TOKEN_OVERLAP
    encoding = _get_encoding()

    opinions = _load_opinions(mode)
    cluster_meta = _load_cluster_metadata({o["cluster_id"] for o in opinions})

    all_chunks = []
    for o in opinions:
        cleaned = clean_text(o["plain_text"])
        cites = get_citations(cleaned)
        citation_spans = [c.full_span() for c in cites]

        cm = cluster_meta.get(o["cluster_id"], {})
        spans = _chunk_one_opinion(cleaned, citation_spans, encoding, token_size, token_overlap)

        for i, (char_start, char_end) in enumerate(spans):
            chunk_text = cleaned[char_start:char_end]
            embedding_text = _make_embedding_text(chunk_text, char_start, citation_spans)
            all_chunks.append({
                "chunk_id": f"{o['id']}_{i}",
                "opinion_id": o["id"],
                "cluster_id": o["cluster_id"],
                "case_name": cm.get("case_name", ""),
                "court": config.COURT_ID,
                "date_filed": cm.get("date_filed", ""),
                "opinion_type": o.get("type", ""),
                "precedential_status": cm.get("precedential_status", ""),
                "char_span": [char_start, char_end],
                "text": chunk_text,
                "embedding_text": embedding_text,
            })

    return all_chunks


if __name__ == "__main__":
    import json

    chunks = build_chunks(mode="loose")
    out_path = config.PROCESSED_DIR / "chunks_loose.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_opinions = len({c["opinion_id"] for c in chunks})
    lens = [len(c["text"]) for c in chunks]
    print(f"{len(chunks)} chunks from {n_opinions} opinions.")
    print(f"Chunk char length — min:{min(lens)} median:{sorted(lens)[len(lens)//2]} max:{max(lens)}")
    print(f"Saved to {out_path}")
