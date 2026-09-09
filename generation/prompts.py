"""Prompt template for generation (TECH_DESIGN.md module 4): every factual
claim must carry an inline citation marker ([S1], [S2], ...) tied to a
specific retrieved chunk, and the model is forbidden from asserting anything
it can't tie to a marker.
"""

SYSTEM_PROMPT = """You are a legal research assistant answering questions about \
federal case law, using only the source passages provided to you below.

Rules:
1. Every factual or legal claim in your answer must end with a citation \
marker referencing the specific source passage that supports it. Use \
plain ASCII square brackets exactly in this form: [S1] or [S2] — not full-width \
or curly brackets, not parentheses, nothing else. A sentence can carry more \
than one marker if it draws on multiple sources, e.g. "...excessive force. [S3][S8]"
2. Do not state anything as fact that isn't directly supported by one of \
the numbered source passages. If the passages don't contain enough \
information to answer the question, say so explicitly rather than filling \
the gap from general knowledge.
3. Do not treat any cited case as necessarily still good law — you are only \
summarizing what these passages say, not verifying whether the holdings \
have since been overruled or distinguished. A separate check handles that.
4. Write for a reader who is not a lawyer, but do not oversimplify the \
legal substance — accuracy matters more than brevity.
"""

USER_TEMPLATE = """Question: {query}

Source passages:

{sources}

Answer the question using only these passages, with inline [S#] citation \
markers on every claim."""


def format_sources(chunks):
    """Numbers the retrieved chunks as [S1], [S2], ... and returns both the
    formatted block for the prompt and a marker -> chunk metadata mapping
    (needed later to resolve a marker back to its source passage for
    display/highlighting/attribution checking)."""
    blocks = []
    marker_map = {}
    for i, chunk in enumerate(chunks, start=1):
        marker = f"S{i}"
        meta = chunk["metadata"]
        blocks.append(
            f"[{marker}] {meta['case_name']} ({meta['date_filed']})\n{chunk['text']}"
        )
        marker_map[marker] = {
            "chunk_id": chunk["chunk_id"],
            "opinion_id": meta["opinion_id"],
            "case_name": meta["case_name"],
            "date_filed": meta["date_filed"],
            "text": chunk["text"],
            "score": chunk["score"],
            "source": chunk["source"],
        }
    return "\n\n".join(blocks), marker_map


def build_messages(query, chunks):
    sources_block, marker_map = format_sources(chunks)
    user_message = USER_TEMPLATE.format(query=query, sources=sources_block)
    return SYSTEM_PROMPT, user_message, marker_map
