"""LLM call + inline citation markers (TECH_DESIGN.md module 4). Two
providers supported behind one interface — Groq (free tier, no funded
Anthropic credits yet) is the current default; Anthropic (Claude) is the
planned "strong model" comparison point once credits are available. Groq's
API is OpenAI-compatible, so it reuses the already-installed `openai`
package rather than adding a new dependency.
"""
import re

import anthropic
from openai import OpenAI

import config
from generation.prompts import build_messages
from llm_utils import with_retry
from retrieval.hybrid_retrieve import hybrid_retrieve

_anthropic_client = None
_groq_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not config.ANTHROPIC_API_KEY:
            raise SystemExit("ANTHROPIC_API_KEY not set — add it to .env.")
        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise SystemExit("GROQ_API_KEY not set — add it to .env.")
        _groq_client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _groq_client


def _call_anthropic(system_prompt, user_message, model, max_tokens):
    response = with_retry(lambda: _get_anthropic_client().messages.create(
        model=model or config.GENERATION_MODEL_API,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ))
    return "".join(block.text for block in response.content if block.type == "text")


def _call_groq(system_prompt, user_message, model, max_tokens):
    response = with_retry(lambda: _get_groq_client().chat.completions.create(
        model=model or config.GENERATION_MODEL_GROQ_PRIMARY,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    ))
    return response.choices[0].message.content


_PROVIDERS = {"anthropic": _call_anthropic, "groq": _call_groq}


def generate_answer(query, chunks=None, provider=None, model=None, max_tokens=1536):
    provider = provider or config.GENERATION_PROVIDER
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r} — expected one of {list(_PROVIDERS)}")

    if chunks is None:
        chunks = hybrid_retrieve(query)
    chunks = chunks[:config.GENERATION_MAX_CONTEXT_CHUNKS]

    system_prompt, user_message, marker_map = build_messages(query, chunks)
    answer_text = _PROVIDERS[provider](system_prompt, user_message, model, max_tokens)

    # Catches ASCII [S1] as instructed, plus full-width/curly variants some
    # models drift into (【S1】, {S1}, etc.) — seen in practice with
    # openai/gpt-oss-120b on Groq — so a bracket-style slip doesn't silently
    # make every citation look unused.
    markers_used = sorted(set(re.findall(r"[\[【{]S(\d+)[\]】}]", answer_text)), key=int)
    markers_used = [f"S{n}" for n in markers_used]

    return {
        "query": query,
        "provider": provider,
        "answer": answer_text,
        "sources": marker_map,
        "markers_used": markers_used,
        "markers_unused": [m for m in marker_map if m not in markers_used],
    }


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Can police use a taser on a suspect who is already handcuffed and restrained?"
    result = generate_answer(query)

    print(f"Query: {result['query']}  (provider: {result['provider']})\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Markers used: {result['markers_used']}")
    print(f"Markers retrieved but not cited: {result['markers_unused']}\n")
    print("Sources:")
    for marker, src in result["sources"].items():
        used = "used" if marker in result["markers_used"] else "unused"
        print(f"  [{marker}] ({used}) {src['case_name']}  ({src['date_filed']})  score={src['score']:.3f}")
