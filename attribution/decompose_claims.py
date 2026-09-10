"""Answer -> atomic claims (TECH_DESIGN.md module 5). Legal sentences bundle
multiple assertions ("Officer X tased the suspect after he was handcuffed,
which the court held excessive [S3]" is really two claims), so this uses an
LLM-prompted decomposition rather than sentence-splitting — each atomic
claim keeps the citation marker(s) it depends on.
"""
import json
import re

from openai import OpenAI

import config
from llm_utils import with_retry

SYSTEM_PROMPT = """You decompose a legal answer into atomic claims for fact-checking.

The answer contains inline citation markers like [S1], [S2] after each \
sentence or clause. Break the answer into the smallest individual factual \
or legal assertions — a sentence with two assertions ("X did Y, which the \
court held Z [S1]") should become two separate claims, not one.

For each atomic claim, keep the citation marker(s) it depends on. If a claim \
has no marker anywhere near it in the source text, its "markers" list should \
be empty — this is important, do not invent a marker for an unmarked claim.

Also classify each claim's type:
- "factual": a specific factual or legal assertion — what a case held, what \
a rule requires, what happened in a dispute. These need a citation to be \
trustworthy.
- "synthesis": a framing, transition, or summary statement that organizes \
or combines claims made elsewhere in the answer, without asserting any new \
fact of its own (e.g. an opening framing sentence, "taken together, these \
cases show...", a concluding restatement). These are expected to lack their \
own citation marker — that's normal, not a defect, since they don't assert \
anything not already covered elsewhere.

Be conservative about labeling something "synthesis" — if it states a \
specific new fact, holding, or rule (even if phrased as a summary), it's \
"factual", not "synthesis".

Return ONLY a JSON array, no other text, no markdown code fence. Each \
element: {"claim": "<atomic claim text>", "markers": ["S1", ...], "claim_type": "factual"}"""

USER_TEMPLATE = """Decompose this answer into atomic claims:

{answer}"""

_client = None


def _get_client():
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise SystemExit("GROQ_API_KEY not set — add it to .env.")
        _client = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _client


def _extract_json_array(text):
    """Models sometimes wrap JSON in a markdown fence despite instructions
    not to — strip that before parsing rather than failing outright."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return json.loads(text)


def decompose_claims(answer_text, model=None, max_tokens=6000):
    """max_tokens=6000: openai/gpt-oss-120b spends a highly variable amount
    on hidden reasoning before the visible JSON (observed 1,297 of 1,797
    tokens on one real answer) — 2048 occasionally ran completely dry with
    no content at all. Input here is small (~600 tokens), so there's ample
    room under Groq's 8,000 TPM free-tier limit even at this budget."""
    client = _get_client()

    def _call_and_parse():
        response = client.chat.completions.create(
            model=model or config.GENERATION_MODEL_GROQ_PRIMARY,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(answer=answer_text)},
            ],
        )
        raw = response.choices[0].message.content
        try:
            return _extract_json_array(raw)
        except (json.JSONDecodeError, AttributeError) as e:
            # Retriable: see attribution/entailment.py's same pattern —
            # retrying only the API call would keep retrying a call that
            # already succeeded-but-empty, never getting a fresh sample.
            raise ValueError(f"Could not parse claim decomposition as JSON. Raw output:\n{raw}") from e

    claims = with_retry(_call_and_parse)

    for c in claims:
        if "claim" not in c or "markers" not in c or "claim_type" not in c:
            raise ValueError(f"Malformed claim object (missing 'claim'/'markers'/'claim_type'): {c}")
        if c["claim_type"] not in ("factual", "synthesis"):
            raise ValueError(f"Unexpected claim_type {c['claim_type']!r} (expected 'factual' or 'synthesis'): {c}")
    return claims


if __name__ == "__main__":
    _TEST_CASES = [
        (
            "Officers may not use a taser on a suspect who is already handcuffed and "
            "not resisting [S3]. The Ninth Circuit has held this constitutes excessive "
            "force under the Fourth Amendment [S3][S7].",
            2,  # expect roughly 2 atomic claims
        ),
        (
            "Qualified immunity shields officers unless the right violated was clearly "
            "established at the time of the conduct [S1].",
            1,
        ),
    ]
    for answer, expected_n in _TEST_CASES:
        claims = decompose_claims(answer)
        print(f"Input: {answer}")
        print(f"-> {len(claims)} claims (expected ~{expected_n}):")
        for c in claims:
            print(f"   {c}")
        print()
