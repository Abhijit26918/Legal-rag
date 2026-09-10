"""Shared retry wrapper for LLM API calls. Full pipeline runs (generation +
decomposition + entailment + currency check) make dozens of sequential Groq
calls — a single transient timeout shouldn't kill the whole run."""
import time


def with_retry(fn, max_attempts=3, backoff_seconds=2):
    last_error = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(backoff_seconds * (attempt + 1))
    raise last_error
