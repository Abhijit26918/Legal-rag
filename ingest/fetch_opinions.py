"""CourtListener live REST API — for spot-checks and small lookups only.

NOT the main ingestion path: the free tier is capped at 125 requests/day,
which can't support pulling ~5,000 opinions. Bulk ingestion and the volume
check both happen in ingest/bulk_ingest.py instead (unauthenticated,
unlimited quarterly CSV snapshots). Keep this module around for occasional
live lookups (e.g. checking a single case's current CourtListener record) —
see TECH_DESIGN.md module 1 for the full rationale.
"""
import requests

import config

SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
OPINION_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"


def _headers():
    if not config.COURTLISTENER_API_TOKEN:
        raise SystemExit(
            "COURTLISTENER_API_TOKEN not set — copy .env.example to .env "
            "and fill in a token from your CourtListener account settings."
        )
    return {"Authorization": f"Token {config.COURTLISTENER_API_TOKEN}"}


def get_opinion(opinion_id):
    """Single-opinion lookup by ID — for spot-checking a bulk-data row
    against the live record, not for bulk pulls."""
    resp = requests.get(f"{OPINION_URL}{opinion_id}/", headers=_headers())
    resp.raise_for_status()
    return resp.json()


def search(query, court=None, filed_after=None, filed_before=None):
    """One-off search, e.g. to sanity-check a doctrine keyword against the
    live index. Returns the first page only — this module deliberately
    doesn't paginate, that's what burns through the 125/day cap."""
    params = {"q": query, "type": "o"}
    if court:
        params["court"] = court
    if filed_after:
        params["filed_after"] = filed_after
    if filed_before:
        params["filed_before"] = filed_before
    resp = requests.get(SEARCH_URL, params=params, headers=_headers())
    resp.raise_for_status()
    return resp.json()
