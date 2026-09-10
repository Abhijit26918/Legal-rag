"""Single place for paths, model names, and tunable thresholds.

Nothing below should be hardcoded again inside ingest/, citations/,
retrieval/, etc. — import from here instead.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Scope ---
COURT_ID = "ca9"  # CourtListener court identifier for the 9th Circuit
DOCTRINE_QUERY = "qualified immunity excessive force"
DATE_START = "2005-01-01"
DATE_END = "2025-01-01"
TARGET_OPINION_COUNT = 5000

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SQLITE_PATH = DATA_DIR / "opinions.db"
GRAPH_PATH = DATA_DIR / "citation_graph.graphml"
CHROMA_DIR = DATA_DIR / "chroma"

# --- Bulk data (primary ingestion path — see TECH_DESIGN.md module 1) ---
# Public S3 bucket, no auth, no rate limit. Files are quarterly snapshots;
# BULK_TABLES gives the base table names, the actual dated filename is
# discovered at runtime (see ingest/bulk_ingest.py:latest_bulk_file).
BULK_S3_BASE = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com"
BULK_TABLES = ["courts", "dockets", "opinion-clusters", "citations", "opinions"]

# --- API keys (set these in a .env file, never commit it) ---
COURTLISTENER_API_TOKEN = os.environ.get("COURTLISTENER_API_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Retrieval ---
CHUNK_TOKEN_SIZE = 300
CHUNK_TOKEN_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI — 1536 dims, $0.02/1M tokens
EMBEDDING_DIMENSIONS = 1536
RETRIEVAL_TOP_K = 10
GRAPH_EXPANSION_HOPS = 1
MAX_CANDIDATE_CHUNKS = 20

# --- Generation ---
# Primary generation right now runs on Groq (free tier, OpenAI-compatible
# API) rather than Anthropic — no funded API credits yet, and this machine
# has no CUDA so running a local model would be impractically slow anyway.
# GENERATION_MODEL_API (Claude) stays the target "strong model" comparison
# point once credits are available; GENERATION_MODEL_GROQ_SMALL is the
# deliberate "small model" comparison point from the original plan, also
# served via Groq instead of local inference for the same CPU-only reason.
GENERATION_PROVIDER = "groq"  # "groq" or "anthropic"
# Model names confirmed against this account's actual /models list (Groq's
# lineup changes over time — llama-3.3-70b-versatile / llama-3.1-8b-instant,
# assumed from general knowledge, turned out not to be available here;
# openai/gpt-oss-* are OpenAI's open-weight models, hosted on Groq).
GENERATION_MODEL_GROQ_PRIMARY = "openai/gpt-oss-120b"
GENERATION_MODEL_GROQ_SMALL = "openai/gpt-oss-20b"
GENERATION_MODEL_API = "claude-sonnet-5"
# Groq's free tier caps openai/gpt-oss-120b at 8,000 tokens/minute per
# request — the full MAX_CANDIDATE_CHUNKS=20 context (system prompt + 20
# chunks + max_tokens headroom) went over that by ~270 tokens on a real
# query. Generation uses a smaller slice of the retrieved chunks than
# retrieval/currency-checking do; raise this if/when on a paid tier.
GENERATION_MAX_CONTEXT_CHUNKS = 12

# --- Attribution ---
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# --- Precedent currency ---
CURRENCY_TRAVERSAL_DEPTH = 2
STALE_TREATMENTS = {"overruled", "superseded", "abrogated"}
# CourtListener's live API (used to fetch text for external citing opinions
# — ones outside our 1,696-opinion corpus) is capped at 125 requests/day
# (see ingest/fetch_opinions.py). A heavily-cited landmark case can have
# dozens of external forward-citers; capping per-call keeps one currency
# check from burning the whole daily budget. Internal citers cost nothing
# (text already local) and are never capped.
CURRENCY_MAX_EXTERNAL_LOOKUPS = 15
