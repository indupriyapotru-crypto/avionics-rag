"""
config.py – Central configuration for Avionics News RAG
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── RSS Feed Sources ──────────────────────────────────────────────────────────
RSS_FEEDS: dict[str, str] = {
    "AIN Online":     "https://www.ainonline.com/rss.xml",
    "AeroTime":       "https://www.aerotime.aero/feed",
    "Aviation Week":  "https://aviationweek.com/rss.xml",
    "AVweb":          "https://avweb.com/feed/",
    "Aviation Today": "https://www.aviationtoday.com/feed/",
    "Simple Flying":  "https://simpleflying.com/feed/",
    "FlightGlobal":   "https://www.flightglobal.com/rss/",
}

# ── Ingestion settings ────────────────────────────────────────────────────────
# How many hours back to start looking. If nothing is found, the window
# automatically expands (see FALLBACK_WINDOWS_HOURS below).
INITIAL_WINDOW_HOURS: int = 24

# Fallback ladder: if the current window yields 0 articles, try the next value.
# [24h → 48h → 72h → 7 days → 30 days → no limit]
FALLBACK_WINDOWS_HOURS: list[int] = [24, 48, 72, 168, 720, 0]

# Maximum articles to pull per feed
MAX_ARTICLES_PER_FEED: int = 20

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int    = 500
CHUNK_OVERLAP: int = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = 5

# ── Groq model ────────────────────────────────────────────────────────────────
GROQ_MODEL: str = "llama-3.1-8b-instant"
