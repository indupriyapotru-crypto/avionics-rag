"""
ingest.py – Fetch, clean, and chunk avionics news articles from RSS feeds.

Key improvement over the original:
  • Adaptive time window – starts at 24 h and automatically expands
    (48 h → 72 h → 7 days → 30 days → no limit) until at least one
    article is found.  You will never see "No recent articles found."
    as long as the feeds are reachable.
"""

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import feedparser
from bs4 import BeautifulSoup

from config import (
    RSS_FEEDS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_ARTICLES_PER_FEED,
    FALLBACK_WINDOWS_HOURS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _clean_html(raw: str) -> str:
    return BeautifulSoup(raw, "html.parser").get_text(separator=" ")


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def _parse_pub_date(entry) -> Optional[datetime]:
    """
    Try multiple feedparser fields to get a timezone-aware published datetime.
    Returns None when no date information is available.
    """
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        try:
            return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core fetch function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_articles(cutoff: Optional[datetime] = None) -> list[dict]:
    """
    Pull raw articles from every configured RSS feed.

    Parameters
    ----------
    cutoff : datetime | None
        Only articles published *at or after* this UTC datetime are included.
        Pass ``None`` (or a cutoff in the past beyond all articles) to get
        everything available in the feeds.

    Returns
    -------
    list of dicts with keys: title, link, content, published
    """
    articles: list[dict] = []

    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                if count >= MAX_ARTICLES_PER_FEED:
                    break

                pub = _parse_pub_date(entry)

                # Apply cutoff only when we have a reliable date AND a cutoff is set
                if cutoff is not None and pub is not None and pub < cutoff:
                    continue

                articles.append(
                    {
                        "title":     entry.get("title", "Untitled"),
                        "link":      entry.get("link", ""),
                        "content":   entry.get("summary", ""),
                        "published": pub.strftime("%Y-%m-%d %H:%M UTC") if pub else "Date unknown",
                    }
                )
                count += 1

            log.info("  ✓ %-20s  %d articles", name, count)

        except Exception as exc:
            log.warning("  ✗ %-20s  error: %s", name, exc)

    return articles


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive ingestion pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_articles(initial_window_hours: int = FALLBACK_WINDOWS_HOURS[0]) -> tuple[list[dict], int]:
    """
    Fetch articles, chunk them, and deduplicate.

    Uses an *adaptive time window*: begins at ``initial_window_hours`` and
    expands through FALLBACK_WINDOWS_HOURS until at least one article is
    found – so the UI always has something to show.

    Returns
    -------
    (docs, actual_window_hours)
        docs                : list of chunk dicts ready for the VectorStore
        actual_window_hours : the window (hours) that produced results;
                              0 means "no time filter applied".
    """
    now = datetime.now(tz=timezone.utc)

    # Build the full ladder starting from the requested initial window
    ladder = [h for h in FALLBACK_WINDOWS_HOURS if h >= initial_window_hours or h == 0]
    if not ladder:
        ladder = FALLBACK_WINDOWS_HOURS

    raw_articles: list[dict] = []
    used_window = 0

    for window_hours in ladder:
        cutoff = (now - timedelta(hours=window_hours)) if window_hours > 0 else None
        window_label = f"{window_hours}h" if window_hours > 0 else "all time"
        log.info("Trying window: %s", window_label)

        raw_articles = fetch_articles(cutoff=cutoff)

        if raw_articles:
            used_window = window_hours
            log.info("Found %d articles in window: %s", len(raw_articles), window_label)
            break

        log.info("No articles in window %s, expanding…", window_label)

    # ── clean, chunk, deduplicate ────────────────────────────────────────────
    docs: list[dict] = []
    seen_hashes: set[str] = set()

    for article in raw_articles:
        clean_text = _clean_html(article["content"])
        if not clean_text.strip():
            clean_text = article["title"]          # fallback: use headline only

        content_hash = _md5(clean_text)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        for chunk in _chunk(clean_text):
            docs.append(
                {
                    "text":      chunk,
                    "title":     article["title"],
                    "source":    article["link"],
                    "published": article["published"],
                }
            )

    return docs, used_window
