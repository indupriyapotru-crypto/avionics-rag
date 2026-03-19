"""
main.py – Command-line entry point for the Avionics News RAG agent.
"""

import logging
import sys

from ingest import process_articles
from retriever import VectorStore
from summarizer import summarize
from config import TOP_K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _window_label(hours: int) -> str:
    if hours == 0:
        return "all available articles (no time limit)"
    if hours < 48:
        return f"last {hours} hours"
    if hours < 168:
        return f"last {hours // 24} days"
    if hours < 720:
        return f"last {hours // 24} days ({hours // 168} week{'s' if hours >= 336 else ''})"
    return f"last ~{hours // 720} month(s)"


def main(query: str = "latest avionics developments") -> None:
    log.info("─── Avionics News RAG Agent ───────────────────────────────")

    log.info("Step 1/4  Fetching & processing articles …")
    docs, window_hours = process_articles()

    if not docs:
        log.error(
            "No articles could be fetched from any RSS feed. "
            "Check your internet connection or the feed URLs in config.py."
        )
        sys.exit(1)

    log.info(
        "Step 2/4  Indexed %d chunks from the %s.",
        len(docs),
        _window_label(window_hours),
    )

    log.info("Step 3/4  Building vector store …")
    vs = VectorStore()
    vs.build(docs)

    log.info("Step 4/4  Searching and summarising for query: %r", query)
    results = vs.search(query, k=TOP_K)
    summary = summarize(results, topic=query)

    # ── pretty print ──────────────────────────────────────────────────────────
    separator = "═" * 60
    print(f"\n{separator}")
    print("  ✈️  AVIONICS NEWS SUMMARY")
    print(f"  Window: {_window_label(window_hours)}")
    print(separator)
    print(summary)
    print(f"\n{separator}")

    print("\n📰 SOURCES:")
    seen: set[str] = set()
    for r in results:
        src = r.get("source", "")
        if src and src not in seen:
            seen.add(src)
            print(f"  • {r['title']}")
            print(f"    {src}")
            print(f"    Published: {r.get('published','unknown')}")


if __name__ == "__main__":
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "latest avionics developments"
    main(query=user_query)
