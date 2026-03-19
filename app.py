"""
app.py – Streamlit web interface for the Avionics News RAG summariser.
"""

import logging
import streamlit as st

from ingest import process_articles
from retriever import VectorStore
from summarizer import summarize
from config import FALLBACK_WINDOWS_HOURS, TOP_K

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="✈️ Avionics News RAG",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    window_options = {
        "Last 24 hours":  24,
        "Last 48 hours":  48,
        "Last 3 days":    72,
        "Last 7 days":    168,
        "Last 30 days":   720,
        "All available":  0,
    }
    selected_window_label = st.selectbox(
        "News window",
        list(window_options.keys()),
        index=0,
        help=(
            "Starting search window. If nothing is found, the agent "
            "automatically widens the window until articles are available."
        ),
    )
    initial_window = window_options[selected_window_label]

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=3,
        max_value=15,
        value=TOP_K,
        help="More chunks = richer context but slower summary.",
    )

    st.markdown("---")
    st.markdown(
        "**RSS Sources**\n"
        "- AIN Online\n"
        "- AeroTime\n"
        "- Aviation Week\n"
        "- AVweb\n"
        "- Aviation Today\n"
        "- Simple Flying\n"
        "- FlightGlobal"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("✈️ Avionics News Summariser")
st.caption(
    "RAG-powered daily digest — auto-expands time window so you always get news."
)

query = st.text_input(
    "🔍 Search query",
    "latest avionics news",
    help="Type any topic — e.g. 'radar', 'GPS', 'autopilot', 'eVTOL'. "
         "Short terms are auto-expanded to find the best matching articles.",
)

run_btn = st.button("🚀 Fetch & Summarise", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _window_label(hours: int) -> str:
    if hours == 0:
        return "all available articles (no time limit applied)"
    if hours < 48:
        return f"last {hours} hours"
    if hours < 168:
        return f"last {hours // 24} days"
    return f"last {hours // 24} days"


if run_btn:
    # Step 1 – Ingest
    with st.status("📡 Fetching articles from RSS feeds…", expanded=True) as status:
        docs, used_window = process_articles(initial_window_hours=initial_window)

        if not docs:
            status.update(label="❌ Could not fetch any articles.", state="error")
            st.error(
                "No articles could be loaded from any RSS feed.\n\n"
                "Please check:\n"
                "- Your internet connection\n"
                "- The feed URLs in `config.py`"
            )
            st.stop()

        # Show which window was actually used
        if used_window != initial_window and used_window != 0:
            st.info(
                f"⚡ No articles found in **{selected_window_label.lower()}**. "
                f"Auto-expanded to **{_window_label(used_window)}** — "
                f"found **{len(docs)} chunks**."
            )
        elif used_window == 0:
            st.info(
                f"⚡ Auto-expanded to **all available articles** (no time limit) — "
                f"found **{len(docs)} chunks**."
            )
        else:
            st.success(
                f"✅ Loaded **{len(docs)} chunks** from the {_window_label(used_window)}."
            )

        status.update(label=f"✅ {len(docs)} chunks ready", state="complete")

    # Step 2 – Build vector store
    with st.spinner("🔢 Building vector store…"):
        vs = VectorStore()
        vs.build(docs)

    # Step 3 – Retrieve + summarise
    with st.spinner("🤖 Generating AI summary…"):
        results = vs.search(query, k=top_k)
        summary = summarize(results, topic=query)

    # ── Output ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Summary")
    st.write(summary)

    # Sources — deduplicated, rendered once from metadata (not from LLM text)
    st.markdown("---")
    st.subheader("🔗 Sources")
    seen: set[str] = set()
    for r in results:
        src = r.get("source", "")
        if not src or src in seen:
            continue
        seen.add(src)
        title     = r.get("title", src)
        published = r.get("published", "")
        col_a, col_b = st.columns([5, 1])
        with col_a:
            st.markdown(f"[{title}]({src})")
        with col_b:
            if published:
                st.caption(published)

    # ── Expander: raw chunks ─────────────────────────────────────────────────
    with st.expander("🔍 View retrieved chunks"):
        for i, r in enumerate(results, 1):
            st.markdown(f"**Chunk {i} — {r['title']}**")
            st.text(r["text"][:400] + ("…" if len(r["text"]) > 400 else ""))
            st.markdown("---")
