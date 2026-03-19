# ✈️ Avionics News RAG

> A Retrieval-Augmented Generation (RAG) pipeline that aggregates live avionics and aerospace news from multiple RSS feeds, retrieves the most relevant articles for your query, and generates a structured summary using an LLM — available as both a **CLI tool** and a **Streamlit web app**.

---


## 🌟 Features

| Feature | Details |
|---|---|
| **7 live RSS feeds** | AIN Online, AeroTime, Aviation Week, AVweb, Aviation Today, Simple Flying, FlightGlobal |
| **Adaptive time window** | Starts at 24 h; auto-expands (48 h → 72 h → 7 days → 30 days → all time) so you **always get news** |
| **TF-IDF + FAISS retrieval** | Fast, fully local semantic search — no paid embedding API required |
| **Streamlit UI** | Clean web interface with sidebar controls and source attribution |
| **CLI mode** | Pipe-friendly terminal output for automation / cron jobs |
| **Zero cloud costs** | Only the Groq API call costs anything (free tier is generous) |

---

## 📁 Project Structure

```
avionics-news-rag/
├── app.py            # Streamlit web interface
├── main.py           # CLI entry point
├── ingest.py         # RSS fetch + adaptive time window + chunking
├── retriever.py      # TF-IDF vectoriser + FAISS index
├── summarizer.py     # Groq/LLaMA summarisation
├── config.py         # All settings in one place
├── requirements.txt
├── .env.example      # Template for your API key
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/indupriyapotru-crypto/avionics-rag.git
cd avionics-rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key

```bash
cp .env.example .env
# Open .env and paste your key:
# GROQ_API_KEY=gsk_...
```

> 🔑 Get a **free** Groq API key at [console.groq.com](https://console.groq.com).

### 5. Run

**Streamlit web app:**
```bash
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Command-line:**
```bash
# Default query
python main.py

# Custom query
python main.py "ADS-B NextGen air traffic management"
```

---

## 🔧 Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `INITIAL_WINDOW_HOURS` | `24` | Starting time window for article search |
| `FALLBACK_WINDOWS_HOURS` | `[24, 48, 72, 168, 720, 0]` | Expansion ladder (0 = no limit) |
| `MAX_ARTICLES_PER_FEED` | `20` | Max articles pulled per RSS source |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | LLM model used for summarisation |

---

## 🏗️ Architecture

```
RSS Feeds (7 sources)
        │
        ▼
  [ ingest.py ]
  Adaptive time window → fetch → clean HTML → chunk → deduplicate
        │
        ▼
  [ retriever.py ]
  TF-IDF vectorise → FAISS index → cosine-similarity search
        │
        ▼
  [ summarizer.py ]
  Top-K chunks → Groq (LLaMA 3.1) → Structured markdown report
        │
        ├──► app.py   (Streamlit UI)
        └──► main.py  (CLI)
```

### Why TF-IDF + FAISS instead of a neural embedder?

- **Zero setup** — no GPU, no embedding API key, no download of a large model
- **Fast** — builds in < 1 s on a laptop for typical article volumes
- **Good enough** — for keyword-heavy aviation domain text, TF-IDF bigrams perform well

If you want neural embeddings (better semantic recall), swap the `VectorStore` for one backed by `sentence-transformers` — the interface is identical.

---

## 📡 RSS Feed Sources

| Name | URL | Focus |
|---|---|---|
| AIN Online | ainonline.com | Business aviation & avionics |
| AeroTime | aerotime.aero | Airline & MRO news |
| Aviation Week | aviationweek.com | Defence & aerospace technology |
| AVweb | avweb.com | GA pilots & avionics |
| Aviation Today | aviationtoday.com | Avionics & communications |
| Simple Flying | simpleflying.com | Commercial aviation |
| FlightGlobal | flightglobal.com | Global aerospace |

To add more sources, simply add an entry to the `RSS_FEEDS` dict in `config.py`.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) — blazing-fast LLM inference
- [feedparser](https://feedparser.readthedocs.io) — robust RSS parsing
- [FAISS](https://github.com/facebookresearch/faiss) — efficient vector search
- [Streamlit](https://streamlit.io) — rapid ML web apps
