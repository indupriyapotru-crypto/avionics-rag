"""
retriever.py – TF-IDF + FAISS vector store for semantic article retrieval.

Includes query expansion so short/specific terms like "radar" or "GPS"
automatically broaden to related aviation keywords, improving recall.
"""

import logging

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Query expansion map
# Short or specific queries are expanded with closely related aviation terms
# so TF-IDF finds relevant chunks even when exact words don't appear.
# ─────────────────────────────────────────────────────────────────────────────
_EXPANSIONS: dict[str, str] = {
    "radar":         "radar weather surveillance TCAS transponder detection tracking",
    "gps":           "GPS navigation GNSS satellite positioning waypoint",
    "autopilot":     "autopilot autoflight automation AFCS fly-by-wire",
    "display":       "display HUD EFIS MFD PFD cockpit screen avionics",
    "communication": "communication VHF radio datalink ACARS SATCOM",
    "engine":        "engine FADEC thrust turbine powerplant propulsion",
    "fuel":          "fuel efficiency consumption SAF sustainable aviation",
    "safety":        "safety certification airworthiness FAA EASA regulation",
    "drone":         "drone UAV UAS autonomous unmanned RPAS",
    "battery":       "battery electric hybrid propulsion eVTOL power",
    "maintenance":   "maintenance MRO inspection repair overhaul",
    "software":      "software update firmware cybersecurity avionics system",
    "sensor":        "sensor lidar camera infrared detection obstacle",
    "landing":       "landing ILS approach autoland runway guidance",
    "weather":       "weather turbulence icing wind shear forecast avoidance",
    "military":      "military defense fighter jet UAV stealth avionics",
    "connectivity":  "connectivity wifi passenger inflight internet satellite",
    "certification": "certification FAA EASA DO-178 DO-254 approval",
    "nextgen":       "NextGen ADS-B air traffic management ATM modernisation",
    "evtol":         "eVTOL urban air mobility AAM electric vertical takeoff",
}


def _expand_query(query: str) -> str:
    """
    Append related terms to short queries so TF-IDF finds more relevant chunks.
    Matches on lowercase words in the query against the expansion map.
    """
    q_lower = query.lower()
    extras  = []
    for keyword, expansion in _EXPANSIONS.items():
        if keyword in q_lower:
            extras.append(expansion)
    if extras:
        expanded = f"{query} {' '.join(extras)}"
        log.info("Query expanded: %r → %r", query, expanded)
        return expanded
    return query


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype("float32")


class VectorStore:
    """In-memory TF-IDF + FAISS cosine-similarity store."""

    def __init__(self):
        self.index: faiss.IndexFlatL2 | None = None
        self.docs:  list[dict]               = []
        self._vec:  TfidfVectorizer | None   = None

    # ── build ────────────────────────────────────────────────────────────────

    def build(self, docs: list[dict]) -> None:
        """Fit the vectoriser and index all document chunks."""
        if not docs:
            raise ValueError("Cannot build VectorStore from an empty document list.")

        texts     = [d["text"] for d in docs]
        self._vec = TfidfVectorizer(
            stop_words="english",
            max_features=20_000,
            ngram_range=(1, 2),
        )
        X          = self._vec.fit_transform(texts).toarray().astype("float32")
        embeddings = _normalize(X)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.docs  = docs
        log.info("VectorStore built: %d chunks, dim=%d", len(docs), embeddings.shape[1])

    # ── search ───────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Return the top-k most relevant chunks for a free-text query.
        Short or domain-specific queries are auto-expanded before retrieval.
        """
        if self.index is None or self._vec is None:
            raise RuntimeError("Call build(docs) before search().")

        k             = min(k, len(self.docs))
        expanded      = _expand_query(query)
        q_vec         = self._vec.transform([expanded]).toarray().astype("float32")
        q_emb         = _normalize(q_vec)
        _, I          = self.index.search(q_emb, k)
        return [self.docs[i] for i in I[0] if i < len(self.docs)]
