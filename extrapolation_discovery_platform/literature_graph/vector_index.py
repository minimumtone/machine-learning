"""
Vector Index Abstraction + FAISS Backend (MVP)
ベクトルインデックス

Provides a ``VectorIndex`` abstract interface and a concrete ``FAISSIndex``
implementation.  The abstraction allows future migration to Milvus / Weaviate
without changing calling code.

Usage
-----
::

    idx = FAISSIndex(dim=384)
    idx.add(ids=["wf1", "wf2"], vectors=np.array(...))
    results = idx.search(query_vector, top_k=10)

For the MVP we use a simple sentence-transformer model to generate embeddings
from workflow_text.  If sentence-transformers is not installed, a lightweight
TF-IDF fallback is provided.
"""

from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract VectorIndex
# ---------------------------------------------------------------------------

class VectorIndex(abc.ABC):
    """Abstract vector index interface.

    Contract
    --------
    - ``add(ids, vectors)`` inserts vectors with string IDs.
    - ``search(query, top_k)`` returns (ids, distances).
    - ``save(path)`` / ``load(path)`` for persistence.
    """

    @abc.abstractmethod
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        ...

    @abc.abstractmethod
    def search(
        self, query: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Return list of (id, distance) sorted by ascending distance."""
        ...

    @abc.abstractmethod
    def save(self, directory: Path) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, directory: Path) -> "VectorIndex":
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...


# ---------------------------------------------------------------------------
# FAISS Backend
# ---------------------------------------------------------------------------

class FAISSIndex(VectorIndex):
    """FAISS-based vector index using L2 distance.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    """

    def __init__(self, dim: int) -> None:
        try:
            import faiss  # type: ignore
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISSIndex. "
                "Install with: pip install faiss-cpu"
            )
        self._dim = dim
        self._index = faiss.IndexFlatL2(dim)
        self._ids: List[str] = []

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"Expected vectors of shape (N, {self._dim}), got {vectors.shape}"
            )
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of ids ({len(ids)}) != number of vectors ({vectors.shape[0]})"
            )
        self._index.add(vectors.astype(np.float32))
        self._ids.extend(ids)

    def search(
        self, query: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        k = min(top_k, len(self._ids))
        if k == 0:
            return []
        distances, indices = self._index.search(query.astype(np.float32), k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self._ids):
                results.append((self._ids[idx], float(dist)))
        return results

    def save(self, directory: Path) -> None:
        import faiss  # type: ignore
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "workflow_vectors.faiss"))
        meta_path = directory / "workflow_vectors.meta.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for wf_id in self._ids:
                f.write(json.dumps({"workflow_id": wf_id}) + "\n")
        logger.info("Saved FAISS index (%d vectors) to %s", len(self._ids), directory)

    @classmethod
    def load(cls, directory: Path) -> "FAISSIndex":
        import faiss  # type: ignore
        directory = Path(directory)
        index = faiss.read_index(str(directory / "workflow_vectors.faiss"))
        dim = index.d
        obj = cls.__new__(cls)
        obj._dim = dim
        obj._index = index
        obj._ids = []
        meta_path = directory / "workflow_vectors.meta.jsonl"
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj._ids.append(json.loads(line)["workflow_id"])
        logger.info("Loaded FAISS index (%d vectors, dim=%d) from %s", len(obj._ids), dim, directory)
        return obj

    def __len__(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# Numpy Fallback (no FAISS required)
# ---------------------------------------------------------------------------

class NumpyFlatIndex(VectorIndex):
    """Pure-numpy brute-force index as fallback when FAISS is unavailable.

    Suitable for small corpora (< 1000 vectors).
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"Expected vectors of shape (N, {self._dim}), got {vectors.shape}"
            )
        if self._vectors is None:
            self._vectors = vectors.astype(np.float32)
        else:
            self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._ids.extend(ids)

    def search(
        self, query: np.ndarray, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        if self._vectors is None or len(self._ids) == 0:
            return []
        if query.ndim == 1:
            query = query.reshape(1, -1)
        # L2 distance
        diff = self._vectors - query.astype(np.float32)
        dists = np.sum(diff ** 2, axis=1)
        k = min(top_k, len(self._ids))
        top_indices = np.argpartition(dists, k)[:k]
        top_indices = top_indices[np.argsort(dists[top_indices])]
        return [(self._ids[i], float(dists[i])) for i in top_indices]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(str(directory / "workflow_vectors.npy"), self._vectors)
        meta_path = directory / "workflow_vectors.meta.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for wf_id in self._ids:
                f.write(json.dumps({"workflow_id": wf_id}) + "\n")
        logger.info("Saved NumpyFlatIndex (%d vectors) to %s", len(self._ids), directory)

    @classmethod
    def load(cls, directory: Path) -> "NumpyFlatIndex":
        directory = Path(directory)
        vectors = np.load(str(directory / "workflow_vectors.npy"))
        dim = vectors.shape[1]
        obj = cls(dim=dim)
        obj._vectors = vectors
        meta_path = directory / "workflow_vectors.meta.jsonl"
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj._ids.append(json.loads(line)["workflow_id"])
        logger.info("Loaded NumpyFlatIndex (%d vectors, dim=%d)", len(obj._ids), dim)
        return obj

    def __len__(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

# Module-level cache for TF-IDF vectorizer so queries use the same vocabulary
_tfidf_vectorizer: Any = None


def _tfidf_embed(
    texts: List[str],
    max_features: int = 512,
    *,
    fit: bool = True,
) -> np.ndarray:
    """Lightweight TF-IDF embedding fallback.

    Parameters
    ----------
    texts : list of str
    max_features : int
    fit : bool
        If True, fit a new vectorizer on *texts* (corpus mode).
        If False, transform *texts* using the previously fitted vectorizer
        (query mode).  Falls back to fitting if no vectorizer exists.

    Returns
    -------
    np.ndarray of shape (N, dim)
    """
    global _tfidf_vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    if fit or _tfidf_vectorizer is None:
        _tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        matrix = _tfidf_vectorizer.fit_transform(texts).toarray().astype(np.float32)
    else:
        matrix = _tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
    return matrix


def embed_workflow_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    *,
    fit: bool = True,
) -> np.ndarray:
    """Generate embeddings for a list of workflow texts.

    Tries sentence-transformers first, falls back to TF-IDF.

    Parameters
    ----------
    texts : list of str
        Canonical workflow text strings.
    model_name : str
        Sentence-transformer model name (only used if available).

    Returns
    -------
    np.ndarray of shape (N, dim)
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name)
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        logger.info(
            "Embedded %d texts with SentenceTransformer(%s), dim=%d",
            len(texts), model_name, vectors.shape[1],
        )
        return vectors.astype(np.float32)
    except ImportError:
        logger.warning(
            "sentence-transformers not available. Falling back to TF-IDF embeddings."
        )
        vectors = _tfidf_embed(texts, fit=fit)
        logger.info("Embedded %d texts with TF-IDF, dim=%d", len(texts), vectors.shape[1])
        return vectors


def build_index(
    workflow_ids: List[str],
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    use_faiss: bool = True,
) -> VectorIndex:
    """Build a vector index from workflow texts.

    Parameters
    ----------
    workflow_ids : list of str
    texts : list of str
    model_name : str
    use_faiss : bool
        If True, try FAISS; fall back to NumpyFlatIndex.

    Returns
    -------
    VectorIndex
    """
    vectors = embed_workflow_texts(texts, model_name=model_name)
    dim = vectors.shape[1]

    if use_faiss:
        try:
            index = FAISSIndex(dim=dim)
        except ImportError:
            logger.warning("FAISS not available, using NumpyFlatIndex fallback.")
            index = NumpyFlatIndex(dim=dim)
    else:
        index = NumpyFlatIndex(dim=dim)

    index.add(workflow_ids, vectors)
    return index
