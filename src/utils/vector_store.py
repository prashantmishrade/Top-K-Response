from pathlib import Path
from typing import List, Dict, Tuple, Optional

import json
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.ingestion import load_jsonl


try:
    import faiss
except ImportError as e:
    faiss = None


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    return np.load(embeddings_path)


def load_chunk_records(chunk_jsonl_path: Path) -> List[Dict]:
    if not chunk_jsonl_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_jsonl_path}")
    return load_jsonl(chunk_jsonl_path)


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a cosine-similarity FAISS index.
    Assumes embeddings are already normalized.
    """
    if faiss is None:
        raise ImportError("faiss is not installed. Run: pip install faiss-cpu")

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product
    index.add(embeddings.astype(np.float32))
    return index


def save_faiss_index(index, index_path: Path):
    if faiss is None:
        raise ImportError("faiss is not installed. Run: pip install faiss-cpu")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: Path):
    if faiss is None:
        raise ImportError("faiss is not installed. Run: pip install faiss-cpu")

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    return faiss.read_index(str(index_path))


def save_json(obj, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize a single embedding vector for cosine similarity search.
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Convert query text into a normalized embedding vector.
    """
    vec = model.encode([query], normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def search_chunks(
    query: str,
    model: SentenceTransformer,
    index,
    chunk_records: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Search the FAISS index and return top-k matching chunks with metadata.
    """
    if not query or not query.strip():
        return []

    if index.ntotal == 0:
        return []

    query_vec = embed_query(query, model)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        record = dict(chunk_records[idx])
        record["score"] = float(score)
        record["rank"] = len(results) + 1
        results.append(record)

    return results


def build_and_save_vector_store(
    chunk_jsonl_path: Path,
    embeddings_path: Path,
    index_path: Path,
    metadata_path: Optional[Path] = None,
):
    """
    Convenience helper to build FAISS index and save it.
    """
    chunk_records = load_chunk_records(chunk_jsonl_path)
    embeddings = load_embeddings(embeddings_path)
    index = build_faiss_index(embeddings)
    save_faiss_index(index, index_path)

    if metadata_path is not None:
        save_json(chunk_records, metadata_path)

    return index, chunk_records


def load_vector_store(
    index_path: Path,
    metadata_path: Path,
):
    """
    Load FAISS index and chunk metadata.
    """
    index = load_faiss_index(index_path)
    chunk_records = load_json(metadata_path)
    return index, chunk_records