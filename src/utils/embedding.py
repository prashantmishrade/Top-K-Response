from pathlib import Path
from typing import List, Tuple, Optional

import json
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.ingestion import load_jsonl, save_jsonl


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Load a sentence-transformers embedding model.
    """
    return SentenceTransformer(model_name)


def get_texts_from_chunk_records(chunk_records: List[dict]) -> List[str]:
    """
    Extract chunk_text values from chunk records.
    """
    texts = []
    for record in chunk_records:
        text = record.get("chunk_text", "")
        if text and text.strip():
            texts.append(text.strip())
        else:
            texts.append("")
    return texts


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Convert a list of texts into embeddings.
    """
    if not texts:
        return np.array([])

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(embeddings)


def save_embeddings(embeddings: np.ndarray, output_path: Path):
    """
    Save embeddings to a .npy file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)


def load_embeddings(input_path: Path) -> np.ndarray:
    """
    Load embeddings from a .npy file.
    """
    return np.load(input_path)


def process_embeddings(
    chunk_jsonl_path: Path,
    embeddings_output_path: Path,
    enriched_chunks_output_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> Tuple[List[dict], np.ndarray]:
    """
    Load chunk records, generate embeddings, and save them.
    Optionally save chunk records enriched with embedding order metadata.
    """
    if not chunk_jsonl_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_jsonl_path}")

    chunk_records = load_jsonl(chunk_jsonl_path)
    if not chunk_records:
        raise ValueError(f"No chunk records found in: {chunk_jsonl_path}")

    model = load_embedding_model(model_name)
    texts = get_texts_from_chunk_records(chunk_records)
    embeddings = embed_texts(
        texts=texts,
        model=model,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    save_embeddings(embeddings, embeddings_output_path)

    if enriched_chunks_output_path is not None:
        # Add embedding-related metadata only; embeddings themselves are saved separately.
        enriched_records = []
        for idx, record in enumerate(chunk_records):
            enriched_record = dict(record)
            enriched_record["embedding_index"] = idx
            enriched_record["embedding_model"] = model_name
            enriched_records.append(enriched_record)

        save_jsonl(enriched_records, enriched_chunks_output_path)

    return chunk_records, embeddings