from pathlib import Path

from src.utils.vector_store import build_and_save_vector_store


def main():
    chunk_jsonl_path = Path("data/processed/rag_paper_chunks_enriched.jsonl")
    embeddings_path = Path("data/processed/rag_paper_embeddings.npy")
    index_path = Path("data/processed/rag_paper_faiss.index")
    metadata_path = Path("data/processed/rag_paper_chunk_metadata.json")

    index, chunk_records = build_and_save_vector_store(
        chunk_jsonl_path=chunk_jsonl_path,
        embeddings_path=embeddings_path,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    print(f"FAISS index built with {index.ntotal} vectors")
    print(f"Chunk metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()