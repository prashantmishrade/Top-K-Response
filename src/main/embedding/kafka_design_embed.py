from pathlib import Path

from src.utils.embedding import process_embeddings


def main():
    # Input: chunks file from previous step
    chunk_jsonl_path = Path("data/processed/kafka_design_chunks.jsonl")

    # Output: embeddings + enriched chunks
    embeddings_output_path = Path("data/processed/kafka_design_embeddings.npy")
    enriched_chunks_output_path = Path("data/processed/kafka_design_chunks_enriched.jsonl")

    if not chunk_jsonl_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunk_jsonl_path}")

    records, embeddings = process_embeddings(
        chunk_jsonl_path=chunk_jsonl_path,
        embeddings_output_path=embeddings_output_path,
        enriched_chunks_output_path=enriched_chunks_output_path,
    )

    print(f"Chunks processed: {len(records)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to: {embeddings_output_path}")
    print(f"Saved enriched chunks to: {enriched_chunks_output_path}")


if __name__ == "__main__":
    main()
