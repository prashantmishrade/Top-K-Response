from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.utils.vector_store import load_vector_store, search_chunks


def main():
    index_path = Path("data/processed/rag_paper_faiss.index")
    metadata_path = Path("data/processed/rag_paper_chunk_metadata.json")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index, chunk_records = load_vector_store(index_path, metadata_path)

    query = "What is retrieval augmented generation?"
    results = search_chunks(
        query=query,
        model=model,
        index=index,
        chunk_records=chunk_records,
        top_k=5,
    )

    for r in results:
        print("=" * 80)
        print(f"Rank: {r['rank']}, Score: {r['score']:.4f}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Page: {r['page_num']}")
        print(r["chunk_text"][:500])


if __name__ == "__main__":
    main()