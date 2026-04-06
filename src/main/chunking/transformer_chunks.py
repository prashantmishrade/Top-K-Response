from pathlib import Path

from src.utils.chunking import process_chunks


def main():
    input_jsonl = Path("data/processed/transformer_pages.jsonl")
    output_jsonl = Path("data/processed/transformer_chunks.jsonl")

    page_records, chunk_records = process_chunks(input_jsonl, output_jsonl)

    print(f"Pages loaded: {len(page_records)}")
    print(f"Chunks created: {len(chunk_records)}")
    print(f"Saved chunks JSONL to: {output_jsonl}")


if __name__ == "__main__":
    main()