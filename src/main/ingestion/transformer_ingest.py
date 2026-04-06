from pathlib import Path

from src.utils.ingestion import process_pdf


def main():
    pdf_path = Path("data/raw_pdfs/transformer.pdf")
    output_dir = Path("data/processed")

    records, manifest = process_pdf(pdf_path, output_dir)

    print(f"Ingested: {manifest['doc_id']}")
    print(f"Pages: {manifest['page_count']}")
    print(f"Saved pages JSONL to: {output_dir / (manifest['doc_id'] + '_pages.jsonl')}")
    print(f"Saved manifest JSON to: {output_dir / (manifest['doc_id'] + '_manifest.json')}")


if __name__ == "__main__":
    main()