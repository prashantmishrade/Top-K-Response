from pathlib import Path
import json
import re
import hashlib
from datetime import datetime

import fitz  # PyMuPDF


def make_doc_id(pdf_path: Path) -> str:
    base = pdf_path.stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"[\\]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_jsonl(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(obj, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ingest_pdf(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pdf_meta = doc.metadata or {}

    doc_id = make_doc_id(pdf_path)
    title = pdf_meta.get("title") or pdf_path.stem
    author = pdf_meta.get("author") or ""
    subject = pdf_meta.get("subject") or ""
    keywords = pdf_meta.get("keywords") or ""

    records = []
    scanned_pages = 0
    total_chars = 0

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        is_scanned_page = len(text) == 0
        if is_scanned_page:
            scanned_pages += 1

        total_chars += len(text)

        records.append(
            {
                "doc_id": doc_id,
                "source_file": str(pdf_path),
                "title": title,
                "author": author,
                "subject": subject,
                "keywords": keywords,
                "page_num": page_num + 1,
                "page_index": page_num,
                "page_label": page_num + 1,
                "text": text,
                "char_count": len(text),
                "text_hash": hash_text(text) if text else None,
                "is_scanned_page": is_scanned_page,
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            }
        )

    manifest = {
        "doc_id": doc_id,
        "source_file": str(pdf_path),
        "title": title,
        "author": author,
        "subject": subject,
        "keywords": keywords,
        "page_count": doc.page_count,
        "scanned_pages": scanned_pages,
        "text_pages": doc.page_count - scanned_pages,
        "total_chars": total_chars,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }

    doc.close()
    return records, manifest


def process_pdf(pdf_path: Path, output_dir: Path):
    records, manifest = ingest_pdf(pdf_path)
    doc_id = manifest["doc_id"]

    pages_path = output_dir / f"{doc_id}_pages.jsonl"
    manifest_path = output_dir / f"{doc_id}_manifest.json"

    save_jsonl(records, pages_path)
    save_json(manifest, manifest_path)

    return records, manifest