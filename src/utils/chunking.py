from pathlib import Path
import json
import re
import hashlib

from src.utils.ingestion import clean_text, load_jsonl, save_jsonl


try:
    import tiktoken
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None


CHUNK_SIZE_TOKENS = 250
CHUNK_OVERLAP_TOKENS = 50


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if ENCODER is not None:
        return len(ENCODER.encode(text))
    return len(text.split())


def split_text_by_words(text: str, chunk_size_tokens: int, chunk_overlap_tokens: int):
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)

    while start < len(words):
        end = min(len(words), start + chunk_size_tokens)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(words):
            break
        start += step

    return chunks


def split_text_by_tokens(text: str, chunk_size_tokens: int, chunk_overlap_tokens: int):
    if ENCODER is None:
        return split_text_by_words(text, chunk_size_tokens, chunk_overlap_tokens)

    tokens = ENCODER.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)

    while start < len(tokens):
        end = min(len(tokens), start + chunk_size_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = ENCODER.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += step

    return chunks


def make_chunk_id(doc_id: str, page_num: int, chunk_index: int) -> str:
    return f"{doc_id}_p{page_num:03d}_c{chunk_index:03d}"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_pages(page_records):
    chunk_records = []

    for page in page_records:
        text = clean_text(page.get("text", ""))
        if not text:
            continue

        doc_id = page["doc_id"]
        page_num = page["page_num"]
        title = page.get("title", "")
        source_file = page.get("source_file", "")
        is_scanned_page = page.get("is_scanned_page", False)

        page_chunks = split_text_by_tokens(
            text,
            chunk_size_tokens=CHUNK_SIZE_TOKENS,
            chunk_overlap_tokens=CHUNK_OVERLAP_TOKENS,
        )

        for i, chunk_text in enumerate(page_chunks):
            chunk_text = clean_text(chunk_text)
            if not chunk_text:
                continue

            chunk_id = make_chunk_id(doc_id, page_num, i)
            token_count = count_tokens(chunk_text)

            chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "source_file": source_file,
                    "page_num": page_num,
                    "page_index": page.get("page_index"),
                    "page_label": page.get("page_label"),
                    "is_scanned_page": is_scanned_page,
                    "chunk_index": i,
                    "chunk_text": chunk_text,
                    "chunk_token_count": token_count,
                    "chunk_char_count": len(chunk_text),
                    "chunk_hash": hash_text(chunk_text),
                    "source_page_label": f"Page {page_num}",
                }
            )

    return chunk_records


def process_chunks(input_jsonl: Path, output_jsonl: Path):
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Missing input file: {input_jsonl}")

    page_records = load_jsonl(input_jsonl)
    chunk_records = chunk_pages(page_records)
    save_jsonl(chunk_records, output_jsonl)

    return page_records, chunk_records