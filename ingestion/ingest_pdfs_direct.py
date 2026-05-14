"""
ingest_pdfs_direct.py
---------------------
Ingest PDFs directly into Qdrant without the LlamaCloud watcher.
Uses PyMuPDF for text extraction then the existing indexer.

Usage (from repo root):
  python ingestion/ingest_pdfs_direct.py ingestion/ingest_pdfs_direct.py ingest_folder/
  # or specific files:
  python ingestion/ingest_pdfs_direct.py ingest_folder/BERT.pdf
"""

import os
import re
import sys
import uuid
from pathlib import Path

import fitz  # PyMuPDF

# Ensure ingestion/ is on path for indexer import
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[1] / "brain" / ".env")

import indexer  # noqa — must import after dotenv so QDRANT_URL is set


# ── Text extraction ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"\s{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_chunks(pdf_path: Path, chunk_size: int = 1200, overlap: int = 150):
    """Extract text from PDF and split into overlapping chunks."""
    doc = fitz.open(str(pdf_path))
    source_name = pdf_path.name

    all_chunks = []
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        text = _clean(raw)
        if not text:
            continue

        # Split page text into sub-chunks if large enough
        words = text.split()
        if len(words) <= 60:
            all_chunks.append({
                "text": text,
                "metadata": {
                    "source_file": source_name,
                    "page_number": page_num,
                    "section_header": f"Page {page_num}",
                    "content_type": "text",
                    "chunk_index": len(all_chunks),
                },
            })
            continue

        # Sliding-window chunking in character space
        start = 0
        chunk_idx = len(all_chunks)
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space
            chunk_text = text[start:end].strip()
            if chunk_text:
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_file": source_name,
                        "page_number": page_num,
                        "section_header": f"Page {page_num}",
                        "content_type": "text",
                        "chunk_index": chunk_idx,
                    },
                })
                chunk_idx += 1
            start = end - overlap if (end - overlap) > start else end

    doc.close()
    return all_chunks


# ── Indexer integration ───────────────────────────────────────────────────────

def ingest_pdf(pdf_path: Path):
    print(f"\n[ingest] {pdf_path.name} ...", flush=True)
    chunks = pdf_to_chunks(pdf_path)
    print(f"  extracted {len(chunks)} chunks")

    if not chunks:
        print("  [skip] no text extracted")
        return

    # indexer.index_chunks expects list of dicts with {text, source_file, ...}
    # It handles collection creation and upsertion.
    indexer.index_chunks(chunks)
    print(f"  [done] {pdf_path.name} -> {len(chunks)} chunks indexed")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    targets = sys.argv[1:] or ["ingest_folder"]
    pdf_files = []

    for t in targets:
        p = Path(t)
        if p.is_dir():
            pdf_files.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf" and p.exists():
            pdf_files.append(p)
        else:
            print(f"[warn] skipping: {t}")

    if not pdf_files:
        print("[error] no PDF files found")
        sys.exit(1)

    print(f"[ingest] Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")
    indexer._ensure_models_loaded()
    indexer._setup_collection()

    for pdf in pdf_files:
        try:
            ingest_pdf(pdf)
        except Exception as e:
            print(f"  [error] {pdf.name}: {e}")

    print("\n[ingest] All done.")
