"""
ingest_all.py — One-shot ingestion of all PDFs in the ingest_folder.

Bypasses the watchdog file watcher (which has a Python 3.13 compatibility issue)
and directly calls the same parse + chunk + index pipeline that watcher.py uses.

Usage:
    cd Academic_RAG/ingestion
    python ingest_all.py
"""

import os
import sys

# Force UTF-8 I/O on Windows (cp1252 chokes on unicode chars in watcher.py)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load from brain/.env (the canonical .env for this project)
_ENV_PATH = Path(__file__).parent.parent / "brain" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# Ensure the ingestion folder is on the path
sys.path.insert(0, str(Path(__file__).parent))

import indexer
from chunker import chunk_markdown

# Import the core async parse function from watcher
from watcher import parse_pdf_agentic

INGEST_FOLDER = Path(__file__).parent.parent / "ingest_folder"


async def ingest_pdf(pdf_path: Path):
    print(f"\n{'='*60}")
    print(f"Ingesting: {pdf_path.name}")
    print(f"{'='*60}")

    try:
        full_markdown = await parse_pdf_agentic(str(pdf_path))
    except Exception as e:
        print(f"[ERROR] Parsing failed for '{pdf_path.name}': {e}")
        return False

    try:
        chunks = chunk_markdown(full_markdown, source_file=pdf_path.name)
        print(f"[chunker] Total chunks: {len(chunks)}")
    except Exception as e:
        print(f"[ERROR] Chunking failed for '{pdf_path.name}': {e}")
        return False

    try:
        indexer.index_chunks(chunks)
        print(f"[OK] '{pdf_path.name}' indexed successfully.")
    except Exception as e:
        print(f"[ERROR] Indexing failed for '{pdf_path.name}': {e}")
        return False

    return True


async def main():
    pdf_files = sorted(INGEST_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in: {INGEST_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDFs to ingest:")
    for p in pdf_files:
        print(f"  - {p.name}")
    print()

    results = {}
    for pdf_path in pdf_files:
        ok = await ingest_pdf(pdf_path)
        results[pdf_path.name] = "OK" if ok else "FAILED"

    print(f"\n{'='*60}")
    print("INGESTION SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "[OK]" if status == "OK" else "[FAIL]"
        print(f"  {icon}  {name}")

    failed = [n for n, s in results.items() if s == "FAILED"]
    if not failed:
        print(f"\nAll {len(pdf_files)} papers ingested successfully!")
    else:
        print(f"\n{len(failed)} paper(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    asyncio.run(main())
