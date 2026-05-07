import sys
import os
from pathlib import Path

# Add the ingestion directory to python path so we can import watcher
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INGEST_DIR = PROJECT_ROOT / "ingestion"
sys.path.append(str(INGEST_DIR))

from watcher import process_pdf_safely

def main():
    target_dir = PROJECT_ROOT / "ingest_folder"
    
    files_to_ingest = [
        "A Multi-Agent Deep Learning and Reinforcement.pdf",
        "Future of Artificial IntelligencedMachine Learning Trends in Pathology.pdf",
        "Systematic review and meta-analysis on prevalence and risk factors for.pdf"
    ]
    
    for filename in files_to_ingest:
        pdf_path = target_dir / filename
        if pdf_path.exists():
            print(f"Starting ingestion for: {filename}")
            process_pdf_safely(str(pdf_path))
        else:
            print(f"File not found: {pdf_path}")

if __name__ == "__main__":
    main()
