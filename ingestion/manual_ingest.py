import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env from brain/.env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / "brain" / ".env"
load_dotenv(env_path)

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
