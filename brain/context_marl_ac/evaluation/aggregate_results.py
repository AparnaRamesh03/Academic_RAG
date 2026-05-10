"""
brain/context_marl_ac/evaluation/aggregate_results.py
-----------------------------------------------------
Aggregates and compares results across different architectures.
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# ── sys.path setup ────────────────────────────────────────────────────────────
_MARL_ROOT = Path(__file__).resolve().parents[1]
_BRAIN_ROOT = _MARL_ROOT.parent

# ── Imports ───────────────────────────────────────────────────────────────────
from context_marl_ac.config import FINAL_EVAL_DIR

def aggregate_all(output_file: str = "comparison_summary.md"):
    if not os.path.exists(FINAL_EVAL_DIR):
        print(f"No results found in {FINAL_EVAL_DIR}")
        return

    jsonl_files = list(FINAL_EVAL_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL result files found.")
        return

    all_stats = []
    
    for f in jsonl_files:
        rows = []
        with open(f, "r") as fh:
            for line in fh:
                rows.append(json.loads(line))
        
        if not rows: continue
        
        df = pd.DataFrame(rows)
        arch_name = df["architecture"].iloc[0] if "architecture" in df.columns else f.stem
        
        # Calculate stats
        # 1. Verification Pass Rate
        pass_rate = df["verification_pass"].mean() if "verification_pass" in df.columns else 0.0
        
        # 2. Citation Support
        cit_support = df["citation_support_rate"].mean() if "citation_support_rate" in df.columns else 0.0
        
        # 3. Latency
        avg_latency = df["latency_sec"].mean()
        
        # 4. LLM Efficiency (if present)
        avg_llm = df["num_llm_calls"].mean() if "num_llm_calls" in df.columns else 0.0
        
        # 5. Accuracy (heuristic: verification_pass == True)
        accuracy = pass_rate
        
        all_stats.append({
            "Architecture": arch_name,
            "Accuracy (Verif)": f"{accuracy:.2%}",
            "Citation Support": f"{cit_support:.2%}",
            "Avg Latency (s)": f"{avg_latency:.2f}",
            "Avg LLM Calls": f"{avg_llm:.2f}",
            "Samples": len(df)
        })

    # Create Summary DataFrame
    summary_df = pd.DataFrame(all_stats)
    
    # Save to CSV
    summary_df.to_csv(FINAL_EVAL_DIR / "comparison_stats.csv", index=False)
    
    # Save to Markdown
    md_content = "# Architecture Performance Comparison\n\n"
    md_content += summary_df.to_markdown(index=False)
    md_content += "\n\n*Generated automatically by aggregate_results.py*\n"
    
    with open(FINAL_EVAL_DIR / output_file, "w") as f:
        f.write(md_content)
        
    print(f"Summary generated at {FINAL_EVAL_DIR / output_file}")

if __name__ == "__main__":
    aggregate_all()
