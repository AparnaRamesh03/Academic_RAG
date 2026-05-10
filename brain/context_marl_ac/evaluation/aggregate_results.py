"""
brain/context_marl_ac/evaluation/aggregate_results.py
-----------------------------------------------------
Aggregates and compares results across different architectures.
"""

import json
import os
from pathlib import Path

import pandas as pd

from context_marl_ac.config import FINAL_EVAL_DIR


def _safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns:
        return default
    if df[col].empty:
        return default
    return float(df[col].fillna(0).mean())


def aggregate_all(output_file: str = "comparison_summary.md"):
    if not os.path.exists(FINAL_EVAL_DIR):
        print(f"No results found in {FINAL_EVAL_DIR}")
        return

    jsonl_files = list(FINAL_EVAL_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL result files found.")
        return

    all_stats = []

    for result_file in jsonl_files:
        rows = []
        with open(result_file, "r") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))

        if not rows:
            continue

        df = pd.DataFrame(rows)

        arch_name = (
            df["architecture"].iloc[0]
            if "architecture" in df.columns
            else result_file.stem
        )

        if "verification_pass" in df.columns:
            pass_rate = _safe_mean(df, "verification_pass")
        elif "final_status" in df.columns:
            pass_rate = float((df["final_status"] == "accepted").mean())
        else:
            pass_rate = 0.0

        cit_support = _safe_mean(df, "citation_support_rate")

        latency_col = None
        if "latency_seconds" in df.columns:
            latency_col = "latency_seconds"
        elif "latency_sec" in df.columns:
            latency_col = "latency_sec"

        avg_latency = _safe_mean(df, latency_col) if latency_col else 0.0

        avg_llm = _safe_mean(df, "num_llm_calls")
        avg_tokens = _safe_mean(df, "token_usage")

        accepted = int((df["final_status"] == "accepted").sum()) if "final_status" in df.columns else 0
        rejected = int((df["final_status"] == "rejected").sum()) if "final_status" in df.columns else 0
        abstained = int((df["final_status"] == "abstained").sum()) if "final_status" in df.columns else 0
        failed = int((df["final_status"] == "generation_failed").sum()) if "final_status" in df.columns else 0

        all_stats.append({
            "Architecture": arch_name,
            "Accuracy / Verification Pass": f"{pass_rate:.2%}",
            "Citation Support": f"{cit_support:.2%}",
            "Avg Latency (s)": f"{avg_latency:.2f}",
            "Avg LLM Calls": f"{avg_llm:.2f}",
            "Avg Token Usage": f"{avg_tokens:.0f}",
            "Accepted": accepted,
            "Rejected": rejected,
            "Abstained": abstained,
            "Generation Failed": failed,
            "Samples": len(df),
        })

    summary_df = pd.DataFrame(all_stats)

    summary_df.to_csv(FINAL_EVAL_DIR / "comparison_stats.csv", index=False)

    md_content = "# Architecture Performance Comparison\n\n"
    md_content += summary_df.to_markdown(index=False)
    md_content += "\n\n*Generated automatically by aggregate_results.py*\n"

    with open(FINAL_EVAL_DIR / output_file, "w") as f:
        f.write(md_content)

    print(f"Summary generated at {FINAL_EVAL_DIR / output_file}")


if __name__ == "__main__":
    aggregate_all()