"""
run_architecture_benchmark.py — Side-by-side architecture comparison.

Reads saved result JSON files (produced by generate_results.py) and computes
non-LLM metrics for each architecture, then prints a formatted comparison table.

Metrics computed (no LLM required — fast, deterministic):
    - Exact Match (EM)
    - Token F1
    - ROUGE-L F1
    - Source Hit Rate
    - Source Precision / Recall / MRR
    - Average latency

Usage:
    # Step 1: Collect results for each architecture (one at a time):
    #   Start simple_hybrid_rag backend → python generate_results.py --arch simple_hybrid_rag
    #   Start final_arch backend        → python generate_results.py --arch final_arch
    #   Start agentic_scholar backend   → python generate_results.py --arch agentic_scholar

    # Step 2: Compare all saved results:
    python run_architecture_benchmark.py

    # Or compare specific result files:
    python run_architecture_benchmark.py --results results_simple_hybrid_rag.json results_final_arch.json results_agentic_scholar.json

    # Or specify custom output path:
    python run_architecture_benchmark.py --output my_benchmark.csv
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# ── Text Metrics ──────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    # Strip the fallback note that agentic_scholar appends so it doesn't penalise metrics
    text = re.split(r"\n\n---\n\*note:", text)[0].strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def exact_match(answer: str, ground_truth: str) -> float:
    return float(normalize_text(answer) == normalize_text(ground_truth))


def token_f1(answer: str, ground_truth: str) -> float:
    pred = tokenize(answer)
    gold = tokenize(ground_truth)

    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def rouge_l_f1(answer: str, ground_truth: str) -> float:
    pred = tokenize(answer)
    gold = tokenize(ground_truth)

    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    lcs = lcs_length(pred, gold)
    precision = lcs / len(pred)
    recall = lcs / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Source Metrics ────────────────────────────────────────────────────────────

def extract_predicted_sources(citations) -> list[str]:
    if not isinstance(citations, list):
        return []
    predicted = []
    for c in citations:
        if isinstance(c, dict):
            src = c.get("source_file")
            if src:
                predicted.append(str(src))
    return predicted


def source_metrics(gold_sources, predicted_sources: list[str]) -> dict:
    if gold_sources is None:
        gold = set()
    elif isinstance(gold_sources, list):
        gold = set(str(v) for v in gold_sources)
    else:
        gold = {str(gold_sources)}

    pred_set = set(predicted_sources)
    intersection = gold & pred_set

    hit = float(len(intersection) > 0)
    precision = (len(intersection) / len(pred_set)) if pred_set else 0.0
    recall = (len(intersection) / len(gold)) if gold else 0.0

    mrr = 0.0
    for idx, src in enumerate(predicted_sources, start=1):
        if src in gold:
            mrr = 1.0 / idx
            break

    return {
        "source_hit": hit,
        "source_precision": precision,
        "source_recall": recall,
        "source_mrr": mrr,
    }


# ── Per-result Scoring ────────────────────────────────────────────────────────

def score_result(row: dict) -> dict:
    answer = row.get("answer", "")
    ground_truth = row.get("ground_truth", "")
    citations = row.get("citations", [])

    predicted_sources = extract_predicted_sources(citations)
    src = source_metrics(row.get("source_file"), predicted_sources)

    return {
        "exact_match": exact_match(answer, ground_truth),
        "token_f1": token_f1(answer, ground_truth),
        "rouge_l_f1": rouge_l_f1(answer, ground_truth),
        **src,
        "latency_sec": row.get("latency_sec"),
    }


# ── Loading ───────────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Aggregation ───────────────────────────────────────────────────────────────

METRIC_COLS = [
    "exact_match",
    "token_f1",
    "rouge_l_f1",
    "source_hit",
    "source_precision",
    "source_recall",
    "source_mrr",
    "latency_sec",
]


def aggregate(df: pd.DataFrame) -> dict:
    out = {}
    for col in METRIC_COLS:
        if col in df.columns:
            valid = df[col].dropna()
            out[col] = float(valid.mean()) if len(valid) else None
        else:
            out[col] = None
    return out


# ── Printing ──────────────────────────────────────────────────────────────────

def fmt(val, pct: bool = True) -> str:
    if val is None:
        return "  —  "
    if pct:
        return f"{val * 100:5.1f}%"
    return f"{val:6.2f}s"


def print_comparison_table(summary_rows: list[dict]) -> None:
    archs = [r["architecture"] for r in summary_rows]
    col_w = max(len(a) for a in archs) + 2

    header_metrics = [
        ("EM",        "exact_match",      True),
        ("Tok-F1",    "token_f1",         True),
        ("ROUGE-L",   "rouge_l_f1",       True),
        ("Src-Hit",   "source_hit",       True),
        ("Src-P",     "source_precision", True),
        ("Src-R",     "source_recall",    True),
        ("Src-MRR",   "source_mrr",       True),
        ("Latency",   "latency_sec",      False),
    ]

    lbl_w = 9
    metric_w = 8

    sep = "─" * (col_w + lbl_w + metric_w * len(header_metrics) + 2)

    print()
    print("═" * len(sep))
    print("  ARCHITECTURE BENCHMARK — OVERALL (non-LLM metrics, gold_standard_dev_24)")
    print("═" * len(sep))

    # Header row
    header = f"{'Architecture':<{col_w}}" + "".join(
        f"{lbl:>{metric_w}}" for lbl, _, _ in header_metrics
    )
    print(header)
    print(sep)

    for row in summary_rows:
        arch = row["architecture"]
        values = "".join(
            fmt(row.get(key), pct) for _, key, pct in header_metrics
        )
        print(f"{arch:<{col_w}}{values}")

    print(sep)

    # Per-category tables
    all_categories = sorted(
        {cat for r in summary_rows for cat in r.get("by_category", {}).keys()}
    )

    if all_categories:
        print()
        print("─" * len(sep))
        print("  BY CATEGORY")
        print("─" * len(sep))

        for cat in all_categories:
            print(f"\n  [{cat}]")
            cat_header = f"  {'Architecture':<{col_w - 2}}" + "".join(
                f"{lbl:>{metric_w}}" for lbl, _, _ in header_metrics
            )
            print(cat_header)

            for row in summary_rows:
                arch = row["architecture"]
                cat_data = row.get("by_category", {}).get(cat, {})
                values = "".join(
                    fmt(cat_data.get(key), pct) for _, key, pct in header_metrics
                )
                print(f"  {arch:<{col_w - 2}}{values}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

DEFAULT_RESULT_FILES = [
    "results_simple_hybrid_rag.json",
    "results_final_arch.json",
    "results_agentic_scholar.json",
]


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple architecture result JSONs side-by-side."
    )
    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
        help=(
            "Result JSON file paths to compare. "
            f"Defaults to: {', '.join(DEFAULT_RESULT_FILES)}"
        ),
    )
    parser.add_argument(
        "--output",
        default="architecture_benchmark_results.csv",
        help="Path to save the summary CSV (default: architecture_benchmark_results.csv).",
    )
    args = parser.parse_args()

    result_files = [Path(p) for p in (args.results or DEFAULT_RESULT_FILES)]

    # Drop files that don't exist and warn
    existing = []
    for p in result_files:
        if p.exists():
            existing.append(p)
        else:
            print(f"[WARN] Result file not found, skipping: {p}")

    if not existing:
        print(
            "\nNo result files found. Run generate_results.py for each architecture first:\n"
            "  python generate_results.py --arch simple_hybrid_rag\n"
            "  python generate_results.py --arch final_arch\n"
            "  python generate_results.py --arch agentic_scholar\n"
        )
        return

    summary_rows = []
    all_detail_rows = []

    for path in existing:
        print(f"Processing: {path}")
        raw_results = load_results(path)

        if not raw_results:
            print(f"  [WARN] Empty results file, skipping.")
            continue

        arch_name = raw_results[0].get("architecture", path.stem)

        # Score each successful result
        detail_rows = []
        for row in raw_results:
            if row.get("status") != "ok":
                continue
            scores = score_result(row)
            detail_rows.append({
                "architecture": arch_name,
                "question": row.get("question"),
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                **scores,
            })

        if not detail_rows:
            print(f"  [WARN] No successful results, skipping.")
            continue

        df = pd.DataFrame(detail_rows)
        all_detail_rows.extend(detail_rows)

        ok_count = len(df)
        total_count = len(raw_results)
        print(f"  Scored {ok_count}/{total_count} successful results.")

        # Overall aggregation
        overall = aggregate(df)
        overall["architecture"] = arch_name
        overall["n_ok"] = ok_count
        overall["n_total"] = total_count

        # Per-category aggregation
        by_category = {}
        if "category" in df.columns:
            for cat, subset in df.groupby("category", dropna=True):
                by_category[str(cat)] = aggregate(subset)

        summary_rows.append({
            **overall,
            "by_category": by_category,
        })

    if not summary_rows:
        print("\nNo data to compare.")
        return

    # Print comparison tables
    print_comparison_table(summary_rows)

    # Save CSV summary (flatten by_category out)
    flat_rows = [{k: v for k, v in r.items() if k != "by_category"} for r in summary_rows]
    summary_df = pd.DataFrame(flat_rows)
    output_path = Path(args.output)
    summary_df.to_csv(output_path, index=False)
    print(f"Saved benchmark summary CSV → {output_path}")

    # Save detailed per-question CSV
    if all_detail_rows:
        detail_df = pd.DataFrame(all_detail_rows)
        detail_path = output_path.with_name(output_path.stem + "_detailed.csv")
        detail_df.to_csv(detail_path, index=False)
        print(f"Saved detailed per-question CSV → {detail_path}")


if __name__ == "__main__":
    main()
