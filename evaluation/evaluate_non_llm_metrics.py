import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Fast non-LLM evaluation for saved RAG results.
# This is intentionally separate from the RAGAS evaluator.

RESULTS_PATH = Path("./results_simple_hybrid_rag.json")

DETAIL_CSV_PATH = Path("./non_llm_metrics_detailed_1.csv")
SUMMARY_JSON_PATH = Path("./non_llm_metrics_summary_1.json")
CATEGORY_CSV_PATH = Path("./non_llm_metrics_by_category_1.csv")
DIFFICULTY_CSV_PATH = Path("./non_llm_metrics_by_difficulty_1.csv")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text: str):
    return normalize_text(text).split()


def exact_match(answer: str, ground_truth: str) -> float:
    return float(normalize_text(answer) == normalize_text(ground_truth))


def token_f1(answer: str, ground_truth: str) -> float:
    pred_tokens = tokenize(answer)
    gold_tokens = tokenize(ground_truth)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = pred_counts & gold_counts
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def lcs_length(a_tokens, b_tokens) -> int:
    m, n = len(a_tokens), len(b_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a_tokens[i] == b_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def rouge_l_f1(answer: str, ground_truth: str) -> float:
    pred_tokens = tokenize(answer)
    gold_tokens = tokenize(ground_truth)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, gold_tokens)
    prec = lcs / len(pred_tokens)
    rec = lcs / len(gold_tokens)

    if prec + rec == 0:
        return 0.0

    return 2 * prec * rec / (prec + rec)


def to_source_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def extract_predicted_sources(citations):
    predicted = []
    if not isinstance(citations, list):
        return predicted

    for c in citations:
        if isinstance(c, dict):
            src = c.get("source_file")
            if src:
                predicted.append(str(src))
    return predicted


def source_metrics(gold_sources, predicted_sources):
    gold = set(to_source_list(gold_sources))
    pred = list(predicted_sources)
    pred_set = set(pred)

    intersection = gold & pred_set

    hit = float(len(intersection) > 0)
    precision = (len(intersection) / len(pred_set)) if pred_set else 0.0
    recall = (len(intersection) / len(gold)) if gold else 0.0

    mrr = 0.0
    for idx, src in enumerate(pred, start=1):
        if src in gold:
            mrr = 1.0 / idx
            break

    return {
        "source_hit": hit,
        "source_precision": precision,
        "source_recall": recall,
        "source_mrr": mrr,
    }


def build_detail_rows(results):
    rows = []
    for row in results:
        answer = row.get("answer", "")
        ground_truth = row.get("ground_truth", "")
        predicted_sources = extract_predicted_sources(row.get("citations", []))
        src_metrics = source_metrics(row.get("source_file"), predicted_sources)

        rows.append(
            {
                "question": row.get("question"),
                "ground_truth": ground_truth,
                "answer": answer,
                "architecture": row.get("architecture"),
                "source_file": json.dumps(row.get("source_file", []), ensure_ascii=False),
                "predicted_sources": json.dumps(predicted_sources, ensure_ascii=False),
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "latency_sec": row.get("latency_sec"),
                "status": row.get("status"),
                "error": json.dumps(row.get("error"), ensure_ascii=False) if row.get("error") is not None else None,
                "exact_match": exact_match(answer, ground_truth) if row.get("status") == "ok" else None,
                "token_f1": token_f1(answer, ground_truth) if row.get("status") == "ok" else None,
                "rouge_l_f1": rouge_l_f1(answer, ground_truth) if row.get("status") == "ok" else None,
                "source_hit": src_metrics["source_hit"] if row.get("status") == "ok" else None,
                "source_precision": src_metrics["source_precision"] if row.get("status") == "ok" else None,
                "source_recall": src_metrics["source_recall"] if row.get("status") == "ok" else None,
                "source_mrr": src_metrics["source_mrr"] if row.get("status") == "ok" else None,
            }
        )
    return rows


def aggregate(df: pd.DataFrame):
    metric_cols = [
        "exact_match",
        "token_f1",
        "rouge_l_f1",
        "source_hit",
        "source_precision",
        "source_recall",
        "source_mrr",
    ]

    summary = {}
    for col in metric_cols:
        if col in df.columns:
            valid = df[col].dropna()
            summary[col] = float(valid.mean()) if len(valid) else None

    return summary


def main():
    print(f"Loading saved results from: {RESULTS_PATH}")
    results = load_json(RESULTS_PATH)

    total_rows = len(results)
    ok_rows = sum(1 for r in results if r.get("status") == "ok")
    failed_rows = total_rows - ok_rows

    detail_rows = build_detail_rows(results)
    detailed_df = pd.DataFrame(detail_rows)
    detailed_df.to_csv(DETAIL_CSV_PATH, index=False)

    ok_df = detailed_df[detailed_df["status"] == "ok"].copy()

    overall = aggregate(ok_df)
    overall["architecture"] = results[0].get("architecture") if results else None
    overall["total_rows"] = total_rows
    overall["successful_rows"] = ok_rows
    overall["failed_rows"] = failed_rows
    overall["failure_rate"] = (failed_rows / total_rows) if total_rows else None
    overall["avg_latency_sec"] = float(ok_df["latency_sec"].dropna().mean()) if len(ok_df) else None

    category_rows = []
    if "category" in ok_df.columns:
        for category, subset in ok_df.groupby("category", dropna=True):
            row = aggregate(subset)
            row["category"] = category
            row["count"] = int(len(subset))
            row["avg_latency_sec"] = float(subset["latency_sec"].dropna().mean()) if len(subset) else None
            category_rows.append(row)

    category_df = pd.DataFrame(category_rows)
    if not category_df.empty:
        category_df.to_csv(CATEGORY_CSV_PATH, index=False)

    difficulty_rows = []
    if "difficulty" in ok_df.columns:
        for difficulty, subset in ok_df.groupby("difficulty", dropna=True):
            row = aggregate(subset)
            row["difficulty"] = difficulty
            row["count"] = int(len(subset))
            row["avg_latency_sec"] = float(subset["latency_sec"].dropna().mean()) if len(subset) else None
            difficulty_rows.append(row)

    difficulty_df = pd.DataFrame(difficulty_rows)
    if not difficulty_df.empty:
        difficulty_df.to_csv(DIFFICULTY_CSV_PATH, index=False)

    summary = {
        "overall": overall,
        "by_category_rows": category_rows,
        "by_difficulty_rows": difficulty_rows,
        "detail_csv": str(DETAIL_CSV_PATH),
        "category_csv": str(CATEGORY_CSV_PATH),
        "difficulty_csv": str(DIFFICULTY_CSV_PATH),
    }

    save_json(summary, SUMMARY_JSON_PATH)

    print("\\n====================")
    print("NON-LLM EVALUATION COMPLETE")
    print("====================")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"\\nSaved detailed metrics to: {DETAIL_CSV_PATH}")
    print(f"Saved summary JSON to: {SUMMARY_JSON_PATH}")
    if not category_df.empty:
        print(f"Saved category breakdown to: {CATEGORY_CSV_PATH}")
    if not difficulty_df.empty:
        print(f"Saved difficulty breakdown to: {DIFFICULTY_CSV_PATH}")


if __name__ == "__main__":
    main()
