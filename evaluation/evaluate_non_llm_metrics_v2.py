import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


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
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


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


def get_primary_metric_hint(category: str) -> tuple[str, str]:
    if category == "direct_fact_lookup":
        return ("exact_match", "token_f1")
    if category == "definition_explanation":
        return ("rouge_l_f1", "token_f1")
    if category == "figure_table_diagram_grounded":
        return ("rouge_l_f1", "token_f1")
    if category == "multi_chunk_synthesis":
        return ("rouge_l_f1", "token_f1")
    if category == "paraphrase_hard_retrieval":
        return ("rouge_l_f1", "token_f1")
    if category == "intra_paper_comparison":
        return ("rouge_l_f1", "token_f1")
    if category == "cross_paper_comparison":
        return ("rouge_l_f1", "token_f1")
    if category == "distractor_edge_case":
        return ("rouge_l_f1", "token_f1")
    return ("token_f1", "rouge_l_f1")


def build_detail_rows(results):
    rows = []
    for row in results:
        answer = row.get("answer", "")
        ground_truth = row.get("ground_truth", "")
        predicted_sources = extract_predicted_sources(row.get("citations", []))
        src_metrics = source_metrics(row.get("source_file"), predicted_sources)

        is_ok = row.get("status") == "ok" or row.get("final_status") in ["accepted", "rejected"]
        status_val = row.get("status") or row.get("final_status")

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
                "status": status_val,
                "error": json.dumps(row.get("error"), ensure_ascii=False) if row.get("error") is not None else None,
                "exact_match": exact_match(answer, ground_truth) if is_ok else None,
                "token_f1": token_f1(answer, ground_truth) if is_ok else None,
                "rouge_l_f1": rouge_l_f1(answer, ground_truth) if is_ok else None,
                "source_hit": src_metrics["source_hit"] if is_ok else None,
                "source_precision": src_metrics["source_precision"] if is_ok else None,
                "source_recall": src_metrics["source_recall"] if is_ok else None,
                "source_mrr": src_metrics["source_mrr"] if is_ok else None,
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
        valid = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        summary[col] = float(valid.mean()) if len(valid) else None

    return summary


def build_recommended_view(ok_df: pd.DataFrame):
    results = {}

    # Fact-only summary: use EM + Token F1 as primary.
    fact_df = ok_df[ok_df["category"] == "direct_fact_lookup"].copy()
    if not fact_df.empty:
        results["direct_fact_lookup_primary_view"] = {
            "count": int(len(fact_df)),
            "primary_metrics": ["exact_match", "token_f1"],
            "exact_match": float(fact_df["exact_match"].mean()),
            "token_f1": float(fact_df["token_f1"].mean()),
            "source_hit": float(fact_df["source_hit"].mean()),
            "source_precision": float(fact_df["source_precision"].mean()),
            "source_recall": float(fact_df["source_recall"].mean()),
            "source_mrr": float(fact_df["source_mrr"].mean()),
            "avg_latency_sec": float(fact_df["latency_sec"].dropna().mean()) if len(fact_df["latency_sec"].dropna()) else None,
        }

    # Explanation/synthesis-style categories: use ROUGE-L + Token F1 as primary.
    explanation_like = ok_df[
        ok_df["category"].isin(
            [
                "definition_explanation",
                "figure_table_diagram_grounded",
                "multi_chunk_synthesis",
                "paraphrase_hard_retrieval",
                "intra_paper_comparison",
                "cross_paper_comparison",
                "distractor_edge_case",
            ]
        )
    ].copy()

    if not explanation_like.empty:
        results["explanation_synthesis_primary_view"] = {
            "count": int(len(explanation_like)),
            "primary_metrics": ["rouge_l_f1", "token_f1"],
            "rouge_l_f1": float(explanation_like["rouge_l_f1"].mean()),
            "token_f1": float(explanation_like["token_f1"].mean()),
            "source_hit": float(explanation_like["source_hit"].mean()),
            "source_precision": float(explanation_like["source_precision"].mean()),
            "source_recall": float(explanation_like["source_recall"].mean()),
            "source_mrr": float(explanation_like["source_mrr"].mean()),
            "avg_latency_sec": float(explanation_like["latency_sec"].dropna().mean()) if len(explanation_like["latency_sec"].dropna()) else None,
        }

    # Retrieval-only view across all successful questions.
    results["retrieval_grounding_view"] = {
        "count": int(len(ok_df)),
        "primary_metrics": ["source_hit", "source_precision", "source_recall", "source_mrr"],
        "source_hit": float(ok_df["source_hit"].mean()),
        "source_precision": float(ok_df["source_precision"].mean()),
        "source_recall": float(ok_df["source_recall"].mean()),
        "source_mrr": float(ok_df["source_mrr"].mean()),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Category-aware non-LLM evaluation for saved RAG results.")
    parser.add_argument("--results", type=str, default="./results_simple_hybrid_rag.json", help="Path to saved results JSON")
    parser.add_argument("--out-prefix", type=str, default="non_llm_metrics_v2", help="Prefix for output files")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_prefix = args.out_prefix

    detail_csv_path = Path(f"{out_prefix}_detailed.csv")
    summary_json_path = Path(f"{out_prefix}_summary.json")
    category_csv_path = Path(f"{out_prefix}_by_category.csv")
    difficulty_csv_path = Path(f"{out_prefix}_by_difficulty.csv")

    print(f"Loading saved results from: {results_path}")
    results = load_json(results_path)

    total_rows = len(results)
    ok_rows = sum(1 for r in results if r.get("status") == "ok" or r.get("final_status") in ["accepted", "rejected"])
    failed_rows = total_rows - ok_rows

    detail_rows = build_detail_rows(results)
    detailed_df = pd.DataFrame(detail_rows)
    detailed_df.to_csv(detail_csv_path, index=False)

    ok_df = detailed_df[detailed_df["status"].isin(["ok", "accepted", "rejected"])].copy()

    overall = aggregate(ok_df)
    overall["architecture"] = results[0].get("architecture") if results else None
    overall["total_rows"] = total_rows
    overall["successful_rows"] = ok_rows
    overall["failed_rows"] = failed_rows
    overall["failure_rate"] = (failed_rows / total_rows) if total_rows else None
    overall["avg_latency_sec"] = float(ok_df["latency_sec"].dropna().mean()) if len(ok_df["latency_sec"].dropna()) else None

    category_rows = []
    if "category" in ok_df.columns:
        for category, subset in ok_df.groupby("category", dropna=True):
            row = aggregate(subset)
            p1, p2 = get_primary_metric_hint(str(category))
            row["category"] = category
            row["count"] = int(len(subset))
            row["avg_latency_sec"] = float(subset["latency_sec"].dropna().mean()) if len(subset["latency_sec"].dropna()) else None
            row["recommended_primary_metric_1"] = p1
            row["recommended_primary_metric_2"] = p2
            category_rows.append(row)

    category_df = pd.DataFrame(category_rows)
    if not category_df.empty:
        category_df.to_csv(category_csv_path, index=False)

    difficulty_rows = []
    if "difficulty" in ok_df.columns:
        for difficulty, subset in ok_df.groupby("difficulty", dropna=True):
            row = aggregate(subset)
            row["difficulty"] = difficulty
            row["count"] = int(len(subset))
            row["avg_latency_sec"] = float(subset["latency_sec"].dropna().mean()) if len(subset["latency_sec"].dropna()) else None
            difficulty_rows.append(row)

    difficulty_df = pd.DataFrame(difficulty_rows)
    if not difficulty_df.empty:
        difficulty_df.to_csv(difficulty_csv_path, index=False)

    recommended_view = build_recommended_view(ok_df)

    summary = {
        "overall": overall,
        "recommended_view": recommended_view,
        "by_category_rows": category_rows,
        "by_difficulty_rows": difficulty_rows,
        "detail_csv": str(detail_csv_path),
        "category_csv": str(category_csv_path),
        "difficulty_csv": str(difficulty_csv_path),
    }

    save_json(summary, summary_json_path)

    print("\n====================")
    print("CATEGORY-AWARE NON-LLM EVALUATION COMPLETE")
    print("====================")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print("\nRecommended metric views:")
    print(json.dumps(recommended_view, indent=2, ensure_ascii=False))
    print(f"\nSaved detailed metrics to: {detail_csv_path}")
    print(f"Saved summary JSON to: {summary_json_path}")
    if not category_df.empty:
        print(f"Saved category breakdown to: {category_csv_path}")
    if not difficulty_df.empty:
        print(f"Saved difficulty breakdown to: {difficulty_csv_path}")


if __name__ == "__main__":
    main()
