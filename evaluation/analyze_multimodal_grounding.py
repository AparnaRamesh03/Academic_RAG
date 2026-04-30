import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


STRUCTURED_TYPES = {"figure_description", "table", "diagram_text", "equation_block"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


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


def lcs_length(a_tokens: List[str], b_tokens: List[str]) -> int:
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


def to_source_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def extract_predicted_sources(citations) -> List[str]:
    predicted = []
    if not isinstance(citations, list):
        return predicted

    for c in citations:
        if isinstance(c, dict):
            src = c.get("source_file")
            if src:
                predicted.append(str(src))
    return predicted


def source_metrics(gold_sources, predicted_sources) -> Dict[str, float]:
    gold = set(to_source_list(gold_sources))
    pred = list(predicted_sources)
    pred_set = set(pred)

    intersection = gold & pred_set

    hit = float(len(intersection) > 0)
    precision = (len(intersection) / len(pred_set)) if pred_set else 0.0
    recall = (len(intersection) / len(gold)) if gold else 0.0

    return {
        "source_hit": hit,
        "source_precision": precision,
        "source_recall": recall,
    }


def doc_content_types(docs: List[Dict[str, Any]]) -> List[str]:
    out = []
    for doc in docs or []:
        meta = doc.get("metadata", {}) or {}
        ctype = meta.get("content_type")
        if ctype:
            out.append(str(ctype))
    return out


def count_type(docs: List[Dict[str, Any]], target: str) -> int:
    return sum(1 for t in doc_content_types(docs) if t == target)


def has_any_structured(docs: List[Dict[str, Any]]) -> bool:
    return any(t in STRUCTURED_TYPES for t in doc_content_types(docs))


def build_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for row in results:
        answer = row.get("answer", "")
        ground_truth = row.get("ground_truth", "")
        citations = row.get("citations", [])
        predicted_sources = extract_predicted_sources(citations)
        src = source_metrics(row.get("source_file"), predicted_sources)

        retrieved_docs = row.get("retrieved_docs", []) or []
        graded_docs = row.get("graded_docs", []) or []

        rows.append(
            {
                "question": row.get("question"),
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "status": row.get("status"),
                "latency_sec": row.get("latency_sec"),
                "token_f1": token_f1(answer, ground_truth) if row.get("status") == "ok" else None,
                "rouge_l_f1": rouge_l_f1(answer, ground_truth) if row.get("status") == "ok" else None,
                "source_hit": src["source_hit"] if row.get("status") == "ok" else None,
                "source_precision": src["source_precision"] if row.get("status") == "ok" else None,
                "source_recall": src["source_recall"] if row.get("status") == "ok" else None,
                "retrieved_has_structured": has_any_structured(retrieved_docs),
                "graded_has_structured": has_any_structured(graded_docs),
                "retrieved_figure_description_count": count_type(retrieved_docs, "figure_description"),
                "graded_figure_description_count": count_type(graded_docs, "figure_description"),
                "retrieved_table_count": count_type(retrieved_docs, "table"),
                "graded_table_count": count_type(graded_docs, "table"),
                "retrieved_diagram_count": count_type(retrieved_docs, "diagram_text"),
                "graded_diagram_count": count_type(graded_docs, "diagram_text"),
                "retrieved_equation_count": count_type(retrieved_docs, "equation_block"),
                "graded_equation_count": count_type(graded_docs, "equation_block"),
                "crag_retries": row.get("crag_retries", 0),
                "verify_retries": row.get("verify_retries", 0),
                "citations_pass": row.get("citations_pass"),
                "unsupported_claim_count": sum(
                    1 for c in (row.get("claim_verification", []) or []) if not bool(c.get("supported", False))
                ),
                "auditor_feedback": row.get("auditor_feedback", ""),
            }
        )

    return rows


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    ok_df = df[df["status"] == "ok"].copy()

    summary = {
        "total_rows": int(len(df)),
        "successful_rows": int(len(ok_df)),
        "avg_token_f1": float(ok_df["token_f1"].dropna().mean()) if len(ok_df["token_f1"].dropna()) else None,
        "avg_rouge_l_f1": float(ok_df["rouge_l_f1"].dropna().mean()) if len(ok_df["rouge_l_f1"].dropna()) else None,
        "avg_source_hit": float(ok_df["source_hit"].dropna().mean()) if len(ok_df["source_hit"].dropna()) else None,
        "avg_source_precision": float(ok_df["source_precision"].dropna().mean()) if len(ok_df["source_precision"].dropna()) else None,
        "avg_source_recall": float(ok_df["source_recall"].dropna().mean()) if len(ok_df["source_recall"].dropna()) else None,
        "avg_latency_sec": float(ok_df["latency_sec"].dropna().mean()) if len(ok_df["latency_sec"].dropna()) else None,
        "retrieved_has_structured_rate": float(ok_df["retrieved_has_structured"].mean()) if len(ok_df) else None,
        "graded_has_structured_rate": float(ok_df["graded_has_structured"].mean()) if len(ok_df) else None,
        "avg_retrieved_figure_description_count": float(ok_df["retrieved_figure_description_count"].mean()) if len(ok_df) else None,
        "avg_graded_figure_description_count": float(ok_df["graded_figure_description_count"].mean()) if len(ok_df) else None,
        "avg_retrieved_table_count": float(ok_df["retrieved_table_count"].mean()) if len(ok_df) else None,
        "avg_graded_table_count": float(ok_df["graded_table_count"].mean()) if len(ok_df) else None,
        "avg_retrieved_diagram_count": float(ok_df["retrieved_diagram_count"].mean()) if len(ok_df) else None,
        "avg_graded_diagram_count": float(ok_df["graded_diagram_count"].mean()) if len(ok_df) else None,
        "avg_retrieved_equation_count": float(ok_df["retrieved_equation_count"].mean()) if len(ok_df) else None,
        "avg_graded_equation_count": float(ok_df["graded_equation_count"].mean()) if len(ok_df) else None,
        "avg_unsupported_claim_count": float(ok_df["unsupported_claim_count"].mean()) if len(ok_df) else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze multimodal grounding on debug results.")
    parser.add_argument("--results", type=str, required=True, help="Path to saved debug results JSON")
    parser.add_argument("--out-prefix", type=str, required=True, help="Prefix for output files")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_prefix = args.out_prefix

    detail_csv_path = Path(f"{out_prefix}_detailed.csv")
    summary_json_path = Path(f"{out_prefix}_summary.json")

    results = load_json(results_path)
    rows = build_rows(results)
    df = pd.DataFrame(rows)

    detail_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(detail_csv_path, index=False)

    summary = summarize(df)
    save_json(summary, summary_json_path)

    print("\n====================")
    print("MULTIMODAL GROUNDING ANALYSIS COMPLETE")
    print("====================")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved detailed analysis to: {detail_csv_path}")
    print(f"Saved summary JSON to: {summary_json_path}")


if __name__ == "__main__":
    main()