import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


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


def token_overlap_stats(answer: str, ground_truth: str) -> Dict[str, float]:
    pred_tokens = tokenize(answer)
    gold_tokens = tokenize(ground_truth)

    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = pred_counts & gold_counts
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


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


def doc_sources(docs: List[Dict[str, Any]]) -> List[str]:
    out = []
    for doc in docs or []:
        meta = doc.get("metadata", {}) or {}
        src = meta.get("source_file")
        if src:
            out.append(str(src))
    return out


def doc_content_types(docs: List[Dict[str, Any]]) -> List[str]:
    out = []
    for doc in docs or []:
        meta = doc.get("metadata", {}) or {}
        ctype = meta.get("content_type")
        if ctype:
            out.append(str(ctype))
    return out


def has_unsupported_claims(claims: List[Dict[str, Any]]) -> bool:
    for c in claims or []:
        if not bool(c.get("supported", False)):
            return True
    return False


def classify_failure(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns:
        (primary_failure_type, explanation)
    """
    if row.get("status") != "ok":
        return ("backend_or_request_error", "The request did not complete successfully.")

    answer = row.get("answer", "") or ""
    ground_truth = row.get("ground_truth", "") or ""
    category = row.get("category")
    gold_sources = to_source_list(row.get("source_file"))
    predicted_sources = extract_predicted_sources(row.get("citations", []))

    retrieved_docs = row.get("retrieved_docs", []) or []
    candidate_docs = row.get("candidate_docs", []) or []
    graded_docs = row.get("graded_docs", []) or []
    claim_verification = row.get("claim_verification", []) or []

    retrieved_sources = set(doc_sources(retrieved_docs))
    candidate_sources = set(doc_sources(candidate_docs))
    graded_sources = set(doc_sources(graded_docs))
    gold_source_set = set(gold_sources)

    selected_content_types = set(doc_content_types(graded_docs))
    overlap = token_overlap_stats(answer, ground_truth)
    source_view = source_metrics(gold_sources, predicted_sources)
    rouge = rouge_l_f1(answer, ground_truth)

    gold_retrieved = len(gold_source_set & retrieved_sources) > 0
    gold_candidate = len(gold_source_set & candidate_sources) > 0
    gold_selected = len(gold_source_set & graded_sources) > 0
    unsupported = has_unsupported_claims(claim_verification)

    # 1. Strongest direct signal: unsupported claims
    if unsupported or row.get("citations_pass") is False:
        return (
            "unsupported_claim",
            "Claim verification reported unsupported or insufficiently grounded claims.",
        )

    # 2. Figure/table-specific miss
    if category == "figure_table_diagram_grounded":
        useful_types = {"figure_description", "table", "diagram_text", "equation_block"}
        has_useful_structured_doc = len(selected_content_types & useful_types) > 0
        if not has_useful_structured_doc and (overlap["f1"] < 0.6 or rouge < 0.6):
            return (
                "figure_table_miss",
                "The question is figure/table grounded, but no structured figure/table/equation chunk was selected.",
            )

    # 3. Cross-paper imbalance
    if category == "cross_paper_comparison" and len(gold_source_set) >= 2:
        selected_gold_count = len(gold_source_set & graded_sources)
        cited_gold_count = len(gold_source_set & set(predicted_sources))
        if max(selected_gold_count, cited_gold_count) < len(gold_source_set):
            return (
                "cross_paper_imbalance",
                "The answer appears to cover only part of a multi-source comparison.",
            )

    # 4. Retrieval miss
    if not gold_retrieved and source_view["source_hit"] == 0.0:
        return (
            "retrieval_miss",
            "No gold source appears in retrieved documents or final citations.",
        )

    # 5. Evidence found but not selected
    if gold_retrieved and not gold_selected:
        return (
            "evidence_found_but_not_selected",
            "A relevant source appeared in retrieval, but it did not survive selection/grading.",
        )

    # 6. Evidence selected but answer still weak
    if gold_selected and (overlap["f1"] < 0.5 and rouge < 0.5):
        return (
            "evidence_selected_but_misused",
            "Relevant evidence was selected, but the final answer still mismatched the ground truth.",
        )

    # 7. Incomplete answer
    if answer.strip() == "" or overlap["recall"] < 0.5:
        return (
            "answer_incomplete",
            "The final answer missed too much of the target information.",
        )

    # 8. Too broad / imprecise answer
    if overlap["precision"] < 0.45 and len(tokenize(answer)) > max(12, int(1.7 * len(tokenize(ground_truth)))):
        return (
            "answer_too_broad",
            "The final answer appears broader or noisier than the target answer.",
        )

    return (
        "other_or_minor",
        "No strong failure signature matched the current taxonomy rules.",
    )


def build_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in results:
        answer = row.get("answer", "")
        ground_truth = row.get("ground_truth", "")
        predicted_sources = extract_predicted_sources(row.get("citations", []))
        source_view = source_metrics(row.get("source_file"), predicted_sources)
        overlap = token_overlap_stats(answer, ground_truth)
        rouge = rouge_l_f1(answer, ground_truth)
        failure_type, explanation = classify_failure(row)

        rows.append(
            {
                "question": row.get("question"),
                "architecture": row.get("architecture"),
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "status": row.get("status"),
                "latency_sec": row.get("latency_sec"),
                "failure_type": failure_type,
                "failure_explanation": explanation,
                "token_precision": overlap["precision"] if row.get("status") == "ok" else None,
                "token_recall": overlap["recall"] if row.get("status") == "ok" else None,
                "token_f1": overlap["f1"] if row.get("status") == "ok" else None,
                "rouge_l_f1": rouge if row.get("status") == "ok" else None,
                "source_hit": source_view["source_hit"] if row.get("status") == "ok" else None,
                "source_precision": source_view["source_precision"] if row.get("status") == "ok" else None,
                "source_recall": source_view["source_recall"] if row.get("status") == "ok" else None,
                "source_mrr": source_view["source_mrr"] if row.get("status") == "ok" else None,
                "gold_sources": json.dumps(to_source_list(row.get("source_file")), ensure_ascii=False),
                "predicted_sources": json.dumps(predicted_sources, ensure_ascii=False),
                "retrieved_source_count": len(set(doc_sources(row.get("retrieved_docs", [])))),
                "candidate_source_count": len(set(doc_sources(row.get("candidate_docs", [])))),
                "graded_source_count": len(set(doc_sources(row.get("graded_docs", [])))),
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
    out: Dict[str, Any] = {}

    out["total_rows"] = int(len(df))
    out["by_failure_type"] = (
        df["failure_type"].value_counts(dropna=False).to_dict() if "failure_type" in df.columns else {}
    )

    by_category = {}
    if "category" in df.columns:
        for category, subset in df.groupby("category", dropna=False):
            key = str(category)
            by_category[key] = subset["failure_type"].value_counts(dropna=False).to_dict()
    out["by_category"] = by_category

    by_difficulty = {}
    if "difficulty" in df.columns:
        for difficulty, subset in df.groupby("difficulty", dropna=False):
            key = str(difficulty)
            by_difficulty[key] = subset["failure_type"].value_counts(dropna=False).to_dict()
    out["by_difficulty"] = by_difficulty

    return out


def build_examples(df: pd.DataFrame, max_per_type: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for _, row in df.iterrows():
        ftype = row["failure_type"]
        if len(examples[ftype]) >= max_per_type:
            continue

        examples[ftype].append(
            {
                "question": row["question"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "failure_explanation": row["failure_explanation"],
                "token_f1": row["token_f1"],
                "rouge_l_f1": row["rouge_l_f1"],
                "source_hit": row["source_hit"],
                "source_recall": row["source_recall"],
                "unsupported_claim_count": row["unsupported_claim_count"],
                "auditor_feedback": row["auditor_feedback"],
            }
        )

    return dict(examples)


def main():
    parser = argparse.ArgumentParser(description="Structured error analysis for debug RAG results.")
    parser.add_argument("--results", type=str, required=True, help="Path to saved debug results JSON")
    parser.add_argument("--out-prefix", type=str, required=True, help="Prefix for output files")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_prefix = args.out_prefix

    detail_csv_path = Path(f"{out_prefix}_detailed.csv")
    summary_json_path = Path(f"{out_prefix}_summary.json")
    category_csv_path = Path(f"{out_prefix}_by_category.csv")
    difficulty_csv_path = Path(f"{out_prefix}_by_difficulty.csv")

    print(f"Loading debug results from: {results_path}")
    results = load_json(results_path)

    rows = build_rows(results)
    df = pd.DataFrame(rows)
    detail_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(detail_csv_path, index=False)

    summary = summarize(df)
    summary["architecture"] = results[0].get("architecture") if results else None
    summary["examples"] = build_examples(df)

    save_json(summary, summary_json_path)

    if "category" in df.columns:
        cat_rows = []
        for category, subset in df.groupby("category", dropna=False):
            counts = subset["failure_type"].value_counts(dropna=False).to_dict()
            row = {"category": category, "count": int(len(subset))}
            row.update({f"failure__{k}": v for k, v in counts.items()})
            cat_rows.append(row)
        pd.DataFrame(cat_rows).to_csv(category_csv_path, index=False)

    if "difficulty" in df.columns:
        diff_rows = []
        for difficulty, subset in df.groupby("difficulty", dropna=False):
            counts = subset["failure_type"].value_counts(dropna=False).to_dict()
            row = {"difficulty": difficulty, "count": int(len(subset))}
            row.update({f"failure__{k}": v for k, v in counts.items()})
            diff_rows.append(row)
        pd.DataFrame(diff_rows).to_csv(difficulty_csv_path, index=False)

    print("\n====================")
    print("STRUCTURED ERROR ANALYSIS COMPLETE")
    print("====================")
    print(json.dumps(summary["by_failure_type"], indent=2, ensure_ascii=False))
    print(f"\nSaved detailed analysis to: {detail_csv_path}")
    print(f"Saved summary JSON to: {summary_json_path}")
    print(f"Saved category breakdown to: {category_csv_path}")
    print(f"Saved difficulty breakdown to: {difficulty_csv_path}")


if __name__ == "__main__":
    main()