import json
import warnings
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

# Offline evaluation of previously saved answer-generation results.
# This script does NOT call the /ask API. It reads the saved JSON file directly.

RESULTS_PATH = Path("./results_simple_hybrid_rag_2.json")
DETAIL_CSV_PATH = Path("./evaluation_results_detailed.csv")
SUMMARY_JSON_PATH = Path("./evaluation_summary.json")
CATEGORY_CSV_PATH = Path("./evaluation_by_category.csv")
DIFFICULTY_CSV_PATH = Path("./evaluation_by_difficulty.csv")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_or_none(series: pd.Series):
    return float(series.mean()) if len(series) else None


def build_dataset_from_results(results: list[dict]):
    kept_rows = []
    for row in results:
        if row.get("status") != "ok":
            continue
        if not row.get("answer", "").strip():
            continue
        contexts = row.get("contexts", [])
        if not isinstance(contexts, list) or len(contexts) == 0:
            continue

        kept_rows.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": contexts,
                "ground_truth": row["ground_truth"],
            }
        )

    if not kept_rows:
        return None

    return Dataset.from_list(kept_rows)


def evaluate_subset(df_subset: pd.DataFrame, evaluator_llm, evaluator_embeddings):
    if df_subset.empty:
        return None

    dataset = Dataset.from_dict(
        {
            "question": df_subset["question"].tolist(),
            "answer": df_subset["answer"].tolist(),
            "contexts": df_subset["contexts"].tolist(),
            "ground_truth": df_subset["ground_truth"].tolist(),
        }
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            AnswerCorrectness(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=1, timeout=360, max_retries=10, max_wait=60),
    )

    return results.to_pandas()


def summarize_metrics(df: pd.DataFrame):
    return {
        "faithfulness": mean_or_none(df["faithfulness"]) if "faithfulness" in df else None,
        "answer_relevancy": mean_or_none(df["answer_relevancy"]) if "answer_relevancy" in df else None,
        "answer_correctness": mean_or_none(df["answer_correctness"]) if "answer_correctness" in df else None,
        "context_precision": mean_or_none(df["context_precision"]) if "context_precision" in df else None,
        "context_recall": mean_or_none(df["context_recall"]) if "context_recall" in df else None,
    }


def main():
    print(f"Loading saved results from: {RESULTS_PATH}")
    results = load_json(RESULTS_PATH)

    total_rows = len(results)
    ok_rows = sum(1 for r in results if r.get("status") == "ok")
    failed_rows = total_rows - ok_rows

    print(f"Total rows: {total_rows}")
    print(f"Successful rows: {ok_rows}")
    print(f"Failed rows: {failed_rows}")

    dataset = build_dataset_from_results(results)
    if dataset is None:
        print("No valid successful rows found for evaluation.")
        return

    print("Setting up offline RAGAS evaluator (Ollama)...")
    # evaluator_llm = LangchainLLMWrapper(
    # ChatGroq(
    #     model="meta-llama/llama-4-scout-17b-16e-instruct",
    #     temperature=0,max_tokens=4096,      # Increase this to fix LLMDidNotFinishException
    #     n=1,                  # Explicitly force n=1 for Groq compatibility
    #     model_kwargs={
    #         "top_p": 1,        # Added for stability
    #     }))

    # evaluator_embeddings = LangchainEmbeddingsWrapper(
    #     HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    # )
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:14b", temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    print("Running full 5-metric RAGAS evaluation on saved results...")
    overall_results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            AnswerCorrectness(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=1, timeout=360, max_retries=10, max_wait=60),
    )

    detailed_df = overall_results.to_pandas()

    merge_meta_rows = []
    for row in results:
        if row.get("status") != "ok":
            continue
        if not row.get("answer", "").strip():
            continue
        contexts = row.get("contexts", [])
        if not isinstance(contexts, list) or len(contexts) == 0:
            continue

        merge_meta_rows.append(
            {
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "architecture": row.get("architecture"),
                "source_file": json.dumps(row.get("source_file", []), ensure_ascii=False),
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "latency_sec": row.get("latency_sec"),
                "answer": row.get("answer"),
                "contexts": json.dumps(row.get("contexts", []), ensure_ascii=False),
            }
        )

    meta_df = pd.DataFrame(merge_meta_rows)
    if len(meta_df) == len(detailed_df):
        detailed_df = pd.concat([meta_df.reset_index(drop=True), detailed_df.reset_index(drop=True)], axis=1)

    detailed_df.to_csv(DETAIL_CSV_PATH, index=False)

    overall_summary = summarize_metrics(detailed_df)
    overall_summary["architecture"] = results[0].get("architecture") if results else None
    overall_summary["total_rows"] = total_rows
    overall_summary["successful_rows"] = ok_rows
    overall_summary["failed_rows"] = failed_rows
    overall_summary["failure_rate"] = (failed_rows / total_rows) if total_rows else None
    overall_summary["avg_latency_sec"] = float(pd.Series([r.get("latency_sec", 0) for r in results if r.get("status") == "ok"]).mean()) if ok_rows else None

    category_rows = []
    if "category" in detailed_df.columns:
        for category in sorted(detailed_df["category"].dropna().unique()):
            subset = detailed_df[detailed_df["category"] == category].copy()
            subset["contexts"] = subset["contexts"].apply(json.loads)
            subset_eval_df = evaluate_subset(subset, evaluator_llm, evaluator_embeddings)
            if subset_eval_df is None:
                continue
            category_summary = summarize_metrics(subset_eval_df)
            category_summary["category"] = category
            category_summary["count"] = int(len(subset))
            category_summary["avg_latency_sec"] = float(subset["latency_sec"].mean()) if "latency_sec" in subset else None
            category_rows.append(category_summary)

    category_df = pd.DataFrame(category_rows)
    if not category_df.empty:
        category_df.to_csv(CATEGORY_CSV_PATH, index=False)

    difficulty_rows = []
    if "difficulty" in detailed_df.columns:
        for difficulty in sorted(detailed_df["difficulty"].dropna().unique()):
            subset = detailed_df[detailed_df["difficulty"] == difficulty].copy()
            subset["contexts"] = subset["contexts"].apply(json.loads)
            subset_eval_df = evaluate_subset(subset, evaluator_llm, evaluator_embeddings)
            if subset_eval_df is None:
                continue
            difficulty_summary = summarize_metrics(subset_eval_df)
            difficulty_summary["difficulty"] = difficulty
            difficulty_summary["count"] = int(len(subset))
            difficulty_summary["avg_latency_sec"] = float(subset["latency_sec"].mean()) if "latency_sec" in subset else None
            difficulty_rows.append(difficulty_summary)

    difficulty_df = pd.DataFrame(difficulty_rows)
    if not difficulty_df.empty:
        difficulty_df.to_csv(DIFFICULTY_CSV_PATH, index=False)

    summary = {
        "overall": overall_summary,
        "by_category_rows": category_rows,
        "by_difficulty_rows": difficulty_rows,
        "detail_csv": str(DETAIL_CSV_PATH),
        "category_csv": str(CATEGORY_CSV_PATH),
        "difficulty_csv": str(DIFFICULTY_CSV_PATH),
    }

    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n====================")
    print("OFFLINE EVALUATION COMPLETE")
    print("====================")
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False))
    print(f"\nSaved detailed metrics to: {DETAIL_CSV_PATH}")
    print(f"Saved summary JSON to: {SUMMARY_JSON_PATH}")
    if not category_df.empty:
        print(f"Saved category breakdown to: {CATEGORY_CSV_PATH}")
    if not difficulty_df.empty:
        print(f"Saved difficulty breakdown to: {DIFFICULTY_CSV_PATH}")


if __name__ == "__main__":
    main()
