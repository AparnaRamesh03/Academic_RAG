"""
benchmark.py — Ideal Unified Benchmark for Academic RAG Architectures
======================================================================

Combines non-LLM metrics (fast, free, deterministic) with a curated
subset of RAGAS metrics (semantic, LLM-judged) into one offline script.

Metrics computed
────────────────
  Non-LLM  (no model needed, instant):
    • Exact Match (EM)           — strict string equality
    • Token F1                   — word overlap
    • ROUGE-L F1                 — longest common subsequence
    • Source Hit                 — correct paper cited (any position)
    • Source Precision           — fraction of cited papers that are correct
    • Source Recall              — fraction of correct papers cited
    • Source MRR                 — mean reciprocal rank of first correct citation

  RAGAS    (requires local Ollama judge + nomic-embed-text):
    • Faithfulness               — hallucination detector (most important)
    • Context Precision          — are retrieved chunks relevant?
    • Context Recall             — do chunks contain needed information?

  Skipped intentionally:
    • Answer Relevancy  — highly correlated with Token F1 for this task
    • Answer Correctness (RAGAS) — ROUGE-L + Token F1 already cover this

Usage
─────
  # Compare architectures from saved result JSONs (no backend needed):
  python benchmark.py

  # Specify which result files to compare:
  python benchmark.py --results results_simple_hybrid_rag.json results_final_arch.json results_agentic_scholar.json

  # Skip RAGAS (fast mode — only non-LLM metrics):
  python benchmark.py --no-ragas

  # Use Groq as judge instead of Ollama:
  python benchmark.py --judge groq

  # Save outputs to a custom prefix:
  python benchmark.py --out-prefix my_run

Prerequisites
─────────────
  pip install ragas datasets pandas requests langchain-ollama langchain-groq
  Ollama running with: ollama pull qwen2.5:14b && ollama pull nomic-embed-text
"""

import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_scholar_note(text: str) -> str:
    """Remove the agentic_scholar ingest note before computing metrics."""
    return re.split(r"\n\n---\n\*[Nn]ote:", text)[0].strip()


def normalize(text: str) -> str:
    text = _strip_scholar_note(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


# ═══════════════════════════════════════════════════════════════════════════════
# NON-LLM METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def exact_match(answer: str, ground_truth: str) -> float:
    return float(normalize(answer) == normalize(ground_truth))


def token_f1(answer: str, ground_truth: str) -> float:
    pred = tokenize(answer)
    gold = tokenize(ground_truth)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pred)
    r = n / len(gold)
    return 2 * p * r / (p + r)


def _lcs(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def rouge_l(answer: str, ground_truth: str) -> float:
    pred = tokenize(answer)
    gold = tokenize(ground_truth)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    lcs = _lcs(pred, gold)
    p = lcs / len(pred)
    r = lcs / len(gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _to_source_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(v) for v in value}
    return {str(value)}


def _predicted_sources(citations) -> list[str]:
    if not isinstance(citations, list):
        return []
    return [str(c["source_file"]) for c in citations if isinstance(c, dict) and c.get("source_file")]


def source_metrics(gold_sources, citations) -> dict:
    gold = _to_source_set(gold_sources)
    pred = _predicted_sources(citations)
    pred_set = set(pred)
    inter = gold & pred_set

    mrr = next((1.0 / (i + 1) for i, s in enumerate(pred) if s in gold), 0.0)

    return {
        "source_hit":       float(len(inter) > 0),
        "source_precision": len(inter) / len(pred_set) if pred_set else 0.0,
        "source_recall":    len(inter) / len(gold)     if gold     else 0.0,
        "source_mrr":       mrr,
    }


def score_non_llm(row: dict) -> dict:
    answer      = row.get("answer", "")
    ground_truth = row.get("ground_truth", "")
    citations   = row.get("citations", [])

    src = source_metrics(row.get("source_file"), citations)
    return {
        "exact_match":  exact_match(answer, ground_truth),
        "token_f1":     token_f1(answer, ground_truth),
        "rouge_l_f1":   rouge_l(answer, ground_truth),
        **src,
        "latency_sec":  row.get("latency_sec"),
    }


NON_LLM_COLS = [
    "exact_match", "token_f1", "rouge_l_f1",
    "source_hit", "source_precision", "source_recall", "source_mrr",
    "latency_sec",
]


# ═══════════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

RAGAS_METRICS_NAMES = ["faithfulness", "context_precision", "context_recall"]


def _build_ragas_judge(judge: str):
    """Return (llm_wrapper, embeddings_wrapper) for the chosen judge."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if judge == "groq":
        import os
        from dotenv import load_dotenv
        load_dotenv()
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        llm = LangchainLLMWrapper(ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            max_tokens=4096,
        ))
        emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        )
    else:  # ollama (default)
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:14b", temperature=0))
        emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    return llm, emb


def run_ragas(rows: list[dict], judge: str) -> pd.DataFrame | None:
    """
    Run Faithfulness + Context Precision + Context Recall on a list of rows.
    Each row must have: question, answer, contexts (list[str]), ground_truth.
    Returns a DataFrame with one row per input row, or None on failure.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall
        from ragas.run_config import RunConfig
    except ImportError as e:
        print(f"  [RAGAS] Import error: {e}")
        print("  Run: pip install ragas datasets")
        return None

    valid_rows = [
        r for r in rows
        if r.get("answer", "").strip()
        and isinstance(r.get("contexts"), list)
        and len(r["contexts"]) > 0
    ]

    if not valid_rows:
        print("  [RAGAS] No valid rows to evaluate (need non-empty answer + contexts).")
        return None

    # Strip scholar notes from answers before RAGAS sees them
    for r in valid_rows:
        r = r.copy()
        r["answer"] = _strip_scholar_note(r.get("answer", ""))

    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       _strip_scholar_note(r.get("answer", "")),
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in valid_rows
    ])

    print(f"  [RAGAS] Evaluating {len(valid_rows)} rows with judge='{judge}'...")
    llm, emb = _build_ragas_judge(judge)

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ContextPrecision(), ContextRecall()],
            llm=llm,
            embeddings=emb,
            run_config=RunConfig(max_workers=1, timeout=360, max_retries=5, max_wait=60),
        )
        return results.to_pandas()
    except Exception as exc:
        print(f"  [RAGAS] Evaluation failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

ALL_METRIC_COLS = NON_LLM_COLS + RAGAS_METRICS_NAMES


def agg(df: pd.DataFrame, cols: list[str] | None = None) -> dict:
    if cols is None:
        cols = ALL_METRIC_COLS
    result = {}
    for col in cols:
        if col in df.columns:
            valid = df[col].dropna()
            result[col] = float(valid.mean()) if len(valid) else None
        else:
            result[col] = None
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PRINTING
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(val, is_pct: bool = True) -> str:
    if val is None:
        return "   —  "
    return f"{val * 100:5.1f}%" if is_pct else f"{val:5.2f}s"


def print_table(title: str, rows: list[dict], metrics: list[tuple]) -> None:
    """
    metrics: list of (label, key, is_pct)
    """
    arch_w = max((len(r["architecture"]) for r in rows), default=12) + 2
    metric_w = 8

    sep = "─" * (arch_w + metric_w * len(metrics))
    print()
    print(f"  {title}")
    print("  " + "═" * (len(sep) - 2))
    header = f"  {'Architecture':<{arch_w - 2}}" + "".join(f"{lbl:>{metric_w}}" for lbl, _, _ in metrics)
    print(header)
    print("  " + sep)
    for row in rows:
        arch = row["architecture"]
        values = "".join(_fmt(row.get(key), pct) for _, key, pct in metrics)
        print(f"  {arch:<{arch_w - 2}}{values}")
    print("  " + sep)


def print_full_comparison(summary_rows: list[dict], has_ragas: bool) -> None:
    non_llm_metrics = [
        ("EM",       "exact_match",      True),
        ("Tok-F1",   "token_f1",         True),
        ("ROUGE-L",  "rouge_l_f1",       True),
        ("Src-Hit",  "source_hit",       True),
        ("Src-P",    "source_precision", True),
        ("Src-R",    "source_recall",    True),
        ("Src-MRR",  "source_mrr",       True),
        ("Latency",  "latency_sec",      False),
    ]

    print("\n" + "═" * 80)
    print("  ACADEMIC RAG — IDEAL BENCHMARK RESULTS")
    print("═" * 80)

    print_table("NON-LLM METRICS  (deterministic)", summary_rows, non_llm_metrics)

    if has_ragas:
        ragas_metrics = [
            ("Faith.",   "faithfulness",      True),
            ("Ctx-P",    "context_precision", True),
            ("Ctx-R",    "context_recall",    True),
        ]
        print()
        print_table("RAGAS METRICS  (LLM-judged, semantic)", summary_rows, ragas_metrics)

    # Per-category breakdown (non-LLM only for readability)
    all_cats = sorted({
        cat
        for r in summary_rows
        for cat in r.get("by_category", {}).keys()
    })
    if all_cats:
        print()
        print("  PER-CATEGORY BREAKDOWN  (Token F1 | ROUGE-L | Source Hit | Latency)")
        print("  " + "─" * 70)
        cat_metrics = [
            ("Tok-F1",  "token_f1",    True),
            ("ROUGE-L", "rouge_l_f1",  True),
            ("Src-Hit", "source_hit",  True),
            ("Latency", "latency_sec", False),
        ]
        arch_w = max((len(r["architecture"]) for r in summary_rows), default=12) + 2
        metric_w = 8
        for cat in all_cats:
            label = cat.replace("_", " ").title()
            print(f"\n  [{label}]")
            for row in summary_rows:
                cat_data = row.get("by_category", {}).get(cat, {})
                values = "".join(_fmt(cat_data.get(k), p) for _, k, p in cat_metrics)
                print(f"    {row['architecture']:<{arch_w}}{values}")

    # Difficulty breakdown
    all_diffs = sorted({
        d
        for r in summary_rows
        for d in r.get("by_difficulty", {}).keys()
    })
    if all_diffs:
        print()
        diff_metrics = [
            ("Tok-F1",  "token_f1",    True),
            ("ROUGE-L", "rouge_l_f1",  True),
            ("Src-Hit", "source_hit",  True),
            ("Latency", "latency_sec", False),
        ]
        arch_w = max((len(r["architecture"]) for r in summary_rows), default=12) + 2
        print("  PER-DIFFICULTY BREAKDOWN  (Token F1 | ROUGE-L | Source Hit | Latency)")
        print("  " + "─" * 70)
        for diff in all_diffs:
            print(f"\n  [{diff.upper()}]")
            for row in summary_rows:
                diff_data = row.get("by_difficulty", {}).get(diff, {})
                values = "".join(_fmt(diff_data.get(k), p) for _, k, p in diff_metrics)
                print(f"    {row['architecture']:<{arch_w}}{values}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_FILES = [
    "results_simple_hybrid_rag.json",
    "results_final_arch.json",
    "results_agentic_scholar.json",
]


def main():
    parser = argparse.ArgumentParser(
        description="Ideal unified benchmark: non-LLM metrics + RAGAS (Faithfulness, Ctx-P, Ctx-R)."
    )
    parser.add_argument(
        "--results", nargs="*", default=None,
        help=f"Result JSON files to compare. Default: {DEFAULT_FILES}",
    )
    parser.add_argument(
        "--no-ragas", action="store_true",
        help="Skip RAGAS evaluation (fast mode, non-LLM metrics only).",
    )
    parser.add_argument(
        "--judge", choices=["ollama", "groq"], default="ollama",
        help="LLM judge for RAGAS: 'ollama' (qwen2.5:14b, default) or 'groq'.",
    )
    parser.add_argument(
        "--out-prefix", default="benchmark",
        help="Prefix for all output files (default: 'benchmark').",
    )
    args = parser.parse_args()

    result_files = [Path(p) for p in (args.results or DEFAULT_FILES)]
    run_ragas_flag = not args.no_ragas

    existing_files = []
    for p in result_files:
        if p.exists():
            existing_files.append(p)
        else:
            print(f"[SKIP] File not found: {p}")

    if not existing_files:
        print(
            "\nNo result files found. Collect results first:\n"
            "  python generate_results.py --arch simple_hybrid_rag\n"
            "  python generate_results.py --arch final_arch\n"
            "  python generate_results.py --arch agentic_scholar\n"
        )
        return

    # ── Process each result file ──────────────────────────────────────────────
    summary_rows = []
    all_detail_rows = []

    for path in existing_files:
        print(f"\n{'─'*60}")
        print(f"Processing: {path}")

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not raw:
            print("  Empty file, skipping.")
            continue

        arch = raw[0].get("architecture", path.stem)
        ok_rows = [r for r in raw if r.get("status") == "ok"]
        print(f"  {len(ok_rows)}/{len(raw)} successful responses.")

        if not ok_rows:
            print("  No successful rows, skipping.")
            continue

        # ── Non-LLM metrics ───────────────────────────────────────────────────
        detail_rows = []
        for row in ok_rows:
            scores = score_non_llm(row)
            detail_rows.append({
                "architecture": arch,
                "question":     row.get("question"),
                "category":     row.get("category"),
                "difficulty":   row.get("difficulty"),
                "answer":       row.get("answer", ""),
                "ground_truth": row.get("ground_truth", ""),
                "contexts":     row.get("contexts", []),
                **scores,
            })

        df = pd.DataFrame(detail_rows)

        # ── RAGAS ─────────────────────────────────────────────────────────────
        ragas_df = None
        if run_ragas_flag:
            print(f"  Running RAGAS (Faithfulness, Ctx-P, Ctx-R) with judge='{args.judge}'...")
            ragas_rows = [
                {
                    "question":     r.get("question"),
                    "answer":       r.get("answer", ""),
                    "contexts":     r.get("contexts", []),
                    "ground_truth": r.get("ground_truth", ""),
                }
                for r in ok_rows
            ]
            ragas_df = run_ragas(ragas_rows, args.judge)
            if ragas_df is not None and len(ragas_df) == len(df):
                for col in RAGAS_METRICS_NAMES:
                    if col in ragas_df.columns:
                        df[col] = ragas_df[col].values
                print(f"  RAGAS complete. Faithfulness={df['faithfulness'].mean():.3f}" if "faithfulness" in df else "")
            else:
                print("  RAGAS failed or row count mismatch — RAGAS columns will be empty.")

        all_detail_rows.extend(df.to_dict("records"))

        # ── Aggregate ─────────────────────────────────────────────────────────
        overall = agg(df)
        overall["architecture"] = arch
        overall["n_ok"] = len(ok_rows)
        overall["n_total"] = len(raw)

        by_category = {}
        if "category" in df.columns:
            for cat, subset in df.groupby("category", dropna=True):
                by_category[str(cat)] = agg(subset)

        by_difficulty = {}
        if "difficulty" in df.columns:
            for diff, subset in df.groupby("difficulty", dropna=True):
                by_difficulty[str(diff)] = agg(subset)

        summary_rows.append({
            **overall,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        })

    if not summary_rows:
        print("\nNo data to compare.")
        return

    # ── Print final comparison tables ─────────────────────────────────────────
    has_ragas = run_ragas_flag and any(
        r.get("faithfulness") is not None for r in summary_rows
    )
    print_full_comparison(summary_rows, has_ragas)

    # ── Save outputs ──────────────────────────────────────────────────────────
    prefix = args.out_prefix

    # Summary CSV (one row per architecture)
    flat = [{k: v for k, v in r.items() if k not in ("by_category", "by_difficulty")} for r in summary_rows]
    summary_df = pd.DataFrame(flat)
    summary_path = Path(f"{prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary CSV         → {summary_path}")

    # Detailed CSV (one row per question × architecture)
    if all_detail_rows:
        detail_df = pd.DataFrame(all_detail_rows)
        # Drop the raw contexts column (too large for CSV)
        if "contexts" in detail_df.columns:
            detail_df = detail_df.drop(columns=["contexts"])
        detail_path = Path(f"{prefix}_detailed.csv")
        detail_df.to_csv(detail_path, index=False)
        print(f"Saved per-question detail → {detail_path}")

    # Summary JSON
    json_path = Path(f"{prefix}_summary.json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [{k: v for k, v in r.items()} for r in summary_rows],
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved summary JSON         → {json_path}")


if __name__ == "__main__":
    main()
