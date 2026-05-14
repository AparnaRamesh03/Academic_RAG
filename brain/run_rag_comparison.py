"""
run_rag_comparison.py
---------------------
Evaluates Simple RAG, Combined RAG (final_arch), and loads existing
MADDPG results — all on the same 5 test questions.

Usage (from brain/):
  python run_rag_comparison.py
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

_BRAIN = Path(__file__).resolve().parent
load_dotenv(_BRAIN / ".env")
sys.path.insert(0, str(_BRAIN))

# ── NLP helpers ───────────────────────────────────────────────────────────────

def _tok(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def token_f1(pred: str, gold: str) -> float:
    p, g = set(_tok(pred)), set(_tok(gold))
    if not p or not g: return 0.0
    tp = len(p & g)
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def rouge_l(pred: str, gold: str) -> float:
    p, g = _tok(pred), _tok(gold)
    m, n = len(p), len(g)
    if not m or not n: return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if p[i-1]==g[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec, rec = lcs/m, lcs/n
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

# ── Benchmark ─────────────────────────────────────────────────────────────────

def load_benchmark(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

# ── Simple RAG runner ─────────────────────────────────────────────────────────

def run_simple_rag(questions: List[Dict]) -> List[Dict]:
    import importlib
    sys.path.insert(0, str(_BRAIN / "simple_hybrid_rag"))
    # Force fresh import in case combined_rag already loaded a 'graph' module
    for mod in ["graph", "node_generator", "config"]:
        sys.modules.pop(mod, None)
    graph_mod = importlib.import_module("graph")
    graph = graph_mod.build_graph()
    results = []
    for q in questions:
        query = q["question"]
        t0 = time.time()
        try:
            state = graph.invoke({
                "original_query": query, "search_query": query,
                "retrieved_docs": [], "candidate_docs": [],
                "weak_signal_docs": [], "graded_docs": [],
                "generation": "", "crag_retries": 0,
                "verify_retries": 0, "citations_pass": True,
                "auditor_feedback": "",
            })
            answer = state.get("generation", "")
            docs   = state.get("graded_docs", [])
        except Exception as e:
            print(f"  [err] simple_rag: {e}")
            answer, docs = "", []
        latency = time.time() - t0

        gold = q.get("ground_truth", "")
        results.append({
            "question_id":   q.get("question_id", ""),
            "question":      query,
            "policy":        "simple_rag",
            "final_answer":  answer,
            "final_status":  "accepted" if answer.strip() else "abstained",
            "token_f1":      round(token_f1(answer, gold), 4),
            "rouge_l":       round(rouge_l(answer, gold), 4),
            "latency_seconds": round(latency, 3),
            "num_steps":     2,
            "retrieved_chunks": docs,
        })
        print(f"  simple_rag  | {query[:50]:<50} | TF1={results[-1]['token_f1']:.3f} | {latency:.1f}s")
    return results

# ── Combined RAG runner ───────────────────────────────────────────────────────

def run_combined_rag(questions: List[Dict]) -> List[Dict]:
    import importlib
    # Remove simple_hybrid_rag from path, add final_arch
    try:
        sys.path.remove(str(_BRAIN / "simple_hybrid_rag"))
    except ValueError:
        pass
    sys.path.insert(0, str(_BRAIN / "final_arch"))
    # Evict all cached modules that conflict with final_arch versions
    for mod in list(sys.modules.keys()):
        if mod in ("graph", "node_generator", "config", "node_grader",
                   "node_rewriter", "node_auditor", "node_context_selector",
                   "node_retrieval_evaluator", "reranker_shared"):
            sys.modules.pop(mod, None)
    graph_mod = importlib.import_module("graph")
    graph = graph_mod.build_graph()
    results = []
    for q in questions:
        query = q["question"]
        t0 = time.time()
        try:
            state = graph.invoke({
                "original_query": query, "search_query": query,
                "retrieved_docs": [], "candidate_docs": [],
                "weak_signal_docs": [], "graded_docs": [],
                "generation": "", "crag_retries": 0,
                "verify_retries": 0, "citations_pass": True,
                "auditor_feedback": "",
            })
            answer = state.get("generation", "")
            docs   = state.get("graded_docs", [])
        except Exception as e:
            print(f"  [err] combined_rag: {e}")
            answer, docs = "", []
        latency = time.time() - t0

        gold = q.get("ground_truth", "")
        results.append({
            "question_id":    q.get("question_id", ""),
            "question":       query,
            "policy":         "combined_rag",
            "final_answer":   answer,
            "final_status":   "accepted" if answer.strip() else "abstained",
            "token_f1":       round(token_f1(answer, gold), 4),
            "rouge_l":        round(rouge_l(answer, gold), 4),
            "latency_seconds": round(latency, 3),
            "num_steps":      5,
            "retrieved_chunks": docs,
        })
        print(f"  combined_rag | {query[:50]:<50} | TF1={results[-1]['token_f1']:.3f} | {latency:.1f}s")
    return results

# ── Aggregate ─────────────────────────────────────────────────────────────────

def agg(results: List[Dict], label: str) -> Dict:
    n = len(results)
    if n == 0:
        return {"label": label, "n": 0}
    failures = sum(1 for r in results if not r.get("final_answer", "").strip())
    return {
        "label":          label,
        "n":              n,
        "mean_token_f1":  round(sum(r["token_f1"]  for r in results) / n, 4),
        "mean_rouge_l":   round(sum(r["rouge_l"]   for r in results) / n, 4),
        "mean_latency":   round(sum(r["latency_seconds"] for r in results) / n, 3),
        "failure_rate":   round(failures / n, 4),
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    test_path = _BRAIN / "maddpg/results/benchmark_splits/test.jsonl"
    questions = load_benchmark(test_path)[:5]
    print(f"[benchmark] {len(questions)} questions from {test_path.name}\n")

    out_dir = _BRAIN / "maddpg/results/rag_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Simple RAG ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Running: Simple RAG (retrieve → top-K → generate)")
    print("=" * 60)
    simple_results = run_simple_rag(questions)
    with open(out_dir / "simple_rag.jsonl", "w") as f:
        for r in simple_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Combined RAG ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Running: Combined RAG (retrieve → rerank → grade → generate → audit)")
    print("=" * 60)
    combined_results = run_combined_rag(questions)
    with open(out_dir / "combined_rag.jsonl", "w") as f:
        for r in combined_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Load existing MADDPG results ──────────────────────────────────────────
    maddpg_noceb_path = _BRAIN / "maddpg/results/noceb_50ep/maddpg_no_ceb_live.jsonl"
    discrete_path     = _BRAIN / "maddpg/results/noceb_50ep/discrete_marl_baseline.jsonl"

    maddpg_noceb, discrete = [], []
    if maddpg_noceb_path.exists():
        with open(maddpg_noceb_path) as f:
            maddpg_noceb = [json.loads(l) for l in f if l.strip()]
    if discrete_path.exists():
        with open(discrete_path) as f:
            discrete = [json.loads(l) for l in f if l.strip()]

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = [
        agg(simple_results,   "Simple RAG"),
        agg(combined_results, "Combined RAG"),
        agg(discrete,         "Discrete MARL"),
        agg(maddpg_noceb,     "MADDPG no-CEB (50ep)"),
    ]

    print("\n" + "=" * 72)
    print(f"  {'System':<25} {'TF1':>6} {'ROUGE-L':>8} {'Latency(s)':>11} {'Fail%':>7}")
    print("-" * 72)
    for r in rows:
        print(
            f"  {r['label']:<25} "
            f"{r['mean_token_f1']:>6.3f} "
            f"{r['mean_rouge_l']:>8.3f} "
            f"{r['mean_latency']:>11.1f} "
            f"{r['failure_rate']*100:>6.0f}%"
        )
    print("=" * 72)

    with open(out_dir / "comparison_all_systems.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[saved] {out_dir}/comparison_all_systems.json")


if __name__ == "__main__":
    main()
