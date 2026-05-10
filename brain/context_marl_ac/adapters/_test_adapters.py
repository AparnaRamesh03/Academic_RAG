"""
Smoke-test all four brain/context_marl_ac adapters in dry-run mode.
Run from repo root:
    python brain/context_marl_ac/adapters/_test_adapters.py
"""

import sys
import importlib
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[3]   # Academic_RAG/
BRAIN_ROOT = REPO_ROOT / "brain"
MARL_ROOT  = BRAIN_ROOT / "context_marl_ac"

for p in [str(REPO_ROOT), str(BRAIN_ROOT), str(MARL_ROOT.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Force DRY_RUN before importing any adapter ────────────────────────────
import context_marl_ac.config as _cfg
_cfg.DRY_RUN = True

# ── Colour helpers ────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"

passed = 0
failed = 0


def ok(label: str):
    global passed
    passed += 1
    print(f"  {GREEN}✓{RESET}  {label}")


def fail(label: str, err: Exception):
    global failed
    failed += 1
    print(f"  {RED}✗{RESET}  {label}")
    print(f"       {RED}{type(err).__name__}: {err}{RESET}")


def section(title: str):
    print(f"\n{BOLD}── {title} ──{RESET}")


# ══════════════════════════════════════════════════════════════════════════
# 1. config.py
# ══════════════════════════════════════════════════════════════════════════
section("config.py")

try:
    from context_marl_ac.config import (
        DRY_RUN, BRAIN_ROOT, ARCHITECTURE_NAME, FEATURE_DIM,
        MAX_STEPS_PER_EPISODE, LEARNING_RATE, W_ANSWER_QUALITY,
    )
    assert DRY_RUN is True,                            "DRY_RUN should be True"
    assert ARCHITECTURE_NAME == "context_marl_ac",     "Wrong architecture name"
    assert FEATURE_DIM == 14,                          "FEATURE_DIM should be 14"
    assert 0 < LEARNING_RATE < 1,                      "LEARNING_RATE out of range"
    assert 0 < W_ANSWER_QUALITY <= 1,                  "W_ANSWER_QUALITY out of range"
    ok("All config values present and sane")
except Exception as e:
    fail("config.py import / values", e)


# ══════════════════════════════════════════════════════════════════════════
# 2. retriever_adapter
# ══════════════════════════════════════════════════════════════════════════
section("retriever_adapter.py")

try:
    from context_marl_ac.adapters.retriever_adapter import (
        retrieve_hybrid,
        retrieve_dense,
        retrieve_sparse,
        retrieve_hybrid_rerank,
        retrieve_more,
    )
    ok("Module imports cleanly")
except Exception as e:
    fail("Module import", e)
    sys.exit(1)

QUERY = "What is a transformer model?"

for fn_name, fn, kwargs in [
    ("retrieve_hybrid",        retrieve_hybrid,        {"query": QUERY, "top_k": 2}),
    ("retrieve_dense",         retrieve_dense,         {"query": QUERY, "top_k": 2}),
    ("retrieve_sparse",        retrieve_sparse,        {"query": QUERY, "top_k": 2}),
    ("retrieve_hybrid_rerank", retrieve_hybrid_rerank, {"query": QUERY, "top_k": 2}),
    ("retrieve_more",          retrieve_more,          {"query": QUERY, "current_chunks": [], "top_k": 1}),
]:
    try:
        result = fn(**kwargs)
        assert isinstance(result, list),                       "Must return a list"
        assert len(result) > 0,                                "Must return at least one chunk"
        first = result[0]
        assert "text"     in first,                            "Missing 'text' key"
        assert "metadata" in first,                            "Missing 'metadata' key"
        assert "score"    in first,                            "Missing 'score' key"
        ok(f"{fn_name}() → {len(result)} chunk(s)")
    except Exception as e:
        fail(f"{fn_name}()", e)


# ══════════════════════════════════════════════════════════════════════════
# 3. llm_adapter
# ══════════════════════════════════════════════════════════════════════════
section("llm_adapter.py")

try:
    from context_marl_ac.adapters.llm_adapter import (
        rewrite_query,
        grade_chunks,
        generate_answer,
        verify_answer,
    )
    ok("Module imports cleanly")
except Exception as e:
    fail("Module import", e)
    sys.exit(1)

SAMPLE_CHUNKS = [
    {
        "text": "Transformers use self-attention.",
        "metadata": {"source_file": "paper.pdf", "page_number": 1,
                     "section_header": "Intro", "content_type": "text"},
        "score": 0.9,
    }
]

SAMPLE_EVIDENCE_PACK = [
    {
        "chunk_id": "c1",
        "source": "paper.pdf",
        "page": 1,
        "section": "Intro",
        "text": "Transformers use self-attention.",
        "retrieval_score": 0.9,
        "grade": "relevant",
        "citation_id": "[1]",
    }
]

try:
    rw = rewrite_query(QUERY, mode="simple_rewrite")
    assert isinstance(rw, str) and len(rw) > 0, "rewrite_query must return non-empty str"
    ok(f"rewrite_query() → '{rw[:60]}'")
except Exception as e:
    fail("rewrite_query()", e)

try:
    graded = grade_chunks(QUERY, SAMPLE_CHUNKS, mode="medium_filter")
    assert isinstance(graded, list), "grade_chunks must return a list"
    ok(f"grade_chunks() → {len(graded)} chunk(s)")
except Exception as e:
    fail("grade_chunks()", e)

try:
    ans = generate_answer(QUERY, SAMPLE_EVIDENCE_PACK, mode="generate_answer")
    assert isinstance(ans, str) and len(ans) > 0, "generate_answer must return non-empty str"
    ok(f"generate_answer() → '{ans[:80]}'")
except Exception as e:
    fail("generate_answer()", e)

try:
    vr = verify_answer(QUERY, "Transformers use self-attention.", SAMPLE_EVIDENCE_PACK)
    assert isinstance(vr, dict),                   "verify_answer must return dict"
    assert "decision" in vr,                       "Missing 'decision' key"
    assert "claims"   in vr,                       "Missing 'claims' key"
    assert vr["decision"] in ("PASS", "FAIL"),     "decision must be PASS or FAIL"
    ok(f"verify_answer() → decision={vr['decision']}, claims={len(vr['claims'])}")
except Exception as e:
    fail("verify_answer()", e)


# ══════════════════════════════════════════════════════════════════════════
# 4. citation_adapter
# ══════════════════════════════════════════════════════════════════════════
section("citation_adapter.py")

try:
    from context_marl_ac.adapters.citation_adapter import (
        build_citations,
        compute_citation_support,
        detect_unsupported_claims,
    )
    ok("Module imports cleanly")
except Exception as e:
    fail("Module import", e)
    sys.exit(1)

try:
    cites = build_citations(SAMPLE_CHUNKS)
    assert isinstance(cites, list),                    "build_citations must return list"
    assert len(cites) > 0,                             "Must return at least one citation"
    c = cites[0]
    assert "source_file" in c,                         "Missing 'source_file'"
    assert "excerpt"     in c,                         "Missing 'excerpt'"
    ok(f"build_citations() → {len(cites)} citation(s)")
except Exception as e:
    fail("build_citations()", e)

try:
    rate = compute_citation_support(
        "Transformers use self-attention.", [], SAMPLE_EVIDENCE_PACK
    )
    assert isinstance(rate, float),                    "Must return float"
    assert 0.0 <= rate <= 1.0,                         "Rate must be in [0, 1]"
    ok(f"compute_citation_support() → {rate:.2f}")
except Exception as e:
    fail("compute_citation_support()", e)

try:
    unsupported = detect_unsupported_claims(
        "Transformers use self-attention.", SAMPLE_EVIDENCE_PACK
    )
    assert isinstance(unsupported, list),              "Must return list"
    ok(f"detect_unsupported_claims() → {len(unsupported)} unsupported claim(s)")
except Exception as e:
    fail("detect_unsupported_claims()", e)


# ══════════════════════════════════════════════════════════════════════════
# 5. combined_arch_adapter
# ══════════════════════════════════════════════════════════════════════════
section("combined_arch_adapter.py")

try:
    from context_marl_ac.adapters.combined_arch_adapter import run_combined_pipeline
    ok("Module imports cleanly")
except Exception as e:
    fail("Module import", e)
    sys.exit(1)

try:
    result = run_combined_pipeline(QUERY)
    assert isinstance(result, dict),               "Must return dict"
    assert "answer"      in result,                "Missing 'answer'"
    assert "citations"   in result,                "Missing 'citations'"
    assert "latency_sec" in result,                "Missing 'latency_sec'"
    assert "status"      in result,                "Missing 'status'"
    assert result["status"] == "ok",               "Status must be 'ok' in dry-run"
    ok(f"run_combined_pipeline() → status={result['status']}, answer='{result['answer'][:60]}'")
except Exception as e:
    fail("run_combined_pipeline()", e)


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'─'*50}")
print(f"{BOLD}Results: {GREEN}{passed} passed{RESET}{BOLD}, {RED}{failed} failed{RESET}{BOLD} / {total} total{RESET}")
if failed == 0:
    print(f"{GREEN}All adapters OK in dry-run mode.{RESET}")
else:
    print(f"{RED}Some checks failed — see above.{RESET}")
    sys.exit(1)
