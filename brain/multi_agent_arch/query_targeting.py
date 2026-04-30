from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List


SOURCE_ALIASES: Dict[str, List[str]] = {
    "AttentionIsAllYouNeed.pdf": [
        "attention is all you need",
        "transformer",
        "vaswani",
    ],
    "BERT.pdf": [
        "bert",
        "bidirectional encoder representations from transformers",
    ],
    "ImgRecog.pdf": [
        "resnet",
        "residual network",
        "deep residual learning",
    ],
    "RAGSurvey.pdf": [
        "naive rag",
        "advanced rag",
        "modular rag",
        "rag survey",
        "survey",
    ],
    "TabNet.pdf": [
        "tabnet",
    ],
}

DISPLAY_NAMES: Dict[str, str] = {
    "AttentionIsAllYouNeed.pdf": "Transformer",
    "BERT.pdf": "BERT",
    "ImgRecog.pdf": "ResNet",
    "RAGSurvey.pdf": "RAG Survey",
    "TabNet.pdf": "TabNet",
}


def is_comparison_query(query: str) -> bool:
    q = (query or "").lower()
    markers = [
        "compare",
        "comparison",
        "versus",
        " vs ",
        "difference",
        "differences",
        "both",
        "each",
        "in contrast",
        "whereas",
    ]
    return any(m in q for m in markers)


def is_underspecified_superlative_query(query: str) -> bool:
    q = (query or "").lower()
    markers = [
        "which architecture",
        "which model",
        "best",
        "most efficient",
        "most effective",
        "solves the efficiency problem best",
    ]
    return any(m in q for m in markers)


def infer_target_sources(query: str) -> List[str]:
    q = (query or "").lower()
    matched = []

    for source_file, aliases in SOURCE_ALIASES.items():
        if any(alias in q for alias in aliases):
            matched.append(source_file)

    return matched


def infer_target_entities(query: str) -> List[str]:
    return [DISPLAY_NAMES[s] for s in infer_target_sources(query)]


def enforce_target_entities(rewritten_query: str, target_entities: List[str]) -> str:
    result = " ".join((rewritten_query or "").split())

    for entity in target_entities:
        if entity.lower() not in result.lower():
            result = f"{result} {entity}".strip()

    return " ".join(result.split())


def source_distribution(docs: List[Dict[str, Any]], top_n: int = 6) -> Dict[str, int]:
    counts = Counter()

    for doc in docs[:top_n]:
        source_file = ((doc.get("metadata") or {}).get("source_file")) or "Unknown"
        counts[source_file] += 1

    return dict(counts.most_common())


def source_distribution_text(docs: List[Dict[str, Any]], top_n: int = 6) -> str:
    dist = source_distribution(docs, top_n=top_n)
    if not dist:
        return "No source distribution available."
    return ", ".join([f"{k}: {v}" for k, v in dist.items()])


def pick_balanced_docs(
    docs: List[Dict[str, Any]],
    target_sources: List[str],
    *,
    per_source: int = 2,
    max_total: int = 6,
) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)

    for doc in docs:
        source_file = ((doc.get("metadata") or {}).get("source_file")) or "Unknown"
        grouped[source_file].append(doc)

    selected: List[Dict[str, Any]] = []

    for source in target_sources:
        selected.extend(grouped.get(source, [])[:per_source])

    if len(selected) < max_total:
        for doc in docs:
            if doc not in selected:
                selected.append(doc)
            if len(selected) >= max_total:
                break

    return selected[:max_total]