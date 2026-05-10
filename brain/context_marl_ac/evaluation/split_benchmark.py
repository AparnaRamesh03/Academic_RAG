"""
brain/context_marl_ac/evaluation/split_benchmark.py
---------------------------------------------------
Splits standard_benchmark_v3.json into train/val/test JSONL files
with stratification by category and difficulty.
"""

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

# ── Path setup ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PATH = REPO_ROOT / "evaluation" / "standard_benchmark_v3.json"
OUTPUT_DIR = REPO_ROOT / "brain" / "context_marl_ac" / "results" / "benchmark_splits"

def split_benchmark(seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.15):
    if not BENCHMARK_PATH.exists():
        print(f"Error: Benchmark not found at {BENCHMARK_PATH}")
        return

    with open(BENCHMARK_PATH, "r") as f:
        data = json.load(f)

    # 1. Group by (category, difficulty) for stratification
    groups = defaultdict(list)
    for item in data:
        key = (item.get("category", "factual"), item.get("difficulty", "medium"))
        groups[key].append(item)

    train_set, val_set, test_set = [], [], []
    
    random.seed(seed)
    
    # 2. Distribute each group across splits
    for key, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    # 3. Save as JSONL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for name, subset in [("train", train_set), ("val", val_set), ("test", test_set)]:
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in subset:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(subset)} items to {path}")

if __name__ == "__main__":
    split_benchmark()
