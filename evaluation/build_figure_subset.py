import json
from pathlib import Path


INPUT_FILES = [
    Path("./gold_standard_benchmark_v2_1.json"),
    Path("./gold_standard_benchmark_v2_2.json"),
]

OUTPUT_PATH = Path("./benchmarks/figure_table_subset.json")
TARGET_CATEGORY = "figure_table_diagram_grounded"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    all_rows = []
    for path in INPUT_FILES:
        all_rows.extend(load_json(path))

    subset = [row for row in all_rows if row.get("category") == TARGET_CATEGORY]

    save_json(subset, OUTPUT_PATH)

    print(f"[✓] Saved {len(subset)} figure-grounded questions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()