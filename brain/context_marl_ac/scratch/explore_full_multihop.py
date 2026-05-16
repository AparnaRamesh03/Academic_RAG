from datasets import load_dataset

print("Checking MultiHopRAG dataset...")
ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
print(f"Total questions in 'MultiHopRAG' split: {len(ds)}")

corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
print(f"Total documents in 'corpus' split: {len(corpus)}")

categories = {}
for item in ds:
    cat = item.get("question_type", "unknown")
    categories[cat] = categories.get(cat, 0) + 1

print("\nCategories:")
for cat, count in categories.items():
    print(f"  - {cat}: {count}")
