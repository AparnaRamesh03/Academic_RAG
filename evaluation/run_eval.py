import json
import time
import requests
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
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
import warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Load the Gold Standard dataset
    print("Loading test cases...")
    with open("gold_standard.json", "r") as f:
        gold_standard = json.load(f)

    # 2. Collect Answers from the LangGraph Brain
    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    latencies = []

    print(f"Querying the local LangGraph Brain for {len(gold_standard)} questions...")
    for item in gold_standard:
        question = item["question"]
        ground_truth = item["ground_truth"]

        try:
            start_time = time.time()
            response = requests.post("http://localhost:8000/ask", json={"query": question}, timeout=400)
            latency = round(time.time() - start_time, 2)

            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")
                contexts = result.get("context_used", [])

                data_samples["question"].append(question)
                data_samples["answer"].append(answer)
                data_samples["contexts"].append(contexts)
                data_samples["ground_truth"].append(ground_truth)
                latencies.append(latency)
                print(f"  -> OK ({latency}s): {question[:50]}...")
            else:
                print(f"  -> FAILED (HTTP {response.status_code}): {question[:50]}...")
        except Exception as e:
            print(f"  -> TIMEOUT/ERROR: {question[:50]}... ({e})")

    if not data_samples["question"]:
        print("No successful queries to the Brain API. Exiting.")
        return

    # 3. Create HuggingFace Dataset
    print("\nFormatting output as a HuggingFace Dataset...")
    dataset = Dataset.from_dict(data_samples)

    # 4. Set up Local Ragas Evaluator (Ollama)
    print("Setting up Qwen2.5 as the RAGAS Evaluator...")
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:14b", temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

    # 5. Run Full 5-Metric Evaluation
    print("\nRunning full 5-Metric RAGAS evaluation (this will take a while)...")
    results = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        AnswerCorrectness(),
        ContextPrecision(),
        ContextRecall()
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,run_config=RunConfig(max_workers=1, timeout=300)
)

    # 6. Output Results
    print("\n====================")
    print("EVALUATION RESULTS")
    print("====================")
    print(results)

    # 7. Export full CSV with latency column
    df = results.to_pandas()
    if len(latencies) == len(df):
        df["e2e_latency_seconds"] = latencies

    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved results (including latency) to 'evaluation_results.csv'!")


if __name__ == "__main__":
    main()
