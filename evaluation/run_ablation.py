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
    print("Loading test cases...")
    with open("gold_standard.json", "r") as f:
        gold_standard = json.load(f)

    modes = ["naive", "no_crag", "no_auditor", "full"]
    all_results = []

    print("Setting up Qwen2.5 as the RAGAS Evaluator...")
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:14b", temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    run_config = RunConfig(max_workers=1, timeout=300)

    for mode in modes:
        print(f"\n==========================================")
        print(f"RUNNING ABLATION MODE: {mode.upper()}")
        print(f"==========================================")
        
        data_samples = {
            "question": [], "answer": [], "contexts": [], "ground_truth": []
        }
        latencies = []

        for item in gold_standard:
            question = item["question"]
            ground_truth = item["ground_truth"]

            try:
                start_time = time.time()
                response = requests.post("http://localhost:8000/ask", json={"query": question, "ablation_mode": mode}, timeout=400)
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
                    print(f"  -> OK ({latency}s): {question[:40]}...")
                else:
                    print(f"  -> FAILED (HTTP {response.status_code}): {question[:40]}...")
            except Exception as e:
                print(f"  -> TIMEOUT/ERROR: {question[:40]}... ({e})")

        if not data_samples["question"]:
            print(f"Skipping evaluation for {mode} due to missing data.")
            continue

        dataset = Dataset.from_dict(data_samples)
        
        print(f"\nEvaluating mode '{mode}' with RAGAS 5-Metrics...")
        results = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), AnswerCorrectness(), ContextPrecision(), ContextRecall()],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config
        )
        print(f"DEBUG: Results for {mode}")
        print(results.to_pandas().head())
        # Convert results to a pandas dataframe to handle scores safely
        res_df = results.to_pandas()
        mode_scores = {
            "ablation_mode": mode,
            "avg_latency": round(sum(latencies)/len(latencies), 2) if latencies else 0,
            # Use .get() or check column existence to prevent KeyErrors
            "faithfulness": res_df["faithfulness"].mean() if "faithfulness" in res_df else 0,
            "answer_relevancy": res_df["answer_relevancy"].mean() if "answer_relevancy" in res_df else 0,
            "answer_correctness": res_df["answer_correctness"].mean() if "answer_correctness" in res_df else 0,
            "context_precision": res_df["context_precision"].mean() if "context_precision" in res_df else 0,
            "context_recall": res_df["context_recall"].mean() if "context_recall" in res_df else 0,
        }
        all_results.append(mode_scores)

    # Save summary dataset
    df = pd.DataFrame(all_results)
    df.to_csv("ablation_summary.csv", index=False)
    
    print("\n==========================================")
    print("ALL ABLATION MODES COMPLETE!")
    print("==========================================")
    print(df.to_string(index=False))
    print("\nSaved summary to 'evaluation/ablation_summary.csv'.")

if __name__ == "__main__":
    main()
