import json
import sys

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = '/Users/spartan/Documents/Academic_RAG/brain/context_marl_ac/results/final_eval/results/multihop_rag_mini_results.jsonl'


results = []
try:
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Each line might be a list or a single dict depending on how it was saved
                if isinstance(data, list):
                    for item in data:
                        results.append({
                            'query': item.get('current_query', 'N/A'),
                            'answer': item.get('generated_answer_preview', 'N/A'),
                            'ground_truth': item.get('ground_truth', 'N/A'),
                            'category': item.get('category', 'N/A'),
                            'status': item.get('final_status', 'N/A'),
                            'verifier_decision': item.get('verifier_decision', 'N/A'),
                            'verifier_reason': item.get('verifier_reason', 'N/A')
                        })
                else:
                    results.append({
                        'query': data.get('current_query', 'N/A'),
                        'answer': data.get('generated_answer_preview', 'N/A'),
                        'ground_truth': data.get('ground_truth', 'N/A'),
                        'category': data.get('category', 'N/A'),
                        'status': data.get('final_status', 'N/A'),
                        'verifier_decision': data.get('verifier_decision', 'N/A'),
                        'verifier_reason': data.get('verifier_reason', 'N/A')
                    })
            except Exception as e:
                print(f"Error parsing line: {e}")
except FileNotFoundError:
    print(f"File not found: {file_path}")

for i, res in enumerate(results):
    print(f"Q{i+1} [{res['category']}]: {res['query']}")
    print(f"  Generated: {res['answer']}")
    print(f"  Gold:      {res['ground_truth']}")
    print(f"  Status:    {res['status']} | Verifier: {res['verifier_decision']}")
    print(f"  Reason:    {res['verifier_reason']}")
    print("-" * 60)
