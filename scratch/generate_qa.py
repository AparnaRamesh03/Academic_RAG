import os
import sys
import json
import asyncio
from pathlib import Path

# Add brain dir to sys.path to import llm config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "brain"))

from llm_config import build_groq_llm
from langchain_core.prompts import PromptTemplate

# Categories needed for multi-agent training
CATEGORIES = [
    "direct_fact",
    "definition_explanation",
    "intra_paper_comparison",
    "cross_paper_comparison",
    "figure_grounded",
    "adversarial_superlative",
    "paraphrase_hard"
]

PROMPT = """You are an expert AI creating a challenging question-answering dataset based on the following text extracted from an academic PDF.
You must generate 10 distinct questions that can be answered using this text. 
Assign a category to each question from this list: {categories}. 

Try to vary the categories. The questions should be academic and specific.

Text Context:
{text_chunk}

Respond ONLY with a valid JSON array of objects, where each object has exactly two keys: "question" and "category". Do not include markdown code blocks around the JSON.
Example:
[
  {{"question": "What is the degradation problem?", "category": "definition_explanation"}},
  {{"question": "How does TabNet use feature selection?", "category": "direct_fact"}}
]
"""

async def generate_questions_for_pdf(pdf_md_path, num_questions=10):
    if not pdf_md_path.exists():
        print(f"File not found: {pdf_md_path}")
        return []

    with open(pdf_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # We take the first 15,000 characters to stay within context limits,
    # or you can randomly sample chunks. For simplicity, we use the first chunk.
    text_chunk = content[:15000] 

    llm = build_groq_llm(temperature=0.7)
    prompt = PromptTemplate.from_template(PROMPT)
    chain = prompt | llm

    print(f"Generating questions for {pdf_md_path.name}...")
    try:
        response = chain.invoke({"categories": ", ".join(CATEGORIES), "text_chunk": text_chunk})
        # Try to parse the JSON output
        result_text = response.content.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
            
        questions = json.loads(result_text.strip())
        return questions
    except Exception as e:
        print(f"Failed to generate questions for {pdf_md_path.name}: {e}")
        # Print raw response if it's a parsing error
        try:
             print("Raw response:", response.content)
        except:
             pass
        return []

async def main():
    target_dir = PROJECT_ROOT / "ingest_folder"
    md_files = [
        "A Multi-Agent Deep Learning and Reinforcement.md",
        "Future of Artificial IntelligencedMachine Learning Trends in Pathology.md",
        "Systematic review and meta-analysis on prevalence and risk factors for.md"
    ]
    
    all_questions = []
    
    for md_filename in md_files:
        md_path = target_dir / md_filename
        questions = await generate_questions_for_pdf(md_path)
        all_questions.extend(questions)
        
    output_path = PROJECT_ROOT / "evaluation" / "new_pdfs_training.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2)
        
    print(f"Successfully generated {len(all_questions)} questions.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
