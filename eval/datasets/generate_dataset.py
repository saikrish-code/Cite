import asyncio
import json
import os
import sys

# Ensure we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))

from app.services.llm import get_llm_client
from app.services.parsers import get_parser
from app.core.config import settings

PROMPT_TEMPLATE = """You are an expert academic evaluator. Given the following page from a research paper, generate {num_q} diverse question-and-answer pairs.

Format your response exactly as a valid JSON array of objects, like this:
[
  {{
    "question": "The question text",
    "ground_truth_answer": "The detailed factual answer",
    "type": "factual | synthesis | numeric | unanswerable",
    "ground_truth_passage": "The exact verbatim quote from the text that answers this question (or empty string if unanswerable)"
  }}
]

For 'unanswerable' questions, invent a plausible-sounding question that is NOT answered in the text, and set ground_truth_answer to "I don't know based on the provided context."

Text from Page {page_num}:
{text}
"""

async def generate_dataset():
    # Force settings override just in case
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_MODEL"] = "openai/gpt-oss-120b"
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
    
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frontend/node_modules/pdf-parse/test/data/04-valid.pdf'))
    parser = get_parser(pdf_path)
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    document = parser.parse(file_bytes, os.path.basename(pdf_path))
    print(f"Parsed document: {document.filename} with {document.total_pages} pages.")
    
    llm = get_llm_client()
    
    dataset = []
    
    for page in document.pages:
        page_num = page.page_number
        text = page.text
        print(f"Generating questions for page {page_num}...")
        
        # 8 questions per page to get ~40 total
        prompt = PROMPT_TEMPLATE.format(num_q=8, page_num=page_num, text=text)
        
        try:
            response = await llm.agenerate(
                prompt=prompt, 
                system_prompt="You are a helpful JSON generator. Output ONLY valid JSON array.",
                max_tokens=4000
            )
            
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
                
            qna_list = json.loads(response)
            
            for qna in qna_list:
                dataset.append({
                    "id": f"q_{len(dataset)+1}",
                    "question": qna["question"],
                    "ground_truth_answer": qna["ground_truth_answer"],
                    "source_document": "04-valid.pdf",
                    "ground_truth_pages": [page_num],
                    "ground_truth_passage": qna.get("ground_truth_passage", ""),
                    "type": qna.get("type", "factual")
                })
        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            try:
                print("Raw response:", response)
            except Exception:
                pass
            
    out_path = os.path.join(os.path.dirname(__file__), "citerag_eval_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(dataset)} questions. Saved to {out_path}")

if __name__ == "__main__":
    asyncio.run(generate_dataset())
