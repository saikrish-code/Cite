# Evaluation Datasets

Store annotated academic paper datasets and benchmark evaluation sets here (e.g. JSON/JSONL format).

Each evaluation item schema:
```json
{
  "id": "eval_001",
  "document_id": "doc_attention_is_all_you_need",
  "question": "What is the computational complexity per layer of multi-head attention compared to recurrent layers?",
  "ground_truth_answer": "...",
  "ground_truth_citations": [
    {
      "page": 6,
      "text_snippet": "..."
    }
  ]
}
```
