# CiteRAG Evaluation Harness

This directory contains benchmarking suites, datasets, and scripts to systematically evaluate CiteRAG's retrieval quality and generation fidelity.

## Key Metrics Evaluated:
1. **Faithfulness / Groundedness**: Ensuring generated answers contain zero ungrounded assertions.
2. **Answer Relevance**: Measuring how accurately answers satisfy user queries.
3. **Context Precision & Recall**: Verifying whether relevant paper chunks are ranked at top positions ($k=1, 3, 5$).
4. **Citation Entailment Accuracy**: Checking if inline citations correctly point to evidence passages.

## Structure
- `datasets/`: Ground-truth Q&A pairs, annotated research papers, and synthetic test sets.
- `benchmarks/`: Automated test runners executing RAG triad evaluations.
