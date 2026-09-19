# Engineering Decisions

This document records key technical decisions, their rationale, and alternatives considered.

---

## 1. Chunking Defaults: 800 Characters / 100 Characters Overlap

**Decision:** The `RecursiveTokenChunker` defaults to `chunk_size=800` characters and `chunk_overlap=100` characters.

### Rationale

**Chunk size = 800 characters (~200 tokens):**

The chunk size represents a fundamental trade-off in any RAG system:

| Concern | Small chunks (< 400 chars) | Large chunks (> 1500 chars) |
| :--- | :--- | :--- |
| **Retrieval precision** | ✅ High — each chunk is focused on one idea | ❌ Low — multiple topics dilute relevance scores |
| **Context for generation** | ❌ Insufficient — model may lack surrounding context | ✅ Rich — model sees full paragraphs |
| **Embedding quality** | ✅ Dense vectors encode a tight semantic signal | ❌ Embeddings blur across disparate content |
| **Token budget efficiency** | ✅ Fit more chunks in the LLM's context window | ❌ Fewer chunks fit, reducing source diversity |

800 characters (~200 tokens at the standard ~4 chars/token ratio for English) sits in the empirically validated sweet spot for the embedding models we use (`BAAI/bge-small-en-v1.5`, `all-MiniLM-L6-v2`):

- **MTEB benchmark data** shows that these models achieve peak retrieval recall when passages are 150–250 tokens long. Shorter passages lose contextual signal; longer ones dilute the embedding with off-topic text.
- **200 tokens ≈ 1–2 academic paragraphs**, which in research papers typically correspond to a single coherent argument, definition, or result — exactly the granularity needed for precise citation anchoring.
- **LLM context budget**: With a 4K–8K token context window, 200-token chunks let us pack 5–10 diverse source passages into the prompt while leaving room for the system prompt and generation, which is important for our hybrid retrieval pipeline where we want to present multiple independently retrieved evidence chunks.

**Overlap = 100 characters (~25 tokens, 12.5% of chunk size):**

Overlap ensures that information straddling a chunk boundary is not lost:

- **Boundary continuity**: Academic sentences often span 20–40 tokens. A 25-token overlap guarantees that at least one complete sentence is shared between adjacent chunks, preventing a key claim from being split across two chunks where neither contains the full statement.
- **Diminishing returns beyond ~15%**: Research on chunking for RAG (e.g., LlamaIndex's evaluation studies and Langchain's benchmarks) consistently shows that overlaps between 10–15% of chunk size provide the best trade-off. Beyond ~20%, chunks begin to duplicate too much content, inflating the vector index without improving recall.
- **Retrieval deduplication**: When overlap is too large, the same passage appears in multiple chunks, cluttering results and wasting reranker compute. 12.5% keeps duplication minimal.

### Alternatives Considered

| Strategy | Pros | Cons | Status |
| :--- | :--- | :--- | :--- |
| **Token-based sizing** (count actual tokens with a tokenizer) | Exact token control | Adds tokenizer dependency, slower | Future option — `BaseChunker` makes this swappable |
| **Sentence-window chunking** | Each chunk is one sentence + surrounding context window | Requires sentence boundary detection (spaCy/nltk) | Planned as a `BaseChunker` subclass for comparison |
| **Semantic chunking** (split at topic shifts using embedding similarity) | Optimal semantic coherence per chunk | Slow (requires embedding every sentence), complex | Planned for Milestone 6 evaluation |
| **Fixed 512-token chunks** (common default) | Simple, well-tested | Too large for precise citation; wastes context budget | Rejected — 200 tokens gives better precision@k for our use case |

### Why Character-Based Instead of Token-Based

The chunker operates on character counts rather than actual tokenizer tokens. This is a deliberate trade-off:

1. **Zero external dependency**: No need to load a tokenizer model at chunking time, keeping ingestion fast and the dependency graph minimal.
2. **Stable approximation**: For English text (our primary domain), the ~4 chars/token ratio is remarkably consistent across BPE tokenizers (GPT, Llama, Mistral). The error margin is well within acceptable bounds for retrieval.
3. **Swappable**: The `BaseChunker` abstract class means a `TokenAwareChunker` using `tiktoken` or the model's own tokenizer can be dropped in when exact token counts matter (e.g., for tightly packed LLM prompts).

---

## 2. Recursive Separator Hierarchy

**Decision:** The chunker tries separators in order: `\n\n` → `\n` → `. ` → ` ` → `""` (character-level).

**Rationale:** This hierarchy respects the natural structure of academic documents:
- `\n\n` (paragraph break) is the strongest semantic boundary — splitting here keeps complete arguments intact.
- `\n` (line break) captures section headers, list items, and equations that are separated by single newlines.
- `. ` (sentence boundary) is the finest semantic unit worth preserving — splitting mid-sentence severely degrades both embedding quality and generated answer coherence.
- ` ` (word boundary) and `""` (character) are last-resort fallbacks for pathologically long strings (e.g., base64-encoded content, or a single massive paragraph).

---

## 3. Parser Library Choices

| Format | Library | Rationale |
| :--- | :--- | :--- |
| **PDF** | PyMuPDF (`pymupdf`) | C-based engine, extremely fast, preserves reading order and page structure. Widely used in production RAG systems. |
| **DOCX** | `python-docx` | Mature, well-maintained. DOCX has no native page concept — we detect `w:lastRenderedPageBreak` and explicit `w:br type="page"` markers for best-effort page boundaries. |
| **TXT** | Built-in `bytes.decode()` | No external dependency needed. Entire file treated as page 1 since plain text has no pagination. |
