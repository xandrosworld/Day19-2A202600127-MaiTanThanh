# Day19-2A202600127-MaiTanThanh

Lab Day 19 - GraphRAG With Tech Company Corpus.

This project builds a small GraphRAG pipeline from Wikipedia pages about technology companies.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put your OpenAI key in `.env`:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-nano-2025-08-07
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

3. Run:

```bash
python graphrag_tech_wiki.py
```

## Outputs

The script writes files to `outputs/`:

- `tech_company_graph.png`: screenshot/visualization of the knowledge graph.
- `benchmark_results.csv`: 20-question comparison between Flat RAG and GraphRAG.
- `benchmark_results.md`: report table for submission.
- `submission_report.md`: Vietnamese submission report.
- `triples.json`: extracted entity-relation triples.
- `corpus_wiki.json`: Wikipedia corpus cache.
- `run_summary.json`: runtime and token/cost-related metadata.

## Method

- Flat RAG retrieves relevant text chunks with OpenAI embeddings and FAISS when available, then sends them to the LLM.
- GraphRAG extracts triples, builds a NetworkX graph, retrieves 2-hop graph neighborhoods, and sends graph context to the LLM.
- The benchmark contains 20 questions, including multi-hop questions that should favor graph traversal.
