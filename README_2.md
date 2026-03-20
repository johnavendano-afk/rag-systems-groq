# 🤖 RAG Systems with Groq API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A collection of **12 complete RAG (Retrieval-Augmented Generation) systems** built with Python, Groq's LLaMA 3.3 70B model, and `sentence-transformers`. Each script showcases a distinct retrieval strategy — from basic keyword search to advanced hybrid fusion, reranking, and contextual retrieval — so you can pick the approach that best fits your use case.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Scripts Reference](#scripts-reference)
- [Comparison Table](#comparison-table)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Most RAG tutorials cover only the basics: chunk, embed, query. This repository goes further by implementing **12 different retrieval strategies** to explore what actually works in practice.

Covered techniques:

- **BM25** — classic probabilistic keyword retrieval, no ML required
- **Semantic search** — dense vector embeddings with local `sentence-transformers` models
- **Hybrid search** — weighted combination of BM25 + embeddings
- **RRF (Reciprocal Rank Fusion)** — parameter-free multi-index combination
- **LLM-based reranking** — Groq as a reranker for high-precision retrieval
- **Contextual Retrieval** — Anthropic's technique for enriched chunk context
- **Simulated embeddings** — hash-based vectors for development/testing without an API
- **VoyageAI embeddings** — production-grade external embedding service
- **Streamlit web app** — interactive browser-based Q&A interface

All scripts are documented, type-hinted, and self-contained.

---

## Architecture

Each system follows the same high-level pipeline:

```
Document → Chunking → Indexing (BM25 / Vectors / Both)
                                      ↓
              Query → Retrieval → (Optional) Reranking
                                      ↓
                            Context Assembly → Groq LLM → Answer
```

The key differentiator between scripts is the **Retrieval** stage, which ranges from simple keyword matching to multi-index fusion with LLM reranking.

---

## Scripts Reference

| # | File | Strategy | Description |
|---|------|----------|-------------|
| 1 | `groq_rag_chunking.py` | Keyword | Basic chunking + keyword overlap scoring |
| 2 | `groq_rag_chunking_improved.py` | Keyword+ | Keyword extraction with phrase-level bonuses |
| 3 | `groq_embeddings_simulated.py` | Simulated | Hash-based pseudo-embeddings for testing |
| 4 | `groq_embeddings_voyage.py` | VoyageAI | Production embeddings via VoyageAI API |
| 5 | `groq_vector_index.py` | Vector (simulated) | Custom in-memory vector index with simulated embeddings |
| 6 | `groq_vector_index_improved.py` | Vector (real) | In-memory vector index with `sentence-transformers` |
| 7 | `groq_bm25_rag.py` | BM25 | Full BM25 probabilistic retrieval from scratch |
| 8 | `groq_hybrid_rag.py` | Hybrid | Weighted BM25 + embeddings fusion |
| 9 | `groq_retriever_rrf.py` | RRF | Multi-index fusion with Reciprocal Rank Fusion |
| 10 | `groq_reranker.py` | RRF + LLM rerank | RRF results re-scored by the Groq LLM |
| 11 | `groq_contextual_retrieval.py` | Contextual + RRF + LLM | Enriched chunk context before indexing, then RRF + rerank |
| 12 | `rag_app.py` | Semantic (web UI) | Streamlit app with semantic search |

> **Note:** `groq_rag_chunking_mejorado.py`, `groq_embeddings_simulado.py`, and `groq_vector_index_mejorado.py` are Spanish-language equivalents of scripts 2, 3, and 6 respectively.

---

## Comparison Table

| Script | Retrieval method | External deps | Quality | Speed | Best for |
|--------|-----------------|---------------|---------|-------|----------|
| `groq_rag_chunking` | Keyword | None | Low | ⚡ High | Quick prototyping |
| `groq_rag_chunking_improved` | Keyword+ | None | Low | ⚡ High | Quick prototyping |
| `groq_embeddings_simulated` | Simulated | None | Low | ⚡ High | Dev / CI testing |
| `groq_embeddings_voyage` | VoyageAI | VoyageAI API | High | Medium | Production |
| `groq_vector_index` | Vector (sim.) | None | Low | ⚡ High | Dev / CI testing |
| `groq_vector_index_improved` | Semantic | sentence-transformers | High | ⚡ High | General production |
| `groq_bm25_rag` | BM25 | None | Medium | ⚡ High | Keyword-heavy domains |
| `groq_hybrid_rag` | Hybrid | sentence-transformers | High | ⚡ High | General purpose |
| `groq_retriever_rrf` | RRF | sentence-transformers | High | ⚡ High | Multi-index setups |
| `groq_reranker` | RRF + LLM | sentence-transformers | Very High | Medium | High-precision retrieval |
| `groq_contextual_retrieval` | Contextual + RRF + LLM | sentence-transformers | Highest | Low | Maximum accuracy |
| `rag_app` | Semantic | sentence-transformers | High | ⚡ High | Interactive demos |

---

## Requirements

- Python 3.10 or higher
- A [Groq API key](https://console.groq.com/) (free tier available)
- *(Optional)* A [VoyageAI API key](https://www.voyageai.com/) — only needed for `groq_embeddings_voyage.py`

Python packages:

```
groq
openai>=1.0.0
python-dotenv>=1.0.0
sentence-transformers>=2.2.0
numpy>=1.24.0
streamlit>=1.28.0
requests
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/johnavendano-afk/rag-systems-groq.git
cd rag-systems-groq

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

Alternatively, use the provided setup script:

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

---

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
VOYAGE_API_KEY=your_voyageai_key_here   # Optional — only for groq_embeddings_voyage.py
```

All scripts load credentials automatically via `python-dotenv`.

---

## Usage

Every script uses `report.md` as the source document. Make sure it is present in the project root before running any script.

### Run a specific RAG system

```bash
# BM25 retrieval
python groq_bm25_rag.py

# Hybrid BM25 + embeddings
python groq_hybrid_rag.py

# RRF multi-index fusion
python groq_retriever_rrf.py

# RRF + LLM reranking
python groq_reranker.py

# Contextual retrieval (highest accuracy, slowest)
python groq_contextual_retrieval.py
```

### Launch the Streamlit web app

```bash
streamlit run rag_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Recommended starting points

| Goal | Start here |
|------|-----------|
| Learning RAG basics | `groq_rag_chunking.py` → `groq_bm25_rag.py` |
| No API keys, just testing | `groq_embeddings_simulated.py` |
| Best quality / production | `groq_contextual_retrieval.py` or `groq_reranker.py` |
| Visual / interactive demo | `rag_app.py` |

---

## Project Structure

```
rag-systems-groq/
├── groq_rag_chunking.py              # Script 1 — keyword chunking
├── groq_rag_chunking_improved.py     # Script 2 — improved keyword chunking
├── groq_rag_chunking_mejorado.py     # Script 2 — Spanish version
├── groq_embeddings_simulated.py      # Script 3 — simulated embeddings (EN)
├── groq_embeddings_simulado.py       # Script 3 — simulated embeddings (ES)
├── groq_embeddings_voyage.py         # Script 4 — VoyageAI embeddings
├── groq_vector_index.py              # Script 5 — vector index (simulated)
├── groq_vector_index_improved.py     # Script 6 — vector index (real, EN)
├── groq_vector_index_mejorado.py     # Script 6 — vector index (real, ES)
├── groq_bm25_rag.py                  # Script 7 — BM25
├── groq_hybrid_rag.py                # Script 8 — hybrid BM25 + embeddings
├── groq_retriever_rrf.py             # Script 9 — RRF fusion
├── groq_reranker.py                  # Script 10 — RRF + LLM reranker
├── groq_contextual_retrieval.py      # Script 11 — contextual retrieval
├── rag_app.py                        # Script 12 — Streamlit web app
├── report.md                         # Source document used by all scripts
├── requirements.txt                  # Python dependencies
├── setup_environment.sh              # Environment setup script
├── bm25_index.json                   # Persisted BM25 index
├── vector_index.json                 # Persisted vector index
├── embeddings.json                   # Persisted embeddings
├── embeddings_simulated.json         # Persisted simulated embeddings
├── embeddings_simulados.json         # Persisted simulated embeddings (ES)
└── RAG_Systems_Documentation.pdf     # Full technical documentation
```

---

## Documentation

A detailed PDF with the architecture, design decisions, and benchmarks for all 12 systems is included in the repository:

📄 [`RAG_Systems_Documentation.pdf`](./RAG_Systems_Documentation.pdf)

---

## Contributing

Contributions are welcome. To propose changes:

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please keep code style consistent with the existing scripts (type hints, docstrings, section comments).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

> Built with [Groq](https://groq.com) · [sentence-transformers](https://www.sbert.net/) · [Streamlit](https://streamlit.io)
