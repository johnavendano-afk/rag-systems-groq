# 🚀 12 RAG Systems with Groq API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

This repository contains **12 complete RAG (Retrieval-Augmented Generation) systems** built with Python and Groq's LLaMA 3.3 70B model. Each program demonstrates a different retrieval strategy, from basic keyword search to advanced hybrid fusion, reranking, and contextual retrieval.

Whether you're prototyping, building a production system, or just learning RAG, you'll find a working example here.

---

## 📖 Table of Contents

- [Why This Repository?](#why-this-repository)
- [Programs Overview](#programs-overview)
- [Comparison Table](#comparison-table)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🤔 Why This Repository?

Most RAG tutorials stop at "chunk + embed + query". I went further and built **12 different retrieval strategies** to understand what really works. This repository is the result of that exploration.

You'll find:
- ✅ **BM25-only** (keyword search from scratch)
- ✅ **Embeddings-only** (semantic search with local models)
- ✅ **Hybrid search** (BM25 + embeddings with weighted fusion)
- ✅ **RRF (Reciprocal Rank Fusion)** — parameter‑free combination
- ✅ **LLM‑based reranking** (Groq as a reranker)
- ✅ **Contextual Retrieval** (Anthropic's technique)
- ✅ **Simulated embeddings** (for testing without APIs)
- ✅ **VoyageAI integration** (production‑grade embeddings)
- ✅ **Vector indexes** (custom in‑memory implementations)
- ✅ **Streamlit web app** (interactive Q&A)

All code is **well‑documented**, **type‑hinted**, and ready to use.

---

## 📦 Programs Overview

| # | File | Description |
|---|------|-------------|
| 1 | `groq_rag_chunking.py` | Basic chunking strategies + keyword search |
| 2 | `groq_rag_chunking_improved.py` | Keyword extraction + phrase‑level bonuses |
| 3 | `groq_embeddings_simulated.py` | Simulated embeddings (hash‑based) for testing |
| 4 | `groq_embeddings_voyage.py` | Production embeddings with VoyageAI |
| 5 | `groq_vector_index.py` | Custom vector index with simulated embeddings |
| 6 | `groq_vector_index_improved.py` | Real embeddings with sentence‑transformers |
| 7 | `groq_bm25_rag.py` | BM25 probabilistic retrieval (no ML) |
| 8 | `groq_hybrid_rag.py` | Hybrid search (BM25 + embeddings) with weights |
| 9 | `groq_retriever_rrf.py` | Multi‑index fusion with Reciprocal Rank Fusion |
| 10 | `groq_reranker.py` | RRF + LLM reranking for high precision |
| 11 | `groq_contextual_retrieval.py` | Anthropic‑style contextual chunks |
| 12 | `rag_app.py` | Streamlit web application |

---

## 📊 Comparison Table

| Program | Retrieval | Dependencies | Quality | Speed | Best For |
|---------|-----------|--------------|---------|-------|----------|
| `groq_rag_chunking` | Keyword | None | Low | High | Prototyping |
| `groq_rag_chunking_improved` | Keyword+ | None | Low | High | Prototyping |
| `groq_embeddings_simulated` | Simulated | None | Low | High | Dev/Test |
| `groq_embeddings_voyage` | VoyageAI | VoyageAI API | High | Medium | Production |
| `groq_vector_index` | Simulated | None | Low | High | Dev/Test |
| `groq_vector_index_improved` | Semantic | sentence‑transformers | High | High | Production |
| `groq_bm25_rag` | BM25 | None | Medium | High | Keyword search |
| `groq_hybrid_rag` | Hybrid | sentence‑transformers | High | High | General purpose |
| `groq_retriever_rrf` | RRF | sentence‑transformers | High | High | Multi‑index |
| `groq_reranker` | RRF+LLM | sentence‑transformers | Very High | Medium | High precision |
| `groq_contextual_retrieval` | Contextual+RRF+LLM | sentence‑transformers | Highest | Low | Max accuracy |
| `rag_app` | Semantic | sentence‑transformers | High | High | Web UI |

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/johnavendano-afk/rag-systems-groq
cd rag-systems-groq