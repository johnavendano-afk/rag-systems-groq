"""
HYBRID RAG System with BM25 + Semantic Embeddings + Groq
Combines term-based and semantic search for better results
"""

import os
import re
import math
import json
import numpy as np
from collections import Counter
from typing import Callable, Optional, Any, List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# ============================================
# 1. GROQ CONFIGURATION
# ============================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"

# ============================================
# 2. CHUNKING FUNCTIONS
# ============================================

def chunk_by_section(document_text):
    """Splits the document into chunks based on sections (##)"""
    pattern = r"\n## "
    sections = re.split(pattern, document_text)

    result = []
    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section
        result.append(section.strip())

    return result

# ============================================
# 3. EMBEDDING MODEL (SENTENCE TRANSFORMERS)
# ============================================

class EmbeddingGenerator:
    """Embedding generator using local models"""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"🔄 Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Dimension: {self.dimension}")

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            return self.model.encode(texts).tolist()
        return self.model.encode(texts, batch_size=batch_size).tolist()

# ============================================
# 4. BM25 INDEX
# ============================================

class BM25Index:
    """BM25 index for term-based search"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer=None):
        self.documents: List[Dict[str, Any]] = []
        self._corpus_tokens: List[List[str]] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Dict[str, int] = {}
        self._avg_doc_len: float = 0.0
        self._idf: Dict[str, float] = {}
        self._index_built: bool = False
        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer if tokenizer else self._default_tokenizer

    def _default_tokenizer(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [token for token in tokens if token]

    def add_document(self, document: Dict[str, Any]):
        content = document.get("content", "")
        doc_tokens = self._tokenizer(content)
        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        self._doc_len.append(len(doc_tokens))

        # Update frequencies
        seen = set()
        for token in doc_tokens:
            if token not in seen:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen.add(token)
        self._index_built = False

    def add_documents(self, documents: List[Dict[str, Any]]):
        print(f"\n📦 Adding {len(documents)} documents to BM25 index...")
        for i, doc in enumerate(documents):
            self.add_document(doc)
            print(f"   📍 Document {i+1}/{len(documents)}", end="\r")
        self._build_index()
        print(f"\n✅ BM25 index created with {len(self)} documents")

    def _build_index(self):
        if not self.documents:
            return
        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            self._idf[term] = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
        self._index_built = True

    def _compute_score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_idx])
        doc_len = self._doc_len[doc_idx]

        for token in query_tokens:
            if token not in self._idf:
                continue
            idf = self._idf[token]
            tf = doc_term_counts.get(token, 0)
            numerator = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self._avg_doc_len))
            score += numerator / (denominator + 1e-9)
        return score

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents or not self._index_built:
            return []

        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []

        scores = []
        for i in range(len(self.documents)):
            score = self._compute_score(query_tokens, i)
            if score > 0:
                scores.append((score, self.documents[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [(doc, score) for score, doc in scores[:k]]

    def __len__(self):
        return len(self.documents)

# ============================================
# 5. VECTOR INDEX (EMBEDDINGS)
# ============================================

class VectorIndex:
    """Vector index for semantic search"""

    def __init__(self, normalize: bool = True):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._dim: Optional[int] = None
        self._normalize = normalize

    def _normalize_vector(self, v):
        norm = math.sqrt(sum(x*x for x in v))
        return [x/norm for x in v] if norm > 0 else v

    def add_document(self, document: Dict[str, Any], embedding: List[float]):
        if self._normalize:
            embedding = self._normalize_vector(embedding)
        if not self.vectors:
            self._dim = len(embedding)
        elif len(embedding) != self._dim:
            raise ValueError("Inconsistent dimension")
        self.vectors.append(embedding)
        self.documents.append(document)

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        print(f"\n📦 Adding {len(documents)} documents to vector index...")
        for doc, emb in zip(documents, embeddings):
            self.add_document(doc, emb)
        print(f"✅ Vector index created with {len(self)} documents")

    def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not self.vectors:
            return []

        if self._normalize:
            query_vector = self._normalize_vector(query_vector)

        similarities = []
        for vec in self.vectors:
            sim = sum(p*q for p, q in zip(query_vector, vec))
            similarities.append(max(-1.0, min(1.0, sim)))

        top_indices = np.argsort(similarities)[-k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]

    def __len__(self):
        return len(self.vectors)

# ============================================
# 6. HYBRID SYSTEM (BM25 + EMBEDDINGS)
# ============================================

class HybridRAG:
    """
    Hybrid RAG system that combines:
    - BM25: exact term-based search
    - Embeddings: semantic search
    """

    def __init__(self, bm25_weight: float = 0.3, embedding_weight: float = 0.7):
        """
        Weights control the relative importance of each method:
        - Higher BM25 weight: better for exact terms
        - Higher Embedding weight: better for conceptual search
        """
        self.bm25_index = BM25Index()
        self.vector_index = VectorIndex()
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.embedding_generator = None
        self.chunks = []

    def initialize_embeddings(self):
        """Initializes the embedding generator (only if needed)"""
        if self.embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator()

    def add_documents(self, chunks: List[str]):
        """Adds documents to both indexes"""
        self.chunks = chunks
        documents = [{"content": chunk} for chunk in chunks]

        # 1. Add to BM25
        self.bm25_index.add_documents(documents)

        # 2. Generate embeddings and add to vector index
        self.initialize_embeddings()
        print(f"\n🔄 Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_generator.encode(chunks)
        self.vector_index.add_documents(documents, embeddings)

        print(f"\n✅ Hybrid system initialized with {len(self)} documents")

    def hybrid_search(self, query: str, k: int = 3) -> List[Tuple[Dict[str, Any], float, float, float]]:
        """
        Hybrid search combining BM25 and embedding results.
        Returns: (document, bm25_score, embedding_score, final_score)
        """
        # 1. Search with BM25
        bm25_results = self.bm25_index.search(query, k=k*2)
        bm25_dict = {doc['content']: (doc, score) for doc, score in bm25_results}

        # 2. Search with embeddings
        query_embedding = self.embedding_generator.encode(query)
        emb_results = self.vector_index.search(query_embedding, k=k*2)
        emb_dict = {doc['content']: (doc, score) for doc, score in emb_results}

        # 3. Combine results (union of all found documents)
        all_contents = set(bm25_dict.keys()) | set(emb_dict.keys())

        combined = []
        for content in all_contents:
            doc = None
            bm25_score = 0.0
            emb_score = 0.0

            if content in bm25_dict:
                doc, bm25_score = bm25_dict[content]
                # Normalize BM25 score (simple normalization)
                bm25_score = min(1.0, bm25_score / 10.0)

            if content in emb_dict:
                doc, emb_score = emb_dict[content]
                # Embeddings are already in [0,1]

            # Weighted final score
            final_score = (self.bm25_weight * bm25_score +
                          self.embedding_weight * emb_score)

            combined.append((doc, bm25_score, emb_score, final_score))

        # Sort by final score
        combined.sort(key=lambda x: x[3], reverse=True)

        return combined[:k]

    def answer(self, query: str, k: int = 2) -> Tuple[str, List]:
        """
        Answers a question using hybrid search.
        """
        print(f"\n📌 Question: {query}")
        print("=" * 60)

        # Hybrid search
        results = self.hybrid_search(query, k=k)

        if not results:
            return "No relevant results found.", []

        # Display results
        print(f"\n📊 Top {k} results (hybrid):")
        for i, (doc, bm25_score, emb_score, final_score) in enumerate(results, 1):
            preview = doc['content'][:150].replace('\n', ' ') + "..."
            print(f"\n   {i}. Final score: {final_score:.4f}")
            print(f"      BM25: {bm25_score:.4f} | Embedding: {emb_score:.4f}")
            print(f"      {preview}")

        # Build context
        context = "\n\n---\n\n".join([doc['content'] for doc, _, _, _ in results])

        prompt = f"""You are an expert document analysis assistant.
Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Use only information from the context
- If the answer is not in the context, say you don't have that information
- Be concise but complete

ANSWER:"""

        # Generate response with Groq
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert document analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        return response.choices[0].message.content, results

    def __len__(self):
        return len(self.bm25_index)

# ============================================
# 7. MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("🚀 HYBRID RAG SYSTEM (BM25 + EMBEDDINGS) + GROQ")
    print("=" * 70)

    # 1. Load document
    try:
        with open("report.md", "r", encoding="utf-8") as f:
            text = f.read()
        print(f"\n📄 Document loaded: report.md")
        print(f"   Length: {len(text)} characters")
    except FileNotFoundError:
        print("❌ Error: report.md file not found")
        exit(1)

    # 2. Chunking
    print("\n📦 Generating chunks...")
    chunks = chunk_by_section(text)
    print(f"   {len(chunks)} chunks created")

    # 3. Create hybrid system
    print("\n🔄 Initializing hybrid system...")
    hybrid = HybridRAG(bm25_weight=0.3, embedding_weight=0.7)

    # 4. Add documents
    hybrid.add_documents(chunks)

    # 5. Test searches
    print("\n" + "=" * 70)
    print("🔍 HYBRID SEARCH TESTS")
    print("=" * 70)

    questions = [
        "What is XDR-471 syndrome?",
        "Software errors in Project Phoenix",
        "Zircon-5 Model specifications",
        "Cybersecurity incident Q4 2023",
        "Synergy Dynamics case"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"📝 QUESTION {i}")
        print(f"{'='*70}")

        answer, results = hybrid.answer(question, k=2)
        print(f"\n✅ Answer:\n{answer}\n")

        if i < len(questions):
            input("Press Enter to continue...")

    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETED")
    print("\n📌 HYBRID SYSTEM FEATURES:")
    print("   ✅ BM25: exact term-based search")
    print("   ✅ Embeddings: semantic search")
    print("   ✅ Weighted combination of results")
    print("   ✅ Best of both worlds")