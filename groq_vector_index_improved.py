"""
RAG System with VectorIndex + Sentence Transformers
REAL embeddings but local (no rate limits, no cost)
"""

import os
import re
import json
import math
import numpy as np
from typing import Optional, Any, List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# ============================================
# 1. INSTALL DEPENDENCIES (run first)
# ============================================
"""
pip install sentence-transformers
"""

# ============================================
# 2. GROQ CONFIGURATION
# ============================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"

# ============================================
# 3. EMBEDDINGS WITH SENTENCE TRANSFORMERS
# ============================================

class EmbeddingGenerator:
    """
    Embedding generator using local models
    No rate limits, no cost, high quality
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Recommended models:
        - "all-MiniLM-L6-v2": Fast, 384 dims, good quality
        - "all-mpnet-base-v2": Best quality, 768 dims, slower
        - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual
        """
        print(f"🔄 Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded. Dimension: {self.dimension}")

    def encode(self, texts, batch_size=32):
        """
        Generates embeddings for one or multiple texts
        """
        if isinstance(texts, str):
            return self.model.encode(texts).tolist()
        else:
            return self.model.encode(texts, batch_size=batch_size).tolist()

# ============================================
# 4. VECTORINDEX (IMPROVED)
# ============================================

class VectorIndex:
    """
    In-memory vector index implementation
    Improved version with automatic normalization
    """

    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_generator=None,
        normalize_vectors: bool = True
    ):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None

        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self._distance_metric = distance_metric
        self._embedding_generator = embedding_generator
        self._normalize_vectors = normalize_vectors

    def _normalize(self, vector):
        """Normalizes a vector (unit vector)"""
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            return [x/norm for x in vector]
        return vector

    def add_document(self, document: Dict[str, Any]):
        """
        Adds a document to the index (automatically generates embedding)
        """
        if not self._embedding_generator:
            raise ValueError("Embedding generator not provided")

        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        # Generate embedding
        vector = self._embedding_generator.encode(content)

        # Normalize if needed
        if self._normalize_vectors and self._distance_metric == "cosine":
            vector = self._normalize(vector)

        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Adds multiple documents to the index
        """
        print(f"\n📦 Adding {len(documents)} documents to index...")

        # Extract only contents
        contents = [doc["content"] for doc in documents]

        # Generate embeddings in batch (much faster)
        print(f"   Generating embeddings in batch...")
        vectors = self._embedding_generator.encode(contents)

        # Add each document with its vector
        for i, (vector, doc) in enumerate(zip(vectors, documents)):
            # Normalize if needed
            if self._normalize_vectors and self._distance_metric == "cosine":
                vector = self._normalize(vector)

            self.add_vector(vector=vector, document=doc)
            print(f"   📍 Document {i+1}/{len(documents)} added", end="\r")

        print(f"\n✅ Index created with {len(self)} documents")

    def search(
        self, query: Any, k: int = 1
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches for the k most similar documents to the query
        """
        if not self.vectors:
            return []

        # Get query vector
        if isinstance(query, str):
            if not self._embedding_generator:
                raise ValueError("Embedding generator required for string query")
            query_vector = self._embedding_generator.encode(query)
            if self._normalize_vectors and self._distance_metric == "cosine":
                query_vector = self._normalize(query_vector)
        elif isinstance(query, list):
            query_vector = query
        else:
            raise TypeError("Query must be string or list of numbers")

        # Check dimensions
        if self._vector_dim is None:
            return []
        if len(query_vector) != self._vector_dim:
            raise ValueError(f"Dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        # Calculate similarities
        similarities = []
        for i, stored_vector in enumerate(self.vectors):
            if self._distance_metric == "cosine":
                # For normalized vectors, dot product = cosine similarity
                sim = sum(p*q for p, q in zip(query_vector, stored_vector))
                # Convert to distance (1 - similarity)
                dist = 1.0 - max(-1.0, min(1.0, sim))
            else:
                # Euclidean distance
                dist = math.sqrt(sum((p-q)**2 for p, q in zip(query_vector, stored_vector)))

            similarities.append((dist, self.documents[i]))

        # Sort by smallest distance
        similarities.sort(key=lambda item: item[0])

        return [(doc, dist) for dist, doc in similarities[:k]]

    def add_vector(self, vector, document: Dict[str, Any]):
        """
        Adds a vector directly to the index
        """
        if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document must contain 'content' key.")

        # Check dimension consistency
        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(f"Inconsistent dimension. Expected {self._vector_dim}, got {len(vector)}")

        self.vectors.append(vector)
        self.documents.append(document)

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}')"

# ============================================
# 5. CHUNKING FUNCTIONS
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
# 6. RAG FUNCTION WITH IMPROVED VECTORINDEX
# ============================================

def answer_with_rag(query, vector_index, k=2):
    """
    Answers a question using the VectorIndex
    """
    print(f"\n📌 Question: {query}")
    print("=" * 60)

    # Search relevant chunks
    results = vector_index.search(query, k=k)

    if not results:
        return "No relevant results found."

    # Display search results
    print(f"\n📊 Top {k} results (distance):")
    for i, (doc, dist) in enumerate(results, 1):
        preview = doc["content"][:150].replace('\n', ' ') + "..."
        similarity = 1 - dist  # Convert distance to similarity
        print(f"\n   {i}. Similarity: {similarity:.4f} (distance: {dist:.4f})")
        print(f"      {preview}")

    # Build context
    context = "\n\n---\n\n".join([doc["content"] for doc, _ in results])

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

    return response.choices[0].message.content

# ============================================
# 7. MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("🚀 RAG SYSTEM WITH REAL EMBEDDINGS (LOCAL)")
    print("=" * 60)

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

    # 3. Initialize embedding generator
    print("\n🤖 Initializing embedding generator...")
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    # 4. Create VectorIndex
    index_file = "vector_index_real.json"

    if os.path.exists(index_file):
        print(f"\n📂 Loading existing index...")
        # Note: Loading from file would require additional serialization
        # For simplicity, regenerate
        print("   Regenerating index with new embeddings...")
        index = VectorIndex(
            distance_metric="cosine",
            embedding_generator=embedder,
            normalize_vectors=True
        )
        documents = [{"content": chunk} for chunk in chunks]
        index.add_documents(documents)
    else:
        print(f"\n🆕 Creating new VectorIndex with REAL embeddings...")
        index = VectorIndex(
            distance_metric="cosine",
            embedding_generator=embedder,
            normalize_vectors=True
        )

        # Prepare documents
        documents = [{"content": chunk} for chunk in chunks]

        # Add documents to index
        index.add_documents(documents)

    print(f"\n📊 Index statistics:")
    print(f"   • Documents: {len(index)}")
    print(f"   • Dimension: {index._vector_dim}")
    print(f"   • Metric: {index._distance_metric}")

    # 5. Test searches
    print("\n" + "=" * 60)
    print("🔍 SEMANTIC SEARCH TESTS")
    print("=" * 60)

    questions = [
        "What is XDR-471 syndrome?",
        "Software errors in Project Phoenix",
        "Zircon-5 Model specifications",
        "Cybersecurity incident Q4 2023",
        "Synergy Dynamics case"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"📝 QUESTION {i}")
        print(f"{'='*60}")

        answer = answer_with_rag(question, index, k=2)
        print(f"\n✅ Answer:\n{answer}\n")

        if i < len(questions):
            input("Press Enter to continue...")

    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETED")
    print("\n📌 FEATURES:")
    print("   ✅ REAL embeddings with sentence-transformers")
    print("   ✅ No rate limits or costs")
    print("   ✅ Model: all-MiniLM-L6-v2 (384 dimensions)")
    print("   ✅ Automatic normalization for cosine similarity")