"""
RAG System with custom VectorIndex + Groq
Adapted from notebook with complete vector database implementation
No VoyageAI dependency - uses simulated embeddings
"""

import os
import re
import json
import math
import numpy as np
import hashlib
from typing import Optional, Any, List, Dict, Tuple
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
# 2. SIMULATED EMBEDDING FUNCTION
# ============================================

def generate_simulated_embedding(text, dimension=384):
    """
    Generates a SIMULATED embedding based on the text hash
    This is NOT a real embedding, but allows testing the RAG flow
    """
    # Create a hash of the text
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()

    # Generate pseudo-random but deterministic vector
    np.random.seed(int.from_bytes(hash_bytes[:4], 'little'))
    embedding = np.random.randn(dimension)

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding.tolist()

def generate_embedding_batch(texts, input_type="document"):
    """
    Generates embeddings for multiple texts (simulated)
    """
    if isinstance(texts, str):
        return generate_simulated_embedding(texts)
    else:
        return [generate_simulated_embedding(t) for t in texts]

# ============================================
# 3. VECTORINDEX IMPLEMENTATION
# ============================================

class VectorIndex:
    """
    In-memory vector index implementation
    Supports cosine or euclidean distance search
    """

    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_fn=None,
    ):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None

        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self._distance_metric = distance_metric
        self._embedding_fn = embedding_fn or generate_embedding_batch

    def add_document(self, document: Dict[str, Any]):
        """
        Adds a document to the index (automatically generates embedding)
        """
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        vector = self._embedding_fn(content)
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Adds multiple documents to the index
        """
        print(f"\n📦 Adding {len(documents)} documents to index...")
        for i, doc in enumerate(documents):
            self.add_document(doc)
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
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(
            isinstance(x, (int, float)) for x in query
        ):
            query_vector = query
        else:
            raise TypeError(
                "Query must be either a string or a list of numbers."
            )

        # Check dimensions
        if self._vector_dim is None:
            return []
        if len(query_vector) != self._vector_dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}"
            )

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        # Select distance function
        if self._distance_metric == "cosine":
            dist_func = self._cosine_distance
        else:
            dist_func = self._euclidean_distance

        # Calculate distances
        distances = []
        for i, stored_vector in enumerate(self.vectors):
            distance = dist_func(query_vector, stored_vector)
            distances.append((distance, self.documents[i]))

        # Sort by smallest distance
        distances.sort(key=lambda item: item[0])

        return [(doc, dist) for dist, doc in distances[:k]]

    def add_vector(self, vector, document: Dict[str, Any]):
        """
        Adds a vector directly to the index (without generating embedding)
        """
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) for x in vector
        ):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        # Check dimension consistency
        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(
                f"Inconsistent vector dimension. Expected {self._vector_dim}, got {len(vector)}"
            )

        self.vectors.append(list(vector))
        self.documents.append(document)

    def _euclidean_distance(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Euclidean distance"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def _dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """Dot product"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        return sum(p * q for p, q in zip(vec1, vec2))

    def _magnitude(self, vec: List[float]) -> float:
        """Vector magnitude"""
        return math.sqrt(sum(x * x for x in vec))

    def _cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Cosine distance (1 - cosine similarity)
        Smaller distance = more similar
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")

        mag1 = self._magnitude(vec1)
        mag2 = self._magnitude(vec2)

        if mag1 == 0 and mag2 == 0:
            return 0.0
        elif mag1 == 0 or mag2 == 0:
            return 1.0

        dot_prod = self._dot_product(vec1, vec2)
        cosine_similarity = dot_prod / (mag1 * mag2)
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        return 1.0 - cosine_similarity

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        has_embed_fn = "Yes" if self._embedding_fn else "No"
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}', has_embedding_fn='{has_embed_fn}')"

# ============================================
# 4. CHUNKING FUNCTIONS
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
# 5. RAG FUNCTION WITH VECTORINDEX
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
    print(f"\n📊 Top {k} results:")
    for i, (doc, dist) in enumerate(results, 1):
        preview = doc["content"][:150].replace('\n', ' ') + "..."
        print(f"\n   {i}. Distance: {dist:.4f}")
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
# 6. SAVE/LOAD INDEX FUNCTIONS
# ============================================

def save_index(vector_index, filename="vector_index.json"):
    """Saves the index to a file"""
    data = {
        "vectors": vector_index.vectors,
        "documents": vector_index.documents,
        "vector_dim": vector_index._vector_dim,
        "distance_metric": vector_index._distance_metric
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Index saved to {filename}")

def load_index(filename="vector_index.json", embedding_fn=None):
    """Loads an index from a file"""
    with open(filename, "r") as f:
        data = json.load(f)

    idx = VectorIndex(
        distance_metric=data["distance_metric"],
        embedding_fn=embedding_fn or generate_embedding_batch
    )
    idx.vectors = data["vectors"]
    idx.documents = data["documents"]
    idx._vector_dim = data["vector_dim"]

    print(f"✅ Index loaded from {filename}")
    return idx

# ============================================
# 7. MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("🚀 RAG SYSTEM WITH VECTORINDEX + GROQ")
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

    # 3. Create VectorIndex
    index_file = "vector_index.json"

    if os.path.exists(index_file):
        print(f"\n📂 Loading existing index...")
        index = load_index(index_file, embedding_fn=generate_embedding_batch)
    else:
        print(f"\n🆕 Creating new VectorIndex...")
        index = VectorIndex(
            distance_metric="cosine",
            embedding_fn=generate_embedding_batch
        )

        # Prepare documents
        documents = [{"content": chunk} for chunk in chunks]

        # Add documents to index
        index.add_documents(documents)

        # Save index
        save_index(index, index_file)

    print(f"\n📊 Index statistics:")
    print(f"   • Documents: {len(index)}")
    print(f"   • Dimension: {index._vector_dim}")
    print(f"   • Metric: {index._distance_metric}")

    # 4. Test searches
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
    print("\n📌 VECTORINDEX FEATURES:")
    print("   ✅ Custom Python implementation")
    print("   ✅ Simulated embeddings (no rate limits)")
    print("   ✅ JSON persistence")
    print("   ✅ Cosine and euclidean distance support")