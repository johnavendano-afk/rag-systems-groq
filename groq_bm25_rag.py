"""
RAG System with BM25 + Groq
Term-based search (complementary to semantic embeddings)
"""

import os
import re
import math
import json
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
# 3. BM25 INDEX IMPLEMENTATION
# ============================================

class BM25Index:
    """
    BM25 Index for term-based search.
    More efficient than embeddings for exact keyword search.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ):
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
        """Default tokenizer (lowercase + split by non-alphanumeric characters)"""
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [token for token in tokens if token]

    def _update_stats_add(self, doc_tokens: List[str]):
        """Updates statistics when adding a document"""
        self._doc_len.append(len(doc_tokens))

        seen_in_doc = set()
        for token in doc_tokens:
            if token not in seen_in_doc:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen_in_doc.add(token)

        self._index_built = False

    def _calculate_idf(self):
        """Calculates IDF (Inverse Document Frequency) for all terms"""
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            # BM25 formula for IDF
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def _build_index(self):
        """Builds the index (calculates global statistics)"""
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return

        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        self._calculate_idf()
        self._index_built = True

    def add_document(self, document: Dict[str, Any]):
        """Adds a document to the index"""
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        doc_tokens = self._tokenizer(content)

        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        self._update_stats_add(doc_tokens)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Adds multiple documents to the index"""
        print(f"\n📦 Adding {len(documents)} documents to the BM25 index...")
        for i, doc in enumerate(documents):
            self.add_document(doc)
            print(f"   📍 Document {i+1}/{len(documents)} added", end="\r")
        self._build_index()
        print(f"\n✅ BM25 index created with {len(self)} documents")

    def _compute_bm25_score(
        self, query_tokens: List[str], doc_index: int
    ) -> float:
        """Calculates BM25 score for a document"""
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_index])
        doc_length = self._doc_len[doc_index]

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            # BM25 formula
            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self._avg_doc_len)
            )
            score += numerator / (denominator + 1e-9)

        return score

    def search(
        self,
        query_text: str,
        k: int = 1,
        score_normalization_factor: float = 0.1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches for the k most relevant documents for the query.
        """
        if not self.documents:
            return []

        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string.")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        query_tokens = self._tokenizer(query_text)
        if not query_tokens:
            return []

        # Calculate scores for all documents
        raw_scores = []
        for i in range(len(self.documents)):
            raw_score = self._compute_bm25_score(query_tokens, i)
            if raw_score > 1e-9:
                raw_scores.append((raw_score, self.documents[i]))

        # Sort by score descending
        raw_scores.sort(key=lambda item: item[0], reverse=True)

        # Normalize scores (optional)
        normalized_results = []
        for raw_score, doc in raw_scores[:k]:
            normalized_score = math.exp(-score_normalization_factor * raw_score)
            normalized_results.append((doc, normalized_score))

        return normalized_results

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"BM25Index(count={len(self)}, k1={self.k1}, b={self.b}, index_built={self._index_built})"

# ============================================
# 4. RAG FUNCTION WITH BM25 + GROQ
# ============================================

def answer_with_bm25(query, bm25_index, k=2):
    """
    Answers a question using BM25 to retrieve chunks.
    """
    print(f"\n📌 Question: {query}")
    print("=" * 60)

    # Search relevant chunks with BM25
    results = bm25_index.search(query, k=k)

    if not results:
        return "No relevant results found."

    # Display results
    print(f"\n📊 Top {k} BM25 results:")
    for i, (doc, score) in enumerate(results, 1):
        preview = doc["content"][:150].replace('\n', ' ') + "..."
        print(f"\n   {i}. Score: {score:.4f}")
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
# 5. SAVE/LOAD INDEX FUNCTIONS
# ============================================

def save_bm25_index(index, filename="bm25_index.json"):
    """Saves the BM25 index to a file (simplified)"""
    # Note: Only documents are saved, tokens are regenerated on load
    data = {
        "documents": index.documents,
        "k1": index.k1,
        "b": index.b
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ BM25 index saved to {filename}")

def load_bm25_index(filename="bm25_index.json"):
    """Loads a BM25 index from a file"""
    with open(filename, "r") as f:
        data = json.load(f)

    index = BM25Index(k1=data["k1"], b=data["b"])
    for doc in data["documents"]:
        index.add_document(doc)

    print(f"✅ BM25 index loaded from {filename}")
    return index

# ============================================
# 6. MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("🚀 RAG SYSTEM WITH BM25 + GROQ")
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

    # 3. Create BM25 index
    index_file = "bm25_index.json"

    if os.path.exists(index_file):
        print(f"\n📂 Loading existing BM25 index...")
        bm25 = load_bm25_index(index_file)
    else:
        print(f"\n🆕 Creating new BM25 index...")
        bm25 = BM25Index(k1=1.5, b=0.75)

        # Prepare documents
        documents = [{"content": chunk} for chunk in chunks]

        # Add documents to index
        bm25.add_documents(documents)

        # Save index
        save_bm25_index(bm25, index_file)

    print(f"\n📊 BM25 Index Statistics:")
    print(f"   • Documents: {len(bm25)}")
    print(f"   • Parameters: k1={bm25.k1}, b={bm25.b}")

    # 4. Test searches
    print("\n" + "=" * 60)
    print("🔍 BM25 SEARCH TESTS")
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

        answer = answer_with_bm25(question, bm25, k=2)
        print(f"\n✅ Answer:\n{answer}\n")

        if i < len(questions):
            input("Press Enter to continue...")

    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETED")
    print("\n📌 BM25 FEATURES:")
    print("   ✅ Term-based search (keywords)")
    print("   ✅ No embeddings needed")
    print("   ✅ Very fast and efficient")
    print("   ✅ Ideal for exact search")