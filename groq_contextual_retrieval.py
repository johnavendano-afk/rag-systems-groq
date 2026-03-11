"""
RAG System with Contextual Retrieval (Chunk Contextualization)
Improves search by adding document context to each chunk
"""

import os
import re
import math
import json
import random
import string
import numpy as np
from collections import Counter
from typing import Callable, Optional, Any, List, Dict, Tuple, Protocol
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
        """Generates embeddings for one or multiple texts"""
        if isinstance(texts, str):
            return self.model.encode(texts).tolist()
        return self.model.encode(texts, batch_size=batch_size).tolist()

# Global instance for use in indexes
embedding_gen = EmbeddingGenerator()

def generate_embedding(texts):
    """Wrapper for compatibility"""
    return embedding_gen.encode(texts)

# ============================================
# 4. HELPER FUNCTIONS
# ============================================

def add_user_message(messages, content):
    """Adds a user message"""
    messages.append({"role": "user", "content": content})

def add_assistant_message(messages, content):
    """Adds an assistant message"""
    messages.append({"role": "assistant", "content": content})

def text_from_message(message):
    """Extracts text from the response"""
    if hasattr(message, 'content') and message.content:
        return message.content
    return str(message)

def chat(messages, system=None, temperature=1.0, stop_sequences=[], tools=None):
    """Chat function with Groq"""
    chat_messages = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    chat_messages.extend(messages)

    params = {
        "model": MODEL,
        "messages": chat_messages,
        "temperature": temperature,
        "max_tokens": 1000,
    }

    if stop_sequences:
        params["stop"] = stop_sequences

    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    response = client.chat.completions.create(**params)
    return response.choices[0].message

# ============================================
# 5. VECTOR INDEX (EMBEDDINGS)
# ============================================

class VectorIndex:
    """Vector index for semantic search"""

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
        self._embedding_fn = embedding_fn or generate_embedding

    def add_document(self, document: Dict[str, Any]):
        """Adds a document to the index"""
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        vector = self._embedding_fn(content)
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Adds multiple documents to the index (batch)"""
        if not self._embedding_fn:
            raise ValueError("Embedding function not provided")
        if not isinstance(documents, list):
            raise TypeError("Documents must be a list")
        if not documents:
            return

        # Validate documents
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {i} must be a dictionary")
            if "content" not in doc:
                raise ValueError(f"Document at index {i} must contain 'content' key")
            if not isinstance(doc["content"], str):
                raise TypeError(f"Document 'content' at index {i} must be a string")

        # Generate embeddings in batch
        contents = [doc["content"] for doc in documents]
        print(f"   Generating embeddings for {len(contents)} documents...")
        vectors = self._embedding_fn(contents)

        # Add each document with its vector
        for vector, document in zip(vectors, documents):
            self.add_vector(vector=vector, document=document)

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """Searches for the k most similar documents"""
        if not self.vectors:
            return []

        # Get query vector
        if isinstance(query, str):
            if not self._embedding_fn:
                raise ValueError("Embedding function required for string query")
            query_vector = self._embedding_fn(query)
        elif isinstance(query, list) and all(isinstance(x, (int, float)) for x in query):
            query_vector = query
        else:
            raise TypeError("Query must be string or list of numbers")

        # Check dimensions
        if self._vector_dim is None:
            return []
        if len(query_vector) != self._vector_dim:
            raise ValueError(f"Dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}")

        if k <= 0:
            raise ValueError("k must be positive")

        # Calculate distances
        if self._distance_metric == "cosine":
            distances = []
            for stored_vector in self.vectors:
                # For normalized vectors, dot product = cosine similarity
                sim = sum(p*q for p, q in zip(query_vector, stored_vector))
                sim = max(-1.0, min(1.0, sim))
                distances.append((1.0 - sim, stored_vector))  # Convert to distance
        else:
            dist_func = self._euclidean_distance
            distances = [(dist_func(query_vector, v), v) for v in self.vectors]

        # Sort by smallest distance
        sorted_idx = np.argsort([d for d, _ in distances])

        results = []
        for idx in sorted_idx[:k]:
            results.append((self.documents[idx], distances[idx][0]))

        return results

    def add_vector(self, vector, document: Dict[str, Any]):
        """Adds a vector directly to the index"""
        if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
            raise TypeError("Vector must be a list of numbers")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary")
        if "content" not in document:
            raise ValueError("Document must contain 'content' key")

        # Check dimension consistency
        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(f"Dimension mismatch. Expected {self._vector_dim}, got {len(vector)}")

        self.vectors.append(list(vector))
        self.documents.append(document)

    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(vec1, vec2)))

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}')"

# ============================================
# 6. BM25 INDEX
# ============================================

class BM25Index:
    """BM25 index for term-based search"""

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
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [token for token in tokens if token]

    def _update_stats_add(self, doc_tokens: List[str]):
        self._doc_len.append(len(doc_tokens))
        seen_in_doc = set()
        for token in doc_tokens:
            if token not in seen_in_doc:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen_in_doc.add(token)
        self._index_built = False

    def _calculate_idf(self):
        N = len(self.documents)
        self._idf = {}
        for term, freq in self._doc_freqs.items():
            idf_score = math.log(((N - freq + 0.5) / (freq + 0.5)) + 1)
            self._idf[term] = idf_score

    def _build_index(self):
        if not self.documents:
            self._avg_doc_len = 0.0
            self._idf = {}
            self._index_built = True
            return

        self._avg_doc_len = sum(self._doc_len) / len(self.documents)
        self._calculate_idf()
        self._index_built = True

    def add_document(self, document: Dict[str, Any]):
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary")
        if "content" not in document:
            raise ValueError("Document must contain a 'content' key")

        content = document.get("content", "")
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string")

        doc_tokens = self._tokenizer(content)
        self.documents.append(document)
        self._corpus_tokens.append(doc_tokens)
        self._update_stats_add(doc_tokens)

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not isinstance(documents, list):
            raise TypeError("Documents must be a list")
        if not documents:
            return

        print(f"   Tokenizing {len(documents)} documents for BM25...")
        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {i} must be a dictionary")
            if "content" not in doc:
                raise ValueError(f"Document at index {i} must contain 'content' key")
            if not isinstance(doc["content"], str):
                raise TypeError(f"Document 'content' at index {i} must be a string")

            content = doc["content"]
            doc_tokens = self._tokenizer(content)
            self.documents.append(doc)
            self._corpus_tokens.append(doc_tokens)
            self._update_stats_add(doc_tokens)

        self._index_built = False
        self._build_index()

    def _compute_bm25_score(self, query_tokens: List[str], doc_index: int) -> float:
        score = 0.0
        doc_term_counts = Counter(self._corpus_tokens[doc_index])
        doc_length = self._doc_len[doc_index]

        for token in query_tokens:
            if token not in self._idf:
                continue

            idf = self._idf[token]
            term_freq = doc_term_counts.get(token, 0)

            numerator = idf * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self._avg_doc_len)
            )
            score += numerator / (denominator + 1e-9)

        return score

    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents:
            return []

        if not isinstance(query, str):
            raise TypeError("Query must be a string for BM25Index")

        if k <= 0:
            raise ValueError("k must be positive")

        if not self._index_built:
            self._build_index()

        if self._avg_doc_len == 0:
            return []

        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []

        # Calculate scores
        scores = []
        for i in range(len(self.documents)):
            score = self._compute_bm25_score(query_tokens, i)
            if score > 1e-9:
                scores.append((score, self.documents[i]))

        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)

        # Normalize
        if scores:
            max_score = scores[0][0]
            return [(doc, score / max_score) for score, doc in scores[:k]]
        return []

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"BM25Index(count={len(self)}, k1={self.k1}, b={self.b})"

# ============================================
# 7. RETRIEVER WITH RRF AND RERANKER
# ============================================

class SearchIndex(Protocol):
    """Protocol that indexes must implement"""
    def add_document(self, document: Dict[str, Any]) -> None: ...
    def add_documents(self, documents: List[Dict[str, Any]]) -> None: ...
    def search(self, query: Any, k: int = 1) -> List[Tuple[Dict[str, Any], float]]: ...

class Retriever:
    """
    Retriever that combines multiple indexes using RRF
    and optionally re-orders with a reranker
    """

    def __init__(
        self,
        *indexes: SearchIndex,
        reranker_fn: Optional[Callable[[List[Dict[str, Any]], str, int], List[str]]] = None,
    ):
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)
        self._reranker_fn = reranker_fn
        print(f"🔄 Retriever initialized with {len(indexes)} indexes")
        if reranker_fn:
            print(f"   ✅ With reranker enabled")

    def add_document(self, document: Dict[str, Any]):
        """Adds a document to all indexes"""
        if "id" not in document:
            document["id"] = "".join(
                random.choices(string.ascii_letters + string.digits, k=4)
            )
        for index in self._indexes:
            index.add_document(document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Adds multiple documents to all indexes"""
        # Ensure each document has an ID
        for doc in documents:
            if "id" not in doc:
                doc["id"] = "".join(
                    random.choices(string.ascii_letters + string.digits, k=4)
                )

        print(f"\n📦 Adding {len(documents)} documents to all indexes...")
        for i, index in enumerate(self._indexes):
            print(f"   Index {i+1}/{len(self._indexes)}:")
            index.add_documents(documents)
        print(f"✅ Documents added to all indexes")

    def search(
        self, query_text: str, k: int = 1, k_rrf: int = 60
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search using RRF and optionally reranking
        """
        if not isinstance(query_text, str):
            raise TypeError("Query text must be a string")
        if k <= 0:
            raise ValueError("k must be positive")
        if k_rrf < 0:
            raise ValueError("k_rrf must be non-negative")

        # Get results from each index (request more than needed)
        all_results = [index.search(query_text, k=k * 5) for index in self._indexes]

        # Collect rankings for each document
        doc_ranks = {}
        for idx, results in enumerate(all_results):
            for rank, (doc, _) in enumerate(results):
                doc_id = doc.get("id", id(doc))
                if doc_id not in doc_ranks:
                    doc_ranks[doc_id] = {
                        "doc_obj": doc,
                        "ranks": [float("inf")] * len(self._indexes),
                    }
                doc_ranks[doc_id]["ranks"][idx] = rank + 1  # +1 because rank starts at 0

        # Calculate RRF score
        def calc_rrf_score(ranks: List[float]) -> float:
            return sum(1.0 / (k_rrf + r) for r in ranks if r != float("inf"))

        scored_docs: List[Tuple[Dict[str, Any], float]] = [
            (ranks["doc_obj"], calc_rrf_score(ranks["ranks"]))
            for ranks in doc_ranks.values()
        ]

        # Filter and sort
        filtered_docs = [(doc, score) for doc, score in scored_docs if score > 0]
        filtered_docs.sort(key=lambda x: x[1], reverse=True)

        result = filtered_docs[:k]

        # Apply reranker if exists
        if self._reranker_fn is not None and result:
            docs_only = [doc for doc, _ in result]

            # Ensure all have ID
            for doc in docs_only:
                if "id" not in doc:
                    doc["id"] = "".join(
                        random.choices(string.ascii_letters + string.digits, k=4)
                    )

            doc_lookup = {doc["id"]: doc for doc in docs_only}
            reranked_ids = self._reranker_fn(docs_only, query_text, k)

            new_result = []
            original_scores = {doc.get("id", id(doc)): score for doc, score in result}

            for doc_id in reranked_ids:
                if doc_id in doc_lookup:
                    doc = doc_lookup[doc_id]
                    score = original_scores.get(doc_id, 0.0)
                    new_result.append((doc, score))

            result = new_result

        return result

    def __len__(self) -> int:
        return len(self._indexes[0]) if self._indexes else 0

# ============================================
# 8. RERANKER WITH GROQ
# ============================================

def reranker_fn(docs, query_text, k):
    """
    Re-ranks documents using Groq as a reranker
    """
    print(f"\n🔄 Applying Groq reranker to {len(docs)} documents...")

    # Build documents with IDs
    joined_docs = ""
    for doc in docs:
        joined_docs += f"""
        <document>
        <document_id>{doc["id"]}</document_id>
        <document_content>{doc["content"][:500]}</document_content>
        </document>
        """

    prompt = (
        "You are about to be given a set of documents, along with an id of each.\n"
        "Your task is to select the " + str(k) + " most relevant documents to answer the user's question.\n\n"
        "Here is the user's question:\n"
        "<question>\n"
        f"{query_text}\n"
        "</question>\n\n"
        "Here are the documents to select from:\n"
        "<documents>\n"
        f"{joined_docs}\n"
        "</documents>\n\n"
        "Respond in the following format:\n"
        "```json\n"
        "{{\n"
        '    "document_ids": string[] # List document ids, ' + str(k) + ' elements long, sorted in order of decreasing relevance to the user\'s query.\n'
        "}}\n"
        "```"
    )

    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")

    result = chat(messages, stop_sequences=["```"])

    try:
        response_text = text_from_message(result)
        # Clean possible markers
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]

        data = json.loads(response_text.strip())
        return data["document_ids"]
    except Exception as e:
        print(f"❌ Error in reranker: {e}")
        # Fallback: return ids in original order
        return [doc["id"] for doc in docs][:k]

# ============================================
# 9. CHUNK CONTEXTUALIZATION
# ============================================

def add_context(text_chunk, source_context):
    """
    Adds context to a chunk to improve semantic search
    """
    prompt = f"""
    Write a short and succinct snippet of text to situate this chunk within the 
    overall source document for the purposes of improving search retrieval of the chunk. 

    Here is the context from the document (surrounding text):
    <context> 
    {source_context}
    </context> 

    Here is the chunk we want to situate:
    <chunk> 
    {text_chunk}
    </chunk>
    
    Answer only with the succinct context and nothing else. 
    """

    messages = []
    add_user_message(messages, prompt)
    result = chat(messages)

    context_snippet = text_from_message(result).strip()
    return context_snippet + "\n" + text_chunk

# ============================================
# 10. RAG FUNCTION WITH CONTEXTUALIZATION
# ============================================

def answer_with_contextual_rag(query, retriever, k=2):
    """
    Answers a question using the Retriever with contextualized chunks
    """
    print(f"\n📌 Question: {query}")
    print("=" * 60)

    # Search with RRF
    results = retriever.search(query, k=k)

    if not results:
        return "No relevant results found.", []

    # Display results
    print(f"\n📊 Top {k} results (after reranker):")
    for i, (doc, score) in enumerate(results, 1):
        preview = doc["content"][:150].replace('\n', ' ') + "..."
        print(f"\n   {i}. RRF Score: {score:.4f}")
        print(f"      ID: {doc.get('id', 'N/A')}")
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

    return response.choices[0].message.content, results

# ============================================
# 11. MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("🚀 RAG SYSTEM WITH CONTEXTUAL RETRIEVAL")
    print("=" * 70)

    # 1. Load document
    try:
        with open("report.md", "r", encoding="utf-8") as f:
            report_text = f.read()
        print(f"\n📄 Document loaded: report.md")
        print(f"   Length: {len(report_text)} characters")
    except FileNotFoundError:
        print("❌ Error: report.md file not found")
        exit(1)

    # 2. Chunking
    print("\n📦 Generating chunks...")
    chunks = chunk_by_section(report_text)
    print(f"   {len(chunks)} chunks created")

    # 3. Create indexes
    print("\n🔄 Creating indexes...")
    vector_index = VectorIndex(embedding_fn=generate_embedding)
    bm25_index = BM25Index()

    # 4. Create retriever with reranker
    retriever = Retriever(bm25_index, vector_index, reranker_fn=reranker_fn)

    # 5. CONTEXTUALIZE CHUNKS (new!)
    print("\n📝 Contextualizing chunks...")
    num_start_chunks = 2
    num_prev_chunks = 2
    contextualized_chunks = []

    for i, chunk in enumerate(chunks):
        context_parts = []

        # Initial chunks of the document
        context_parts.extend(chunks[: min(num_start_chunks, len(chunks))])

        # Previous chunks
        start_idx = max(0, i - num_prev_chunks)
        context_parts.extend(chunks[start_idx:i])

        context = "\n".join(context_parts)

        print(f"   Contextualizing chunk {i+1}/{len(chunks)}", end="\r")
        contextualized_chunks.append(add_context(chunk, context))

    print(f"\n✅ {len(contextualized_chunks)} chunks contextualized")

    # 6. Add contextualized documents to retriever
    documents = [{"content": chunk} for chunk in contextualized_chunks]
    retriever.add_documents(documents)

    print(f"\n📊 Statistics:")
    print(f"   • Documents: {len(retriever)}")
    print(f"   • Indexes: {len(retriever._indexes)}")
    print(f"   • Reranker: Enabled")
    print(f"   • Contextualization: Enabled")

    # 7. Test search
    print("\n" + "=" * 70)
    print("🔍 SEARCH TEST WITH CONTEXTUALIZATION")
    print("=" * 70)

    question = "what did the eng team do with INC-2023-Q4-011?"

    print(f"\n{'='*70}")
    print(f"📝 QUESTION")
    print(f"{'='*70}")

    answer, results = answer_with_contextual_rag(question, retriever, k=2)
    print(f"\n✅ Answer:\n{answer}\n")

    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETED")
    print("\n📌 SYSTEM FEATURES:")
    print("   ✅ Chunk contextualization")
    print("   ✅ Multiple indexes (BM25 + Embeddings)")
    print("   ✅ Rank fusion (RRF)")
    print("   ✅ LLM reranking (Groq)")
    print("   ✅ Higher search precision")