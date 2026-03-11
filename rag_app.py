"""
RAG Application with Streamlit + Groq + Sentence Transformers
Web interface for querying documents with semantic search
"""

import os
import re
import json
import math
import streamlit as st
import numpy as np
from typing import Optional, Any, List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# Page configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 1. LOAD CONFIGURATION
# ============================================

load_dotenv()

# Check API key
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY not found in .env file")
    st.stop()

# ============================================
# 2. INITIALIZE CLIENTS (with cache)
# ============================================

@st.cache_resource
def get_groq_client():
    """Initializes Groq client (cached)"""
    return OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

@st.cache_resource
def get_embedding_model():
    """Initializes embedding model (cached)"""
    from sentence_transformers import SentenceTransformer
    with st.spinner("🔄 Loading embedding model..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        st.success(f"✅ Model loaded (dimension: {model.get_sentence_embedding_dimension()})")
        return model

client = get_groq_client()
embedding_model = get_embedding_model()

# ============================================
# 3. VECTORINDEX (COMPLETE CLASS)
# ============================================

class VectorIndex:
    """Vector index for semantic search"""

    def __init__(self, distance_metric: str = "cosine", normalize_vectors: bool = True):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None
        self._distance_metric = distance_metric
        self._normalize_vectors = normalize_vectors

    def _normalize(self, vector):
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            return [x/norm for x in vector]
        return vector

    def add_document(self, document: Dict[str, Any], embedding):
        """Adds a document with its embedding"""
        vector = embedding
        if self._normalize_vectors and self._distance_metric == "cosine":
            vector = self._normalize(vector)

        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(f"Inconsistent dimension")

        self.vectors.append(vector)
        self.documents.append(document)

    def add_documents_batch(self, documents: List[Dict[str, Any]], embeddings):
        """Adds multiple documents with their embeddings"""
        for doc, emb in zip(documents, embeddings):
            self.add_document(doc, emb)

    def search(self, query_vector, k: int = 3):
        """Searches for the k most similar documents"""
        if not self.vectors:
            return []

        if len(query_vector) != self._vector_dim:
            raise ValueError(f"Incorrect dimension")

        # Calculate similarities
        similarities = []
        for stored_vector in self.vectors:
            if self._distance_metric == "cosine":
                # Dot product = cosine similarity if normalized
                sim = sum(p*q for p, q in zip(query_vector, stored_vector))
                sim = max(-1.0, min(1.0, sim))
            else:
                # Euclidean distance
                dist = math.sqrt(sum((p-q)**2 for p, q in zip(query_vector, stored_vector)))
                sim = 1.0 / (1.0 + dist)  # Convert to similarity

            similarities.append(sim)

        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = [(self.documents[i], similarities[i]) for i in top_indices]

        return results

    def __len__(self):
        return len(self.vectors)

# ============================================
# 4. PROCESSING FUNCTIONS
# ============================================

def chunk_by_section(text):
    """Splits text into sections by ##"""
    pattern = r"\n## "
    sections = re.split(pattern, text)
    result = []
    for i, sec in enumerate(sections):
        if i > 0:
            sec = "## " + sec
        result.append(sec.strip())
    return result

def load_document(file):
    """Loads document from uploaded file"""
    content = file.read().decode("utf-8")
    return content

@st.cache_data
def process_document(content):
    """Processes full document (cached)"""
    chunks = chunk_by_section(content)

    # Generate embeddings
    with st.spinner(f"🔄 Generating embeddings for {len(chunks)} chunks..."):
        embeddings = embedding_model.encode(chunks)

    # Create index
    index = VectorIndex()
    documents = [{"content": chunk} for chunk in chunks]
    index.add_documents_batch(documents, embeddings)

    return chunks, index

def answer_question(question, index, k=3):
    """Answers a question using RAG"""

    # Generate question embedding
    question_embedding = embedding_model.encode(question)

    # Search relevant chunks
    results = index.search(question_embedding, k=k)

    if not results:
        return "No relevant results found.", []

    # Build context
    context = "\n\n---\n\n".join([doc["content"] for doc, _ in results])

    prompt = f"""You are an expert document analysis assistant.
Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Use only information from the context
- If the answer is not in the context, say you don't have that information
- Be concise but complete

ANSWER:"""

    # Generate response with Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert document analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content, results

# ============================================
# 5. STREAMLIT INTERFACE
# ============================================

def main():
    st.title("🤖 RAG Assistant with Groq")
    st.markdown("---")

    # Sidebar - Configuration
    with st.sidebar:
        st.header("📁 Document")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload a .md or .txt file",
            type=["md", "txt"],
            help="The document will be processed and indexed"
        )

        st.markdown("---")

        st.header("⚙️ Settings")
        k_results = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=5,
            value=2,
            help="More chunks give more context but may dilute the answer"
        )

        st.markdown("---")

        st.header("ℹ️ Info")
        st.markdown("""
        **Models:**
        - Embeddings: all-MiniLM-L6-v2
        - Generation: Llama 3.3 70B

        **Features:**
        - Semantic search
        - No rate limits
        - Local processing
        """)

    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Queries")

        # Session state
        if "index" not in st.session_state:
            st.session_state.index = None
        if "chunks" not in st.session_state:
            st.session_state.chunks = []
        if "history" not in st.session_state:
            st.session_state.history = []

    with col2:
        st.header("📊 Statistics")
        if st.session_state.index:
            st.metric("Indexed chunks", len(st.session_state.index))
            st.metric("Embedding dimension", st.session_state.index._vector_dim)
        else:
            st.info("No document loaded")

    # Process uploaded document
    if uploaded_file is not None:
        with st.spinner("📄 Processing document..."):
            content = load_document(uploaded_file)
            chunks, index = process_document(content)

            st.session_state.chunks = chunks
            st.session_state.index = index

            st.success(f"✅ Document processed: {len(chunks)} chunks")

            # Show preview
            with st.expander("📖 Chunk preview"):
                for i, chunk in enumerate(chunks[:3]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    st.markdown("---")

    # Question area
    st.markdown("---")

    # Question input
    question = st.text_input("📝 Your question:", placeholder="Type your question here...")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear history", use_container_width=True)

    if clear_button:
        st.session_state.history = []
        st.rerun()

    # Process question
    if ask_button and question:
        if st.session_state.index is None:
            st.warning("⚠️ Please upload a document first")
        else:
            with st.spinner("🤔 Thinking..."):
                answer, results = answer_question(question, st.session_state.index, k=k_results)

                # Save to history
                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "results": results
                })

    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.header("📜 Query history")

        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"❓ {item['question']}", expanded=i==0):
                st.markdown("**Answer:**")
                st.write(item['answer'])

                with st.expander("📚 Chunks used"):
                    for j, (doc, sim) in enumerate(item['results']):
                        st.markdown(f"**Chunk {j+1}** (similarity: {sim:.4f})")
                        st.text(doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content'])
                        st.markdown("---")

# ============================================
# 6. RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()