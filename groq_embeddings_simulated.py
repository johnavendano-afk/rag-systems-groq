"""
RAG System with SIMULATED Embeddings using Groq
No VoyageAI dependency - avoids rate limits
"""

import os
import re
import json
import numpy as np
import hashlib
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
# 3. SIMULATED EMBEDDINGS (hash-based)
# ============================================

def generate_simulated_embedding(text, dimension=384):
    """
    Generates a SIMULATED embedding based on the text hash.
    This is NOT a real embedding, but allows testing the RAG flow.
    """
    # Create a hash of the text
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()

    # Generate pseudo-random but deterministic vector
    np.random.seed(int.from_bytes(hash_bytes[:4], 'little'))
    embedding = np.random.randn(dimension)

    # Normalize the vector
    embedding = embedding / np.linalg.norm(embedding)

    print(f"📊 Simulated embedding generated (dimension: {dimension})")
    return embedding.tolist()

def generate_embeddings_for_chunks(chunks):
    """Generates simulated embeddings for all chunks"""
    print(f"\n📊 Generating SIMULATED embeddings for {len(chunks)} chunks...")

    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"   📍 Chunk {i+1}/{len(chunks)}", end="\r")
        emb = generate_simulated_embedding(chunk)
        embeddings.append(emb)

    print(f"\n✅ Simulated embeddings generated!")
    return embeddings

# ============================================
# 4. SEMANTIC SEARCH FUNCTIONS
# ============================================

def cosine_similarity(a, b):
    """Calculates cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_semantic(query, chunks, chunk_embeddings, top_k=3):
    """
    Searches for semantically similar chunks to the query.
    """
    print(f"\n🔍 Searching: '{query}'")

    # Generate simulated embedding for the query
    query_embedding = generate_simulated_embedding(query)

    # Calculate similarities
    similarities = []
    for i, emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Display results
    print(f"\n📊 Top {top_k} results:")
    for i in range(min(top_k, len(similarities))):
        idx, sim = similarities[i]
        preview = chunks[idx][:150].replace('\n', ' ') + "..."
        print(f"\n   {i+1}. Similarity: {sim:.4f}")
        print(f"      {preview}")

    return similarities[:top_k]

# ============================================
# 5. RAG WITH SIMULATED EMBEDDINGS
# ============================================

def answer_with_rag(query, chunks, chunk_embeddings, top_k=3):
    """
    Answers a question using the most relevant chunks.
    """
    print(f"\n📌 Question: {query}")
    print("=" * 60)

    # Find relevant chunks
    results = search_semantic(query, chunks, chunk_embeddings, top_k)

    if not results:
        return "No relevant results found."

    # Build context from most relevant chunks
    context = "\n\n---\n\n".join([chunks[idx] for idx, _ in results])

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
# 6. SAVE/LOAD EMBEDDINGS
# ============================================

def save_embeddings(embeddings, chunks, filename="embeddings_simulated.json"):
    """Saves embeddings and chunks to a file"""
    data = {
        "chunks": chunks,
        "embeddings": embeddings
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Embeddings saved to {filename}")

def load_embeddings(filename="embeddings_simulated.json"):
    """Loads embeddings and chunks from a file"""
    with open(filename, "r") as f:
        data = json.load(f)
    print(f"✅ Embeddings loaded from {filename}")
    return data["chunks"], data["embeddings"]

# ============================================
# 7. MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("🚀 RAG SYSTEM WITH SIMULATED EMBEDDINGS")
    print("=" * 60)
    print("📌 NOTE: This version does NOT use VoyageAI")
    print("   Embeddings are simulated (hash-based)")
    print("   Perfect for testing without rate limits")

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

    # 3. Check if embeddings already exist
    embeddings_file = "embeddings_simulated.json"

    if os.path.exists(embeddings_file):
        print(f"\n📂 Loading existing embeddings...")
        chunks, embeddings = load_embeddings(embeddings_file)
    else:
        # 4. Generate simulated embeddings (no rate limits)
        print(f"\n🆕 Generating SIMULATED embeddings...")
        embeddings = generate_embeddings_for_chunks(chunks)
        save_embeddings(embeddings, chunks, embeddings_file)

    # 5. Test semantic search
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

        answer = answer_with_rag(question, chunks, embeddings)
        print(f"\n✅ Answer:\n{answer}\n")

        if i < len(questions):
            input("Press Enter to continue...")

    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETED")
    print("\n📌 ADVANTAGES OF THIS VERSION:")
    print("   ✅ No rate limits")
    print("   ✅ No VoyageAI API key needed")
    print("   ✅ All chunks processed")
    print("   ✅ Ideal for testing and development")