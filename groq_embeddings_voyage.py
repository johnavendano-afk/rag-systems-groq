"""
Embedding System with VoyageAI + Groq
Version with rate limit handling and test mode
Based on notebook 002_embeddings.ipynb
"""

import os
import re
import json
import numpy as np
from dotenv import load_dotenv
import voyageai
from openai import OpenAI

# ============================================
# 1. API CONFIGURATION
# ============================================

load_dotenv()

# Configure VoyageAI (for embeddings)
vo_client = voyageai.Client()

# Configure Groq (for generation)
groq_client = OpenAI(
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

def chunk_by_char(text, chunk_size=500, chunk_overlap=50):
    """Splits text into chunks based on character count"""
    chunks = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunks.append(text[start_idx:end_idx])
        start_idx = end_idx - chunk_overlap if end_idx < len(text) else len(text)

    return chunks

# ============================================
# 3. EMBEDDING FUNCTIONS (WITH ERROR HANDLING)
# ============================================

def generate_embedding(text, model="voyage-3-large", input_type="document"):
    """
    Generates an embedding for a text using VoyageAI
    With error handling for rate limits
    """
    print(f"📊 Generating embedding for text of {len(text)} characters...")

    try:
        result = vo_client.embed(
            [text],
            model=model,
            input_type=input_type
        )

        embedding = result.embeddings[0]
        print(f"   ✅ Embedding dimension: {len(embedding)}")
        return embedding

    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   ⚠️ Returning simulated embedding (zero vector)")
        # Return a zero vector as fallback (dimension 1024 for voyage-3-large)
        return [0.0] * 1024

def generate_embeddings_for_chunks(chunks, batch_size=2, max_chunks=None):
    """
    Generates embeddings for multiple chunks, with optional limit
    to avoid rate limits
    """
    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"\n📊 Generating embeddings for {len(chunks)} chunks (limited mode)...")
    else:
        print(f"\n📊 Generating embeddings for {len(chunks)} chunks...")

    embeddings = []
    successful = 0
    failed = 0

    for i, chunk in enumerate(chunks):
        print(f"\n   📍 Chunk {i+1}/{len(chunks)}")
        try:
            emb = generate_embedding(chunk, input_type="document")
            embeddings.append(emb)
            successful += 1
        except Exception as e:
            print(f"   ❌ Critical error: {e}")
            embeddings.append([0.0] * 1024)  # Fallback
            failed += 1

        # Small pause to avoid saturating the API
        if i < len(chunks) - 1:
            print("   ⏱️  Waiting 1 second...")
            import time
            time.sleep(1)

    print(f"\n✅ Summary: {successful} successful, {failed} fallbacks")
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
    Searches for semantically similar chunks to the query
    """
    print(f"\n🔍 Searching: '{query}'")

    # Generate embedding for the query
    query_embedding = generate_embedding(query, input_type="query")

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
# 5. RAG FUNCTION WITH EMBEDDINGS
# ============================================

def answer_with_rag(query, chunks, chunk_embeddings, top_k=3):
    """
    Answers a question using the most semantically relevant chunks
    """
    print(f"\n📌 Question: {query}")
    print("=" * 60)

    # Find semantically relevant chunks
    results = search_semantic(query, chunks, chunk_embeddings, top_k)

    if not results:
        return "No relevant results found."

    # Build context with the most relevant chunks
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
    response = groq_client.chat.completions.create(
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
# 6. SAVE/LOAD EMBEDDINGS FUNCTIONS
# ============================================

def save_embeddings(embeddings, chunks, filename="embeddings.json"):
    """Saves embeddings and chunks to a file"""
    data = {
        "chunks": chunks,
        "embeddings": embeddings
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Embeddings saved to {filename}")

def load_embeddings(filename="embeddings.json"):
    """Loads embeddings and chunks from a file"""
    with open(filename, "r") as f:
        data = json.load(f)
    print(f"✅ Embeddings loaded from {filename}")
    return data["chunks"], data["embeddings"]

# ============================================
# 7. MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("🚀 RAG SYSTEM WITH VOYAGEAI + GROQ")
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

    # 3. Check if embeddings already exist
    embeddings_file = "embeddings.json"
    test_mode = True  # Change to False when API key is configured with payment

    if os.path.exists(embeddings_file) and not test_mode:
        print(f"\n📂 Loading existing embeddings...")
        chunks, embeddings = load_embeddings(embeddings_file)
    else:
        # 4. Generate embeddings with limit to avoid rate limits
        print(f"\n🆕 Generating new embeddings (TEST MODE - only 2 chunks)...")
        print(f"   ⚠️ To process all chunks, add a payment method in VoyageAI")

        # ONLY process 2 chunks to avoid rate limits
        max_chunks = 2
        embeddings = generate_embeddings_for_chunks(chunks, max_chunks=max_chunks)

        # Save generated embeddings
        save_embeddings(embeddings, chunks[:max_chunks], embeddings_file)
        print(f"   ℹ️ Note: Only {max_chunks} chunks saved due to test mode")

        # Use only the first chunks for the rest of the demo
        chunks = chunks[:max_chunks]

    # 5. Test semantic search
    print("\n" + "=" * 60)
    print("🔍 SEMANTIC SEARCH TESTS")
    print("=" * 60)

    questions = [
        "What is XDR-471 syndrome?",
        "Software errors in Project Phoenix",
        "Zircon-5 Model specifications",
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
    print("\n📌 NOTE: This demo used only 2 chunks due to rate limits.")
    print("   To process the full document:")
    print("   1. Add a payment method at https://dashboard.voyageai.com/")
    print("   2. Change test_mode = False in the code")
    print("   3. Run the script again")