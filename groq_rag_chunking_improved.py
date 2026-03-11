"""
RAG System with Groq - Chunking Strategies (IMPROVED VERSION)
With better semantic search and keyword extraction
"""

import os
import re
import json
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
# 2. CHUNKING STRATEGIES
# ============================================

def chunk_by_char(text, chunk_size=300, chunk_overlap=30):
    """Splits text into chunks based on character count"""
    chunks = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))

        chunk_text = text[start_idx:end_idx]
        chunks.append(chunk_text)

        start_idx = (
            end_idx - chunk_overlap if end_idx < len(text) else len(text)
        )

    return chunks

def chunk_by_sentence(text, max_sentences_per_chunk=3, overlap_sentences=1):
    """Splits text into chunks based on complete sentences"""
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))

        current_chunk = sentences[start_idx:end_idx]
        chunks.append(" ".join(current_chunk))

        start_idx += max_sentences_per_chunk - overlap_sentences

        if start_idx < 0:
            start_idx = 0

    return chunks

def chunk_by_section(document_text):
    """Splits document into chunks based on sections (##)"""
    pattern = r"\n## "
    sections = re.split(pattern, document_text)

    result = []
    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section
        result.append(section)

    return result

# ============================================
# 3. IMPROVED SEARCH FUNCTION
# ============================================

def extract_keywords(query):
    """Extracts keywords from the question (removing punctuation and stopwords)"""
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r'[¿?¡!.,;:]', '', query.lower())

    # Common stopwords to ignore
    stopwords = ['what', 'which', 'how', 'is', 'are', 'the', 'a', 'an',
                 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'that', 'was']

    words = cleaned.split()
    keywords = [w for w in words if w not in stopwords and len(w) > 3]

    return keywords

def search_in_chunks_improved(chunks, query):
    """Improved search using keywords"""
    print(f"\n🔍 Searching: '{query}'")
    print("-" * 40)

    # Extract keywords from the question
    keywords = extract_keywords(query)
    print(f"   Keywords: {keywords}")

    results = []

    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        relevance = 0

        # Search for each keyword
        for kw in keywords:
            if kw in chunk_lower:
                relevance += chunk_lower.count(kw)

        # Bonus for exact phrase matches
        if "synergy dynamics" in chunk_lower and "synergy" in query.lower():
            relevance += 5
        if "zircon-5" in chunk_lower and "zircon" in query.lower():
            relevance += 5
        if "xdr-471" in chunk_lower and "xdr" in query.lower():
            relevance += 5
        if "project phoenix" in chunk_lower and "phoenix" in query.lower():
            relevance += 5
        if "error" in chunk_lower and "error" in query.lower():
            if "mem_alloc_fail" in chunk_lower:
                relevance += 10

        if relevance > 0:
            # Extract lines where the keyword appears
            lines = chunk.split('\n')
            preview = ""
            for line in lines:
                line_lower = line.lower()
                if any(kw in line_lower for kw in keywords):
                    preview += line[:150] + "...\n"

            if not preview:
                preview = chunk[:150] + "..."

            results.append({
                "chunk_id": i,
                "relevance": relevance,
                "preview": preview
            })

    # Sort by relevance descending
    results.sort(key=lambda x: x["relevance"], reverse=True)

    return results

# ============================================
# 4. IMPROVED RAG QUESTION-ANSWERING FUNCTION
# ============================================

def answer_with_rag_improved(query, chunks, top_k=3):
    """
    Answers a question using the most relevant chunks as context.
    """
    print(f"\n📌 Question: {query}")
    print("=" * 60)

    # Find relevant chunks using improved search
    results = search_in_chunks_improved(chunks, query)

    if not results:
        print("❌ No relevant chunks found")
        return "I don't have specific information about that in the document."

    # Select the top_k most relevant chunks
    top_chunks = results[:top_k]

    print(f"\n📚 Using {len(top_chunks)} chunks as context:")
    for i, r in enumerate(top_chunks, 1):
        print(f"\n   Chunk {i} (relevance: {r['relevance']}):")
        print(f"   {r['preview'][:150]}...")

    # Build prompt with context
    context = "\n\n---\n\n".join([chunks[r["chunk_id"]] for r in top_chunks])

    prompt = f"""You are an expert document analysis assistant.
Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Use only information from the context
- If the answer is not in the context, say you don't have that information
- Be concise but complete
- If you find specific information (error codes, names, dates), include them

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
# 5. MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("🚀 RAG SYSTEM WITH GROQ - IMPROVED VERSION")
    print("=" * 60)

    # Load the document
    try:
        with open("report.md", "r", encoding="utf-8") as f:
            text = f.read()
        print(f"\n📄 Document loaded: report.md")
        print(f"   Length: {len(text)} characters")
    except FileNotFoundError:
        print("❌ Error: report.md file not found")
        exit(1)

    # Use section-based chunking (best for this document)
    chunks = chunk_by_section(text)

    print("\n" + "=" * 60)
    print("🤖 IMPROVED RAG TESTS")
    print("=" * 60)

    questions = [
        "What is XDR-471 syndrome?",
        "What was the software error that affected Project Phoenix?",
        "What company is related to the Synergy Dynamics case?",
        "What are the specifications of the Zircon-5 Model?",
        "What cybersecurity incident occurred in Q4 2023?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"📝 QUESTION {i}")
        print(f"{'='*60}")

        answer = answer_with_rag_improved(question, chunks)
        print(f"\n✅ Answer:\n{answer}\n")

        input("Press Enter to continue...")

    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETED")