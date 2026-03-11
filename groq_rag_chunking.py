"""
RAG System with Groq - Chunking Strategies
Based on the notebook 001_chunking.ipynb from the course
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

def chunk_by_char(text, chunk_size=150, chunk_overlap=20):
    """
    Splits text into chunks based on character count.
    Useful for models with strict token limits.
    """
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

def chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1):
    """
    Splits text into chunks based on complete sentences.
    Preserves semantic integrity of ideas.
    """
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
    """
    Splits document into chunks based on sections (##).
    Ideal for structured documents like reports.
    """
    # Split by section headers (##)
    pattern = r"\n## "
    sections = re.split(pattern, document_text)

    # The first section may not have the ## prefix
    result = []
    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section
        result.append(section)

    return result

# ============================================
# 3. CHUNK ANALYSIS FUNCTIONS
# ============================================

def analyze_chunks(chunks, method_name):
    """Analyzes and displays information about generated chunks"""
    print(f"\n{'='*60}")
    print(f"📊 CHUNK ANALYSIS - {method_name}")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")

    total_chars = sum(len(chunk) for chunk in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0

    print(f"Total characters: {total_chars}")
    print(f"Average per chunk: {avg_chars:.1f} characters")

    print(f"\n📝 FIRST CHUNK:")
    print("-" * 40)
    print(chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0])

    print(f"\n📝 LAST CHUNK:")
    print("-" * 40)
    print(chunks[-1][:200] + "..." if len(chunks[-1]) > 200 else chunks[-1])

    return chunks

# ============================================
# 4. CHUNK SEARCH FUNCTION
# ============================================

def search_in_chunks(chunks, query):
    """Simple keyword-based search in chunks"""
    print(f"\n🔍 Searching: '{query}'")
    print("-" * 40)

    results = []
    query_lower = query.lower()

    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        if query_lower in chunk_lower:
            relevance = chunk_lower.count(query_lower)
            results.append({
                "chunk_id": i,
                "relevance": relevance,
                "preview": chunk[:150] + "..."
            })

    # Sort by relevance descending
    results.sort(key=lambda x: x["relevance"], reverse=True)

    return results

# ============================================
# 5. RAG QUESTION-ANSWERING FUNCTION
# ============================================

def answer_with_rag(query, chunks, top_k=3):
    """
    Answers a question using the most relevant chunks as context.
    """
    print(f"\n📌 Question: {query}")
    print("=" * 60)

    # Find relevant chunks
    results = search_in_chunks(chunks, query)

    if not results:
        print("❌ No relevant chunks found")
        return "I don't have information about that."

    # Select the top_k most relevant chunks
    top_chunks = results[:top_k]

    print(f"\n📚 Using {len(top_chunks)} chunks as context:")
    for i, r in enumerate(top_chunks, 1):
        print(f"\n   Chunk {i} (relevance: {r['relevance']}):")
        print(f"   {r['preview']}")

    # Build prompt with context
    context = "\n\n".join([chunks[r["chunk_id"]] for r in top_chunks])

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
# 6. MAIN DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("🚀 RAG SYSTEM WITH GROQ - CHUNKING STRATEGIES")
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

    # ============================================
    # TEST DIFFERENT CHUNKING STRATEGIES
    # ============================================

    # 1. Character-based chunking
    chunks_char = chunk_by_char(text, chunk_size=300, chunk_overlap=30)
    analyze_chunks(chunks_char, "CHARACTER-BASED CHUNKING")

    # 2. Sentence-based chunking
    chunks_sentence = chunk_by_sentence(text, max_sentences_per_chunk=3, overlap_sentences=1)
    analyze_chunks(chunks_sentence, "SENTENCE-BASED CHUNKING")

    # 3. Section-based chunking
    chunks_section = chunk_by_section(text)
    analyze_chunks(chunks_section, "SECTION-BASED CHUNKING")

    # ============================================
    # TEST CHUNK SEARCH
    # ============================================

    print("\n" + "=" * 60)
    print("🔍 SEARCH TESTS")
    print("=" * 60)

    # Use the best strategy (sections)
    chunks = chunks_section

    # Search 1: XDR-471
    results = search_in_chunks(chunks, "XDR-471")
    print(f"\n✅ Results found: {len(results)}")

    # Search 2: Financial Analysis
    results = search_in_chunks(chunks, "Financial Analysis")
    print(f"\n✅ Results found: {len(results)}")

    # Search 3: ERR_MEM_ALLOC_FAIL
    results = search_in_chunks(chunks, "ERR_MEM_ALLOC_FAIL")
    print(f"\n✅ Results found: {len(results)}")

    # ============================================
    # TEST RAG RESPONSES
    # ============================================

    print("\n" + "=" * 60)
    print("🤖 RAG TESTS")
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

        answer = answer_with_rag(question, chunks)
        print(f"\n✅ Answer:\n{answer}\n")

        input("Press Enter to continue...")

    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETED")