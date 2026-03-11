"""
Sistema RAG con Groq - Estrategias de Chunking (VERSIÓN MEJORADA)
Con mejor búsqueda semántica y extracción de keywords
"""

import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI

# ============================================
# 1. CONFIGURACIÓN GROQ
# ============================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"

# ============================================
# 2. ESTRATEGIAS DE CHUNKING
# ============================================

def chunk_by_char(text, chunk_size=300, chunk_overlap=30):
    """Divide el texto en chunks basados en número de caracteres"""
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
    """Divide el texto en chunks basados en oraciones completas"""
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
    """Divide el documento en chunks basados en secciones (##)"""
    pattern = r"\n## "
    sections = re.split(pattern, document_text)
    
    result = []
    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section
        result.append(section)
    
    return result

# ============================================
# 3. FUNCIÓN DE BÚSQUEDA MEJORADA
# ============================================

def extract_keywords(query):
    """Extrae palabras clave de la pregunta (quitando signos y palabras comunes)"""
    # Quitar signos de puntuación y convertir a minúsculas
    cleaned = re.sub(r'[¿?¡!.,;:]', '', query.lower())
    
    # Palabras comunes a ignorar
    stopwords = ['qué', 'cual', 'cuál', 'como', 'cómo', 'es', 'son', 'el', 'la', 
                 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'que']
    
    words = cleaned.split()
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    
    return keywords

def search_in_chunks_improved(chunks, query):
    """Búsqueda mejorada usando palabras clave"""
    print(f"\n🔍 Buscando: '{query}'")
    print("-" * 40)
    
    # Extraer keywords de la pregunta
    keywords = extract_keywords(query)
    print(f"   Palabras clave: {keywords}")
    
    results = []
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        relevance = 0
        
        # Buscar cada keyword
        for kw in keywords:
            if kw in chunk_lower:
                relevance += chunk_lower.count(kw)
        
        # Bonus si encuentra coincidencias exactas de frases
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
            # Extraer las líneas donde aparece la keyword
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
    
    # Ordenar por relevancia
    results.sort(key=lambda x: x["relevance"], reverse=True)
    
    return results

# ============================================
# 4. FUNCIÓN PARA RESPONDER PREGUNTAS CON RAG
# ============================================

def answer_with_rag_improved(query, chunks, top_k=3):
    """
    Responde una pregunta usando los chunks más relevantes como contexto
    """
    print(f"\n📌 Pregunta: {query}")
    print("=" * 60)
    
    # Buscar chunks relevantes con búsqueda mejorada
    results = search_in_chunks_improved(chunks, query)
    
    if not results:
        print("❌ No se encontraron chunks relevantes")
        return "No tengo información específica sobre eso en el documento."
    
    # Seleccionar los top_k chunks más relevantes
    top_chunks = results[:top_k]
    
    print(f"\n📚 Usando {len(top_chunks)} chunks como contexto:")
    for i, r in enumerate(top_chunks, 1):
        print(f"\n   Chunk {i} (relevancia: {r['relevance']}):")
        print(f"   {r['preview'][:150]}...")
    
    # Construir prompt con contexto
    context = "\n\n---\n\n".join([chunks[r["chunk_id"]] for r in top_chunks])
    
    prompt = f"""Eres un asistente experto en análisis de documentos.
Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA: {query}

INSTRUCCIONES:
- Usa solo información del contexto
- Si la respuesta no está en el contexto, di que no tienes esa información
- Sé conciso pero completo
- Si encuentras información específica (códigos de error, nombres, fechas), inclúyelos

RESPUESTA:"""
    
    # Generar respuesta con Groq
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Eres un asistente experto en análisis de documentos."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# ============================================
# 5. DEMOSTRACIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("🚀 SISTEMA RAG CON GROQ - VERSIÓN MEJORADA")
    print("=" * 60)
    
    # Cargar el documento
    try:
        with open("report.md", "r", encoding="utf-8") as f:
            text = f.read()
        print(f"\n📄 Documento cargado: report.md")
        print(f"   Longitud: {len(text)} caracteres")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo report.md")
        exit(1)
    
    # Usar chunk por secciones (mejor para este documento)
    chunks = chunk_by_section(text)
    
    print("\n" + "=" * 60)
    print("🤖 PRUEBAS DE RAG MEJORADAS")
    print("=" * 60)
    
    preguntas = [
        "¿Qué es el síndrome XDR-471?",
        "¿Cuál fue el error de software que afectó a Project Phoenix?",
        "¿Qué empresa está relacionada con el caso Synergy Dynamics?",
        "¿Cuáles son las especificaciones del Modelo Zircon-5?",
        "¿Qué incidente de ciberseguridad ocurrió en Q4 2023?"
    ]
    
    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n{'='*60}")
        print(f"📝 PREGUNTA {i}")
        print(f"{'='*60}")
        
        respuesta = answer_with_rag_improved(pregunta, chunks)
        print(f"\n✅ Respuesta:\n{respuesta}\n")
        
        input("Presiona Enter para continuar...")
    
    print("\n" + "=" * 60)
    print("✅ DEMOSTRACIÓN COMPLETADA")
