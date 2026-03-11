"""
Sistema RAG con Embeddings SIMULADOS usando Groq
Sin dependencia de VoyageAI - evita rate limits
"""

import os
import re
import json
import numpy as np
import hashlib
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
# 2. FUNCIONES DE CHUNKING
# ============================================

def chunk_by_section(document_text):
    """Divide el documento en chunks basados en secciones (##)"""
    pattern = r"\n## "
    sections = re.split(pattern, document_text)
    
    result = []
    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section
        result.append(section.strip())
    
    return result

# ============================================
# 3. EMBEDDINGS SIMULADOS (basados en hash)
# ============================================

def generate_simulated_embedding(text, dimension=384):
    """
    Genera un embedding SIMULADO basado en el hash del texto
    Esto NO es un embedding real, pero permite probar el flujo RAG
    """
    # Crear un hash del texto
    hash_obj = hashlib.md5(text.encode())
    hash_bytes = hash_obj.digest()
    
    # Generar vector pseudo-aleatorio pero determinístico
    np.random.seed(int.from_bytes(hash_bytes[:4], 'little'))
    embedding = np.random.randn(dimension)
    
    # Normalizar
    embedding = embedding / np.linalg.norm(embedding)
    
    print(f"📊 Embedding simulado generado (dimensión: {dimension})")
    return embedding.tolist()

def generate_embeddings_for_chunks(chunks):
    """Genera embeddings simulados para todos los chunks"""
    print(f"\n📊 Generando embeddings SIMULADOS para {len(chunks)} chunks...")
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"   📍 Chunk {i+1}/{len(chunks)}", end="\r")
        emb = generate_simulated_embedding(chunk)
        embeddings.append(emb)
    
    print(f"\n✅ Embeddings simulados generados!")
    return embeddings

# ============================================
# 4. FUNCIONES DE BÚSQUEDA SEMÁNTICA
# ============================================

def cosine_similarity(a, b):
    """Calcula la similitud coseno entre dos vectores"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_semantic(query, chunks, chunk_embeddings, top_k=3):
    """
    Busca chunks semánticamente similares a la consulta
    """
    print(f"\n🔍 Buscando: '{query}'")
    
    # Generar embedding simulado para la consulta
    query_embedding = generate_simulated_embedding(query)
    
    # Calcular similitudes
    similarities = []
    for i, emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))
    
    # Ordenar por similitud descendente
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Mostrar resultados
    print(f"\n📊 Top {top_k} resultados:")
    for i in range(min(top_k, len(similarities))):
        idx, sim = similarities[i]
        preview = chunks[idx][:150].replace('\n', ' ') + "..."
        print(f"\n   {i+1}. Similitud: {sim:.4f}")
        print(f"      {preview}")
    
    return similarities[:top_k]

# ============================================
# 5. FUNCIÓN RAG CON EMBEDDINGS SIMULADOS
# ============================================

def answer_with_rag(query, chunks, chunk_embeddings, top_k=3):
    """
    Responde una pregunta usando los chunks más relevantes
    """
    print(f"\n📌 Pregunta: {query}")
    print("=" * 60)
    
    # Buscar chunks relevantes
    results = search_semantic(query, chunks, chunk_embeddings, top_k)
    
    if not results:
        return "No se encontraron resultados relevantes."
    
    # Construir contexto con los chunks más relevantes
    context = "\n\n---\n\n".join([chunks[idx] for idx, _ in results])
    
    prompt = f"""Eres un asistente experto en análisis de documentos.
Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA: {query}

INSTRUCCIONES:
- Usa solo información del contexto
- Si la respuesta no está en el contexto, di que no tienes esa información
- Sé conciso pero completo

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
# 6. FUNCIÓN PARA GUARDAR/CARGAR EMBEDDINGS
# ============================================

def save_embeddings(embeddings, chunks, filename="embeddings_simulados.json"):
    """Guarda embeddings y chunks en un archivo"""
    data = {
        "chunks": chunks,
        "embeddings": embeddings
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Embeddings guardados en {filename}")

def load_embeddings(filename="embeddings_simulados.json"):
    """Carga embeddings y chunks desde un archivo"""
    with open(filename, "r") as f:
        data = json.load(f)
    print(f"✅ Embeddings cargados de {filename}")
    return data["chunks"], data["embeddings"]

# ============================================
# 7. DEMOSTRACIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("🚀 SISTEMA RAG CON EMBEDDINGS SIMULADOS")
    print("=" * 60)
    print("📌 NOTA: Esta versión NO usa VoyageAI")
    print("   Los embeddings son simulados (basados en hash)")
    print("   Perfecto para pruebas sin rate limits")
    
    # 1. Cargar documento
    try:
        with open("report.md", "r", encoding="utf-8") as f:
            text = f.read()
        print(f"\n📄 Documento cargado: report.md")
        print(f"   Longitud: {len(text)} caracteres")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo report.md")
        exit(1)
    
    # 2. Chunking
    print("\n📦 Generando chunks...")
    chunks = chunk_by_section(text)
    print(f"   {len(chunks)} chunks creados")
    
    # 3. Verificar si ya existen embeddings guardados
    embeddings_file = "embeddings_simulados.json"
    
    if os.path.exists(embeddings_file):
        print(f"\n📂 Cargando embeddings existentes...")
        chunks, embeddings = load_embeddings(embeddings_file)
    else:
        # 4. Generar embeddings simulados (sin rate limits)
        print(f"\n🆕 Generando embeddings SIMULADOS...")
        embeddings = generate_embeddings_for_chunks(chunks)
        save_embeddings(embeddings, chunks, embeddings_file)
    
    # 5. Probar búsqueda semántica
    print("\n" + "=" * 60)
    print("🔍 PRUEBAS DE BÚSQUEDA SEMÁNTICA")
    print("=" * 60)
    
    preguntas = [
        "¿Qué es el síndrome XDR-471?",
        "Errores de software en Project Phoenix",
        "Especificaciones del Modelo Zircon-5",
        "Incidente de ciberseguridad Q4 2023",
        "Caso Synergy Dynamics"
    ]
    
    for i, pregunta in enumerate(preguntas, 1):
        print(f"\n{'='*60}")
        print(f"📝 PREGUNTA {i}")
        print(f"{'='*60}")
        
        respuesta = answer_with_rag(pregunta, chunks, embeddings)
        print(f"\n✅ Respuesta:\n{respuesta}\n")
        
        if i < len(preguntas):
            input("Presiona Enter para continuar...")
    
    print("\n" + "=" * 60)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("\n📌 VENTAJAS DE ESTA VERSIÓN:")
    print("   ✅ Sin rate limits")
    print("   ✅ Sin necesidad de API key de VoyageAI")
    print("   ✅ Todos los chunks procesados")
    print("   ✅ Ideal para pruebas y desarrollo")
