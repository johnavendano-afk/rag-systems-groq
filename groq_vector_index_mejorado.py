"""
Sistema RAG con VectorIndex + Sentence Transformers
Embeddings REALES pero locales (sin rate limits, sin costo)
"""

import os
import re
import json
import math
import numpy as np
from typing import Optional, Any, List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

# ============================================
# 1. INSTALAR DEPENDENCIAS (ejecutar primero)
# ============================================
"""
pip install sentence-transformers
"""

# ============================================
# 2. CONFIGURACIÓN GROQ
# ============================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"

# ============================================
# 3. EMBEDDINGS CON SENTENCE TRANSFORMERS
# ============================================

class EmbeddingGenerator:
    """
    Generador de embeddings usando modelos locales
    Sin rate limits, sin costo, alta calidad
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Modelos recomendados:
        - "all-MiniLM-L6-v2": Rápido, 384 dims, buena calidad
        - "all-mpnet-base-v2": Mejor calidad, 768 dims, más lento
        - "paraphrase-multilingual-MiniLM-L12-v2": Multilingüe
        """
        print(f"🔄 Cargando modelo de embeddings: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Modelo cargado. Dimensión: {self.dimension}")
    
    def encode(self, texts, batch_size=32):
        """
        Genera embeddings para uno o múltiples textos
        """
        if isinstance(texts, str):
            return self.model.encode(texts).tolist()
        else:
            return self.model.encode(texts, batch_size=batch_size).tolist()

# ============================================
# 4. VECTORINDEX (MEJORADO)
# ============================================

class VectorIndex:
    """
    Implementación de un índice vectorial en memoria
    Versión mejorada con normalización automática
    """
    
    def __init__(
        self,
        distance_metric: str = "cosine",
        embedding_generator=None,
        normalize_vectors: bool = True
    ):
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self._vector_dim: Optional[int] = None
        
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError("distance_metric must be 'cosine' or 'euclidean'")
        self._distance_metric = distance_metric
        self._embedding_generator = embedding_generator
        self._normalize_vectors = normalize_vectors

    def _normalize(self, vector):
        """Normaliza un vector (unitario)"""
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            return [x/norm for x in vector]
        return vector

    def add_document(self, document: Dict[str, Any]):
        """
        Añade un documento al índice (genera embedding automáticamente)
        """
        if not self._embedding_generator:
            raise ValueError("Embedding generator not provided")
            
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document dictionary must contain a 'content' key.")

        content = document["content"]
        if not isinstance(content, str):
            raise TypeError("Document 'content' must be a string.")

        # Generar embedding
        vector = self._embedding_generator.encode(content)
        
        # Normalizar si es necesario
        if self._normalize_vectors and self._distance_metric == "cosine":
            vector = self._normalize(vector)
            
        self.add_vector(vector=vector, document=document)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Añade múltiples documentos al índice
        """
        print(f"\n📦 Añadiendo {len(documents)} documentos al índice...")
        
        # Extraer solo los contenidos
        contents = [doc["content"] for doc in documents]
        
        # Generar embeddings en batch (mucho más rápido)
        print(f"   Generando embeddings en batch...")
        vectors = self._embedding_generator.encode(contents)
        
        # Añadir cada documento con su vector
        for i, (vector, doc) in enumerate(zip(vectors, documents)):
            # Normalizar si es necesario
            if self._normalize_vectors and self._distance_metric == "cosine":
                vector = self._normalize(vector)
            
            self.add_vector(vector=vector, document=doc)
            print(f"   📍 Documento {i+1}/{len(documents)} añadido", end="\r")
        
        print(f"\n✅ Índice creado con {len(self)} documentos")

    def search(
        self, query: Any, k: int = 1
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Busca los k documentos más similares a la query
        """
        if not self.vectors:
            return []

        # Obtener vector de la query
        if isinstance(query, str):
            if not self._embedding_generator:
                raise ValueError("Embedding generator required for string query")
            query_vector = self._embedding_generator.encode(query)
            if self._normalize_vectors and self._distance_metric == "cosine":
                query_vector = self._normalize(query_vector)
        elif isinstance(query, list):
            query_vector = query
        else:
            raise TypeError("Query must be string or list of numbers")

        # Verificar dimensiones
        if self._vector_dim is None:
            return []
        if len(query_vector) != self._vector_dim:
            raise ValueError(f"Dimension mismatch. Expected {self._vector_dim}, got {len(query_vector)}")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        # Calcular similitudes
        similarities = []
        for i, stored_vector in enumerate(self.vectors):
            if self._distance_metric == "cosine":
                # Para vectores normalizados, el producto punto = similitud coseno
                sim = sum(p*q for p, q in zip(query_vector, stored_vector))
                # Convertimos a distancia (1 - similitud)
                dist = 1.0 - max(-1.0, min(1.0, sim))
            else:
                # Distancia euclidiana
                dist = math.sqrt(sum((p-q)**2 for p, q in zip(query_vector, stored_vector)))
            
            similarities.append((dist, self.documents[i]))

        # Ordenar por menor distancia
        similarities.sort(key=lambda item: item[0])

        return [(doc, dist) for dist, doc in similarities[:k]]

    def add_vector(self, vector, document: Dict[str, Any]):
        """
        Añade un vector directamente al índice
        """
        if not isinstance(vector, list) or not all(isinstance(x, (int, float)) for x in vector):
            raise TypeError("Vector must be a list of numbers.")
        if not isinstance(document, dict):
            raise TypeError("Document must be a dictionary.")
        if "content" not in document:
            raise ValueError("Document must contain 'content' key.")

        # Verificar consistencia de dimensiones
        if not self.vectors:
            self._vector_dim = len(vector)
        elif len(vector) != self._vector_dim:
            raise ValueError(f"Inconsistent dimension. Expected {self._vector_dim}, got {len(vector)}")

        self.vectors.append(vector)
        self.documents.append(document)

    def __len__(self) -> int:
        return len(self.vectors)

    def __repr__(self) -> str:
        return f"VectorIndex(count={len(self)}, dim={self._vector_dim}, metric='{self._distance_metric}')"

# ============================================
# 5. FUNCIONES DE CHUNKING
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
# 6. FUNCIÓN RAG CON VECTORINDEX MEJORADO
# ============================================

def answer_with_rag(query, vector_index, k=2):
    """
    Responde una pregunta usando el VectorIndex
    """
    print(f"\n📌 Pregunta: {query}")
    print("=" * 60)
    
    # Buscar chunks relevantes
    results = vector_index.search(query, k=k)
    
    if not results:
        return "No se encontraron resultados relevantes."
    
    # Mostrar resultados de búsqueda
    print(f"\n📊 Top {k} resultados (distancia):")
    for i, (doc, dist) in enumerate(results, 1):
        preview = doc["content"][:150].replace('\n', ' ') + "..."
        similitud = 1 - dist  # Convertir distancia a similitud
        print(f"\n   {i}. Similitud: {similitud:.4f} (distancia: {dist:.4f})")
        print(f"      {preview}")
    
    # Construir contexto
    context = "\n\n---\n\n".join([doc["content"] for doc, _ in results])
    
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
# 7. DEMOSTRACIÓN PRINCIPAL
# ============================================

if __name__ == "__main__":
    print("🚀 SISTEMA RAG CON EMBEDDINGS REALES (LOCALES)")
    print("=" * 60)
    
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
    
    # 3. Inicializar generador de embeddings
    print("\n🤖 Inicializando generador de embeddings...")
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    # 4. Crear VectorIndex
    index_file = "vector_index_reales.json"
    
    if os.path.exists(index_file):
        print(f"\n📂 Cargando índice existente...")
        # Nota: La carga desde archivo requeriría serialización adicional
        # Por simplicidad, regeneramos
        print("   Regenerando índice con nuevos embeddings...")
        index = VectorIndex(
            distance_metric="cosine",
            embedding_generator=embedder,
            normalize_vectors=True
        )
        documents = [{"content": chunk} for chunk in chunks]
        index.add_documents(documents)
    else:
        print(f"\n🆕 Creando nuevo VectorIndex con embeddings REALES...")
        index = VectorIndex(
            distance_metric="cosine",
            embedding_generator=embedder,
            normalize_vectors=True
        )
        
        # Preparar documentos
        documents = [{"content": chunk} for chunk in chunks]
        
        # Añadir documentos al índice
        index.add_documents(documents)
    
    print(f"\n📊 Estadísticas del índice:")
    print(f"   • Documentos: {len(index)}")
    print(f"   • Dimensión: {index._vector_dim}")
    print(f"   • Métrica: {index._distance_metric}")
    
    # 5. Probar búsquedas
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
        
        respuesta = answer_with_rag(pregunta, index, k=2)
        print(f"\n✅ Respuesta:\n{respuesta}\n")
        
        if i < len(preguntas):
            input("Presiona Enter para continuar...")
    
    print("\n" + "=" * 60)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("\n📌 CARACTERÍSTICAS:")
    print("   ✅ Embeddings REALES con sentence-transformers")
    print("   ✅ Sin rate limits ni costos")
    print("   ✅ Modelo: all-MiniLM-L6-v2 (384 dimensiones)")
    print("   ✅ Normalización automática para similitud coseno")
