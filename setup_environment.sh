#!/bin/bash
# Script de instalación completa para el entorno de Groq
# Ejecutar UNA SOLA VEZ para configurar todo

echo "🚀 CONFIGURANDO ENTORNO COMPLETO PARA GROQ"
echo "=========================================="

# 1. Verificar si el entorno virtual existe
if [ ! -d "anthropic-env" ]; then
    echo "📁 Creando entorno virtual..."
    python3 -m venv anthropic-env
else
    echo "✅ Entorno virtual ya existe"
fi

# 2. Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source anthropic-env/bin/activate

# 3. Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

# 4. Instalar TODAS las dependencias necesarias
echo "📚 Instalando paquetes necesarios..."
pip install \
    openai \
    python-dotenv \
    chromadb \
    sentence-transformers \
    requests \
    numpy \
    pandas \
    jupyter \
    ipykernel

# 5. Verificar instalaciones
echo "✅ Verificando instalaciones..."
pip list | grep -E "openai|dotenv|chroma|sentence|requests"

# 6. Crear archivo .env si no existe
if [ ! -f ".env" ]; then
    echo "📝 Creando archivo .env (debes editar con tu API key)"
    echo "# GROQ API KEY" > .env
    echo "GROQ_API_KEY=\"tu-api-key-aqui\"" >> .env
    echo "⚠️  No olvides editar .env con tu API key de Groq"
else
    echo "✅ Archivo .env ya existe"
fi

# 7. Crear archivo .gitignore si no existe
if [ ! -f ".gitignore" ]; then
    echo "📝 Creando .gitignore..."
    echo ".env" > .gitignore
    echo "anthropic-env/" >> .gitignore
    echo "__pycache__/" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo ".ipynb_checkpoints/" >> .gitignore
    echo "test_files/" >> .gitignore
    echo ".backups/" >> .gitignore
else
    echo "✅ .gitignore ya existe"
fi

# 8. Crear directorios necesarios
echo "📁 Creando directorios de trabajo..."
mkdir -p test_files
mkdir -p .backups
mkdir -p notebooks

# 9. Mensaje final
echo ""
echo "🎉 CONFIGURACIÓN COMPLETADA"
echo "=========================================="
echo "✅ Entorno listo para usar Groq"
echo ""
echo "📌 Próximos pasos:"
echo "1. Edita el archivo .env con tu API key de Groq"
echo "2. Activa el entorno: source anthropic-env/bin/activate"
echo "3. Ejecuta cualquier script: python groq_rag_chunking.py"
echo ""
echo "📁 Directorios creados:"
echo "   - test_files/: Para archivos de prueba"
echo "   - .backups/: Backups automáticos"
echo "   - notebooks/: Para Jupyter notebooks"
echo ""
echo "🚀 ¡Todo listo para trabajar!"
