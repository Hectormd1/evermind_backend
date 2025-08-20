#!/usr/bin/env python3
"""
Punto de entrada para Hugging Face Spaces
Este archivo simplemente importa y ejecuta main.py
"""

# Importar la aplicación desde main.py
from main import app, demo

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Puerto para Hugging Face Spaces
    port = int(os.environ.get("PORT", 7860))
    
    print(f"🚀 Iniciando Evermind Backend en puerto {port}")
    print("🌐 Interfaz Gradio disponible en: /")
    print("📡 API FastAPI disponible en: /docs")
    
    # Ejecutar la aplicación
    uvicorn.run(app, host="0.0.0.0", port=port)
