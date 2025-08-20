#!/usr/bin/env python3
"""
Punto de entrada para Hugging Face Spaces
"""

try:
    # Intentar importar con Gradio (en HF Spaces)
    from main import app
    print("✅ Aplicación importada exitosamente")
except Exception as e:
    print(f"❌ Error importando aplicación: {e}")
    # Crear una app básica de fallback
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/")
    def fallback():
        return {"error": "Error importando la aplicación principal", "details": str(e)}

# HF Spaces ejecutará automáticamente la app
