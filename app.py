#!/usr/bin/env python3
"""
Punto de entrada para Hugging Face Spaces
"""

print("DEBUG: Iniciando app.py para HF Spaces")

# Importar la aplicación directamente desde main.py
from main import app as fastapi_app

print(f"DEBUG: App importada: {type(fastapi_app)}")
print(f"DEBUG: App routes: {len(fastapi_app.routes)}")

# HF Spaces necesita que la variable se llame exactamente 'app'
app = fastapi_app

print("DEBUG: Variable 'app' asignada para HF Spaces")
print(f"DEBUG: App final: {app}")

if __name__ == "__main__":
    import uvicorn
    print("DEBUG: Ejecutando uvicorn desde app.py")
    uvicorn.run(app, host="0.0.0.0", port=7860)
