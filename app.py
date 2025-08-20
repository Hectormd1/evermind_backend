#!/usr/bin/env python3
"""
Punto de entrada para Hugging Face Spaces
Solo importa la aplicación, no la ejecuta
"""

# Importar la aplicación desde main.py
from main import app

# HF Spaces ejecutará automáticamente la app
# No necesitamos uvicorn.run() aquí
