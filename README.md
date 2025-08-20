---
title: Evermind Backend
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.43.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: evermind-backend
---

# 🧠 Evermind AI Backend

Backend completo para la aplicación Evermind que combina transcripción de audio y chat con IA.

## ✨ Características

- 🎤 **Transcripción de Audio**: Convierte audio a texto usando Whisper AI
- 💬 **Chat con IA**: Conversación inteligente con múltiples proveedores
- 🔄 **APIs RESTful**: Endpoints para integración con aplicaciones móviles
- 🌐 **Interfaz Web**: Panel de control con Gradio

## 🚀 APIs Disponibles

### Transcripción
- `POST /transcribe` - Transcribir archivo de audio

### Chat
- `POST /chat` - Enviar mensaje al chat de IA

### Sistema
- `GET /ping` - Verificar estado del servidor
- `HEAD /ping` - Keep-alive para workers

## 🔧 Uso

### Interfaz Web
Usa la interfaz Gradio en esta página para probar:
- Subir archivos de audio para transcripción
- Chatear con la IA
- Verificar estado del sistema

### API REST
```python
import requests

# Transcripción
files = {"file": open("audio.wav", "rb")}
response = requests.post("https://hectormd1-evermind-backend.hf.space/transcribe", files=files)

# Chat
data = {"messages": [{"role": "user", "content": "Hola"}]}
response = requests.post("https://hectormd1-evermind-backend.hf.space/chat", json=data)
```

## 🏗️ Tecnologías

- **FastAPI**: Framework web moderno
- **Whisper AI**: Transcripción de audio
- **Gradio**: Interfaz web interactiva
- **Multiple AI Providers**: Together AI, Groq, OpenRouter

## 🔑 Variables de Entorno

Configura estas variables en Settings > Variables and secrets:

- `TOGETHER_API_KEY`: API key para Together AI
- `GROQ_API_KEY`: API key para Groq
- `OPENROUTER_API_KEY`: API key para OpenRouter

---

Desarrollado para Evermind - Tu compañero de reflexión personal 🌟
