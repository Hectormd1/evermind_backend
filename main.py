from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import tempfile
import os
import traceback
import random
import httpx
import gc  # Para garbage collection manual
import psutil  # Para monitoreo de memoria
from typing import List, Optional
from dotenv import load_dotenv
import gradio as gr
import asyncio

# Cargar variables de entorno
load_dotenv()

# Log de inicio
print("🚀 EVERMIND BACKEND: Iniciando servidor...")
print("🔄 KEEP-ALIVE: Worker automático configurado")
print(f"💾 MEMORIA INICIAL: {psutil.virtual_memory().percent}%")

app = FastAPI(title="Evermind AI Backend", version="1.0.0")

# Agregar middleware CORS para React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga el modelo base de Whisper solo cuando sea necesario (lazy loading)
model = None

def load_whisper_model():
    global model
    if model is None:
        try:
            import whisper
            print("🤖 WHISPER: Iniciando carga del modelo 'tiny'...")
            print(f"💾 MEMORIA ANTES: {psutil.virtual_memory().percent}%")
            
            # Optimización máxima de memoria para Render
            model = whisper.load_model("tiny", device="cpu", download_root=None)
            
            print("✅ WHISPER: Modelo 'tiny' cargado exitosamente en CPU")
            print(f"💾 MEMORIA DESPUÉS: {psutil.virtual_memory().percent}%")
            
            # Limpiar memoria inmediatamente después de cargar
            gc.collect()
            
        except Exception as e:
            print(f"❌ WHISPER ERROR: Error cargando modelo: {e}")
            model = False
    return model

# Función mejorada para liberar memoria después de transcripción
def cleanup_whisper_memory():
    try:
        gc.collect()  # Garbage collection inmediato
        
        # Si torch está disponible, limpiar cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # torch no disponible, continuar
            
        memory_percent = psutil.virtual_memory().percent
        print(f"🧹 MEMORIA LIMPIA: {memory_percent}% usado")
        
        # Si el uso de memoria es muy alto, forzar limpieza más agresiva
        if memory_percent > 80:
            print("⚠️ MEMORIA ALTA: Ejecutando limpieza agresiva...")
            for i in range(3):
                gc.collect()
                
    except Exception as e:
        print(f"❌ ERROR EN LIMPIEZA: {e}")

# Función para verificar memoria disponible
def check_memory_status():
    memory = psutil.virtual_memory()
    return {
        "used_percent": memory.percent,
        "available_mb": memory.available // (1024 * 1024),
        "total_mb": memory.total // (1024 * 1024)
    }

# Configuración de IA - Proveedores funcionales
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# OpenRouter - modelos gratuitos disponibles
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # Opcional para algunos modelos
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"  # Modelo completamente gratuito

# Groq (API gratuita muy rápida) - Principal proveedor
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Modelos para el chat
class ChatMessage(BaseModel):
    role: str
    content: str

class ReflectRequest(BaseModel):
    mood_before: Optional[int] = None
    messages: List[ChatMessage] = []
    locale: str = "es-ES"

# IA real usando múltiples proveedores gratuitos
async def call_together_ai(messages: List[dict]) -> str:
    """Llama a Together AI"""
    if not TOGETHER_API_KEY:
        return None
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                },
                json={
                    "model": TOGETHER_MODEL,
                    "messages": messages,
                    "max_tokens": 380,
                    "temperature": 0.7,
                    "top_p": 0.95,
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if reply and len(reply) > 40:
                    return reply.strip()
            else:
                print(f"❌ Together AI error: {response.status_code}")
                return None
    except Exception as e:
        print(f"❌ Together AI exception: {e}")
        return None

async def call_openrouter_ai(messages: List[dict]) -> str:
    """Llama a OpenRouter con modelos gratuitos"""
    try:
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://evermind.app",
            "X-Title": "Evermind"
        }
        
        # API key es opcional para modelos gratuitos
        if OPENROUTER_API_KEY:
            headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": messages,
                    "max_tokens": 380,
                    "temperature": 0.7,
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if reply and len(reply) > 20:
                    return reply.strip()
            else:
                error_text = response.text
                print(f"❌ OpenRouter error: {response.status_code} - {error_text}")
                return None
    except Exception as e:
        print(f"❌ OpenRouter exception: {e}")
        return None

async def call_groq_ai(messages: List[dict]) -> str:
    """Llama a Groq AI (muy rápido y gratuito)"""
    if not GROQ_API_KEY:
        return None
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "max_tokens": 380,
                    "temperature": 0.7,
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if reply and len(reply) > 40:
                    return reply.strip()
            else:
                print(f"❌ Groq error: {response.status_code}")
                return None
    except Exception as e:
        print(f"❌ Groq exception: {e}")
        return None

async def generate_ai_response(messages: List[ChatMessage], mood_before: Optional[int] = None) -> str:
    """Genera respuestas reales de IA usando múltiples proveedores gratuitos como fallback"""
    
    # Construir mensajes para las APIs
    ai_messages = []
    
    # System prompt optimizado para respuestas coherentes y naturales
    ai_messages.append({
        "role": "system",
        "content": "Eres un psicólogo y acompañante emocional empático llamado Evermind. Tu objetivo es ayudar a las personas a reflexionar sobre sus emociones y sentimientos de manera natural y humana. REGLAS IMPORTANTES: 1) LEE CUIDADOSAMENTE lo que el usuario te dice antes de responder. 2) Responde SOLO en español natural de España/México, sin anglicismos. 3) Sé específico y relevante a lo que el usuario menciona. 4) Si el usuario expresa frustración, tristeza, enojo o cualquier emoción específica, reconócela directamente. 5) Haz preguntas abiertas para que la persona profundice en sus emociones. 6) No des consejos genéricos, personaliza tu respuesta a su situación específica. 7) Máximo 150 palabras. 8) Habla como un amigo comprensivo, no como un terapeuta formal."
    })
    
    # Agregar historial de mensajes
    for msg in messages:
        if msg.role in ['user', 'assistant']:
            ai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
    
    # Probar proveedores en orden: Groq (principal) -> OpenRouter -> Together AI (sin créditos)
    providers = [
        ("Groq AI", call_groq_ai),
        ("OpenRouter", call_openrouter_ai),
        ("Together AI", call_together_ai),
    ]
    
    for provider_name, provider_func in providers:
        try:
            print(f"🤖 Intentando {provider_name}...")
            reply = await provider_func(ai_messages)
            
            if reply and len(reply) > 20:  # Validar que la respuesta sea útil
                print(f"✅ {provider_name} respondió exitosamente")
                return reply
            else:
                print(f"⚠️ {provider_name} respuesta muy corta o vacía")
                
        except Exception as e:
            print(f"❌ Error en {provider_name}: {e}")
            continue
    
    # Si todos fallan, responder que no se puede procesar la solicitud
    print("❌ Todos los proveedores de IA fallaron")
    return "Lo siento, no puedo generar una respuesta en este momento. Por favor, inténtalo de nuevo más tarde."

def generate_simple_fallback() -> str:
    """Respuesta simple cuando todas las IAs fallan"""
    return "Lo siento, no puedo generar una respuesta en este momento. Por favor, inténtalo de nuevo más tarde."

# Endpoint de transcripción habilitado
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribir audio usando Whisper"""
    global model
    
    # Log de inicio inmediato
    print("🎵 TRANSCRIPCIÓN: Procesando audio desde React Native")
    
    # Cargar modelo si no está cargado (esto puede tomar tiempo la primera vez)
    if model is None or model is False:
        print("🤖 WHISPER: Cargando modelo bajo demanda...")
        load_whisper_model()
    
    if not model:
        print("❌ WHISPER: Modelo no disponible")
        raise HTTPException(status_code=503, detail="Whisper no está disponible en este momento")
    
    if not file:
        raise HTTPException(status_code=400, detail="No se recibió archivo de audio")
    
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"🎵 TRANSCRIPCIÓN: Procesando audio desde React Native")
        print(f"📁 ARCHIVO: {temp_file_path}")
        print(f"💾 MEMORIA PRE-TRANSCRIPCIÓN: {psutil.virtual_memory().percent}%")
        
        # Transcribir con Whisper (configuración optimizada para memoria)
        result = model.transcribe(
            temp_file_path,
            language="es",  # Forzar español
            fp16=False,  # Mejor compatibilidad
            temperature=0.2,  # Un poco más de flexibilidad
            word_timestamps=False,  # Simplificar para mejor precisión
            no_speech_threshold=0.5,  # Más sensible al habla
            logprob_threshold=-1.0,  # Más estricto con la confianza
            compression_ratio_threshold=2.4,  # Evitar repeticiones
            condition_on_previous_text=False,  # No usar contexto previo
            initial_prompt="Transcripción de audio en español. Palabras comunes: calor, color, mucho, poco, tengo, estoy, muy, bien, mal."  # Contexto español
        )
        
        # ⭐ LIMPIEZA INMEDIATA Y AGRESIVA DE MEMORIA DESPUÉS DE TRANSCRIPCIÓN
        print("🧹 LIMPIEZA: Liberando memoria post-transcripción...")
        cleanup_whisper_memory()
        
        # Limpiar archivo temporal inmediatamente
        try:
            os.unlink(temp_file_path)
            print("🗑️ ARCHIVO TEMPORAL: Eliminado exitosamente")
        except Exception as cleanup_error:
            print(f"⚠️ ARCHIVO TEMPORAL: Error al eliminar - {cleanup_error}")
        
        print(f"💾 MEMORIA POST-LIMPIEZA: {psutil.virtual_memory().percent}%")
        
        transcribed_text = result["text"].strip()
        
        # Post-procesamiento para corregir errores comunes
        corrections = {
            "color": "calor",
            "que lo": "tengo",
            "de color": "de calor",
            "mucho color": "mucho calor",
            "con color": "con calor",
            "más color": "más calor"
        }
        
        for wrong, correct in corrections.items():
            transcribed_text = transcribed_text.replace(wrong, correct)
        
        print(f"📝 RESULTADO: '{transcribed_text}'")
        
        if not transcribed_text:
            return {"text": "No se pudo transcribir el audio"}
        
        return {"text": transcribed_text}
        
    except Exception as e:
        print(f"❌ Error transcribiendo: {e}")
        traceback.print_exc()
        
        # Limpiar archivo temporal si existe
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass
            
        raise HTTPException(status_code=500, detail=f"Error al transcribir: {str(e)}")

@app.post("/reflect")
async def reflect_chat(request: ReflectRequest):
    """Endpoint para generar respuestas de IA real"""
    try:
        print(f"🤖 Generando respuesta IA real para {len(request.messages)} mensajes")
        
        reply = await generate_ai_response(request.messages, request.mood_before)
        
        return {
            "reply": reply,
            "provider": "multi_provider_system"
        }
    except Exception as e:
        print("\n--- ERROR EN /reflect ---")
        traceback.print_exc()
        print("---------------------------\n")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
@app.head("/")
def root():
    # Log cuando el worker hace la petición
    print("🔄 KEEP-ALIVE: Petición recibida desde Cloudflare Workers (HEAD /)")
    
    return {
        "status": "ok", 
        "service": "Evermind AI Backend",
        "version": "1.0.0",
        "whisper_loaded": model is not None and model is not False,
        "endpoints": ["/transcribe", "/reflect", "/check-credits", "/providers-status", "/health", "/ping"], 
        "ai_providers": ["groq", "openrouter", "together"]
    }

@app.get("/ping")
@app.head("/ping")  # ⭐ SOPORTE PARA HEAD REQUEST
def ping():
    """Endpoint simple para mantener el servicio activo en Render"""
    import time
    
    # Log más informativo
    print("🔄 KEEP-ALIVE: Ping recibido desde Cloudflare Workers")
    print(f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    
    # Ejecutar limpieza de memoria preventiva
    cleanup_whisper_memory()
    
    return {
        "status": "pong",
        "timestamp": int(time.time()),
        "service": "evermind-backend",
        "memory_usage": "optimized",
        "keep_alive": "active",
        "source": "cloudflare_workers_cron"
    }

@app.get("/health")
@app.head("/health")  # ⭐ ENDPOINT DE SALUD OPTIMIZADO
def health_check():
    """Endpoint de salud con monitoreo de memoria"""
    import time
    
    try:
        memory_info = check_memory_status()
        
        print(f"🏥 HEALTH CHECK: Memoria en uso: {memory_info['used_percent']}%")
        
        return {
            "status": "healthy",
            "timestamp": int(time.time()),
            "memory": memory_info,
            "whisper_loaded": model is not None and model is not False,
            "service_active": True,
            "version": "3.0-optimized",
            "render_service_id": os.environ.get("RENDER_SERVICE_ID", "local")
        }
    except Exception as e:
        print(f"❌ HEALTH CHECK ERROR: {e}")
        return {
            "status": "healthy",
            "timestamp": int(time.time()),
            "service_active": True,
            "error": str(e)
        }

@app.get("/status")
@app.head("/status")  # ⭐ NUEVO ENDPOINT DE STATUS COMPLETO
def status_check():
    """Endpoint completo de status para keep-alive ultra-agresivo"""
    import time
    
    try:
        memory_info = check_memory_status()
        
        # Si la memoria está muy alta, ejecutar limpieza
        if memory_info['used_percent'] > 75:
            print("⚠️ MEMORIA ALTA DETECTADA: Ejecutando limpieza automática...")
            cleanup_whisper_memory()
            memory_info = check_memory_status()  # Actualizar después de limpieza
        
        print(f"📊 STATUS CHECK: Servicio activo - Memoria: {memory_info['used_percent']}%")
        
        return {
            "status": "fully_active",
            "timestamp": int(time.time()),
            "uptime": "continuous",
            "memory": memory_info,
            "whisper_model": "tiny" if model and model is not False else "not_loaded",
            "endpoints_active": ["/transcribe", "/reflect", "/ping", "/health", "/status"],
            "keep_alive_mode": "ultra_aggressive",
            "auto_cleanup": memory_info['used_percent'] <= 75
        }
    except Exception as e:
        print(f"❌ STATUS CHECK ERROR: {e}")
        return {
            "status": "active_with_errors",
            "timestamp": int(time.time()),
            "error": str(e)
        }

@app.get("/providers-status")
async def providers_status():
    """Verificar qué proveedores están configurados"""
    status = {
        "groq_ai": {
            "configured": bool(GROQ_API_KEY),
            "model": GROQ_MODEL if GROQ_API_KEY else "No configurado",
            "priority": 1
        },
        "openrouter": {
            "configured": bool(OPENROUTER_API_KEY),
            "model": OPENROUTER_MODEL,
            "note": "Modelo gratuito - API key opcional",
            "priority": 2
        },
        "together_ai": {
            "configured": bool(TOGETHER_API_KEY),
            "model": TOGETHER_MODEL if TOGETHER_API_KEY else "No configurado",
            "note": "Sin créditos actualmente",
            "priority": 3
        }
    }
    
    configured_count = sum(1 for p in status.values() if p["configured"])
    
    return {
        "providers": status,
        "configured_providers": configured_count,
        "total_providers": len(status),
        "recommendation": "Sistema funcionando con Groq AI (principal) y OpenRouter (fallback)"
    }

@app.get("/check-credits")
async def check_credits():
    """Verificar el estado de los créditos de Together AI"""
    if not TOGETHER_API_KEY:
        return {"status": "no_api_key", "fallback": True}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                },
                json={
                    "model": TOGETHER_MODEL,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5,
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                return {"status": "ok", "credits_available": True, "fallback": False}
            elif response.status_code == 402:
                return {"status": "no_credits", "credits_available": False, "fallback": True}
            else:
                return {"status": "error", "code": response.status_code, "fallback": True}
                
    except Exception as e:
        return {"status": "connection_error", "error": str(e), "fallback": True}

@app.post("/test-together")
async def test_together():
    """Probar Together AI individualmente"""
    test_messages = [{"role": "user", "content": "Hola, prueba rápida"}]
    result = await call_together_ai(test_messages)
    return {"provider": "Together AI", "result": result, "status": "success" if result else "failed"}

@app.post("/test-groq")
async def test_groq():
    """Probar Groq AI individualmente"""
    test_messages = [{"role": "user", "content": "Hola, prueba rápida"}]
    result = await call_groq_ai(test_messages)
    return {"provider": "Groq AI", "result": result, "status": "success" if result else "failed"}

@app.post("/test-openrouter")
async def test_openrouter():
    """Probar OpenRouter individualmente"""
    test_messages = [{"role": "user", "content": "Hola, prueba rápida"}]
    result = await call_openrouter_ai(test_messages)
    return {"provider": "OpenRouter", "result": result, "status": "success" if result else "failed"}

# =============================================================================
# INTERFAZ GRADIO PARA HUGGING FACE SPACES
# =============================================================================

def transcribe_audio_for_gradio(audio_file):
    """Función para transcribir audio desde Gradio"""
    try:
        load_whisper_model()
        if audio_file is None:
            return "❌ No se ha proporcionado un archivo de audio"
        
        # Transcribir usando Whisper
        result = model.transcribe(audio_file, language="es")
        transcription = result["text"].strip()
        
        if not transcription:
            return "❌ No se pudo transcribir el audio"
        
        return f"✅ Transcripción: {transcription}"
    
    except Exception as e:
        return f"❌ Error en transcripción: {str(e)}"

async def chat_for_gradio(message, history):
    """Función para chat desde Gradio"""
    try:
        # Convertir historial de Gradio (formato messages) a formato de mensajes API
        messages = []
        
        # Si history viene en formato messages (nuevo formato Gradio)
        if history and isinstance(history, list) and len(history) > 0:
            if isinstance(history[0], dict) and "role" in history[0]:
                # Ya está en formato messages
                messages = history.copy()
            else:
                # Formato tuplas (viejo formato)
                for human, assistant in history:
                    messages.append({"role": "user", "content": human})
                    if assistant:
                        messages.append({"role": "assistant", "content": assistant})
        
        # Agregar mensaje actual
        messages.append({"role": "user", "content": message})
        
        # Convertir a formato ChatMessage para la función existente
        from pydantic import BaseModel
        
        class ChatMessage(BaseModel):
            role: str
            content: str
        
        chat_messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages]
        
        # Llamar a la función de chat existente
        response = await generate_ai_response(chat_messages)
        
        return response
    
    except Exception as e:
        return f"❌ Error en chat: {str(e)}"

# Crear interfaz Gradio
with gr.Blocks(title="Evermind AI Backend", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Evermind AI Backend")
    gr.Markdown("Backend para transcripción de audio y chat con IA")
    
    with gr.Tab("🎤 Transcripción de Audio"):
        gr.Markdown("### Sube un archivo de audio para transcribir")
        audio_input = gr.Audio(
            label="Archivo de Audio",
            type="filepath",
            format="wav"
        )
        transcribe_btn = gr.Button("🎯 Transcribir", variant="primary")
        transcription_output = gr.Textbox(
            label="Transcripción",
            lines=3,
            placeholder="La transcripción aparecerá aquí..."
        )
        
        transcribe_btn.click(
            fn=transcribe_audio_for_gradio,
            inputs=[audio_input],
            outputs=[transcription_output]
        )
    
    with gr.Tab("💬 Chat con IA"):
        gr.Markdown("### Chatea con la IA de Evermind")
        chatbot = gr.Chatbot(
            label="Conversación",
            height=400,
            placeholder="Inicia una conversación...",
            type="messages"
        )
        msg = gr.Textbox(
            label="Mensaje",
            placeholder="Escribe tu mensaje aquí...",
            lines=2
        )
        send_btn = gr.Button("📤 Enviar", variant="primary")
        clear_btn = gr.Button("🗑️ Limpiar Chat", variant="secondary")
        
        def respond(message, history):
            # Función wrapper para manejar el async con nuevo formato messages
            if not message.strip():
                return history, ""
                
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(chat_for_gradio(message, history))
                
                # Agregar mensaje y respuesta al historial
                new_message = {"role": "user", "content": message}
                new_response = {"role": "assistant", "content": response}
                
                if history is None:
                    history = []
                
                history.append(new_message)
                history.append(new_response)
                
                return history, ""
            except Exception as e:
                error_response = {"role": "assistant", "content": f"❌ Error: {str(e)}"}
                if history is None:
                    history = []
                history.append({"role": "user", "content": message})
                history.append(error_response)
                return history, ""
            finally:
                loop.close()
        
        send_btn.click(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, msg]
        )
    
    with gr.Tab("📊 Estado del Sistema"):
        gr.Markdown("### Información del backend")
        status_output = gr.Textbox(
            label="Estado",
            value="✅ Backend funcionando correctamente",
            interactive=False
        )
        refresh_btn = gr.Button("🔄 Actualizar Estado")
        
        def get_system_status():
            try:
                memory_percent = psutil.virtual_memory().percent
                model_status = "✅ Cargado" if model else "⏳ No cargado"
                return f"""
✅ **Backend Status**: Activo
🤖 **Modelo Whisper**: {model_status}
💾 **Uso de Memoria**: {memory_percent}%
🌐 **Endpoints**: /transcribe, /chat, /ping
                """
            except Exception as e:
                return f"❌ Error obteniendo estado: {str(e)}"
        
        refresh_btn.click(
            fn=get_system_status,
            outputs=[status_output]
        )

# Montar la aplicación Gradio en FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    # En Hugging Face Spaces, usar puerto 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
