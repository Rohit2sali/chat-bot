from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.schemas import ChatRequest, ChatResponse
from app.llm import chat_with_llm
from app.vector_store import store_message, retrieve_context

app = FastAPI()

# --- Absolute path resolution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /opt/render/project/src/app
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")     # /opt/render/project/src/static

# Serve static files safely
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    context = retrieve_context(req.message)
    reply = chat_with_llm(req.message, context)

    store_message(req.message, "user")
    store_message(reply, "assistant")

    return ChatResponse(response=reply)

