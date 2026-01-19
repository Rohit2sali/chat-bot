from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import ChatRequest, ChatResponse
from app.llm import chat_with_llm
from app.vector_store import store_message, retrieve_context

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    context = retrieve_context(req.message)
    reply = chat_with_llm(req.message, context)

    store_message(req.message, "user")
    store_message(reply, "assistant")

    return ChatResponse(response=reply)
