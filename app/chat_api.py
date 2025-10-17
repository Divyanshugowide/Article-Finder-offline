# app/chat_api.py
import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

# ----------------------------
# Import your Retriever
# ----------------------------
from app.retrieval import Retriever

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI(title="Article Finder Chatbot (Ollama Local)", version="1.0")

# ----------------------------
# Detect /static folder robustly
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # one level up from app/
STATIC_DIR = ROOT_DIR / "static"

print(f"📁 Static folder resolved as: {STATIC_DIR}")

if not STATIC_DIR.exists():
    raise RuntimeError(f"⚠️ Static folder not found at {STATIC_DIR} — please create /static/chat_ui.html")

# Mount /static folder for serving JS/CSS/HTML
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ----------------------------
# Pydantic Models
# ----------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: Optional[List[Message]] = None
    conversation_id: Optional[str] = None
    use_retriever: Optional[bool] = True
    roles: Optional[List[str]] = ["staff"]
    topk: Optional[int] = 3

class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    sources: Optional[List[Dict[str, Any]]] = None

# ----------------------------
# Load Retriever
# ----------------------------
BM25_PKL = "data/idx/bm25.pkl"
FAISS_IDX = "data/idx/faiss.index"      # ✅ Fixed file name
META_JSON = "data/idx/meta.json"

retriever = None
try:
    retriever = Retriever(
        bm25_path=BM25_PKL,
        faiss_path=FAISS_IDX,
        meta_path=META_JSON,
        alpha=0.45,
    )
    print("✅ Retriever initialized successfully.")
except Exception as e:
    print(f"[WARN] Retriever failed to initialize: {e}")

# ----------------------------
# In-memory conversation store
# ----------------------------
CONV_MEMORY: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY = 12

# ----------------------------
# Call Ollama API locally
# ----------------------------
async def call_ollama_api(messages: list[dict[str, str]], model: str = "phi3") -> str:
    """
    Use local Ollama model (phi3, llama3, mistral, etc.)
    """
    OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code,
                                    detail=f"Ollama API error: {response.text}")
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API call failed: {e}")

# ----------------------------
# Main chat endpoint
# ----------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    conv_id = request.conversation_id or str(uuid.uuid4())
    CONV_MEMORY.setdefault(conv_id, [])

    if request.messages:
        for msg in request.messages:
            CONV_MEMORY[conv_id].append({"role": msg.role, "content": msg.content})
        CONV_MEMORY[conv_id] = CONV_MEMORY[conv_id][-MAX_HISTORY:]

    user_msgs = [m for m in request.messages if m.role == "user"] if request.messages else []
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message found.")
    user_text = user_msgs[-1].content

    # Retriever
    sources, context_text = [], ""
    if request.use_retriever and retriever:
        try:
            results = retriever.search(user_text, roles=request.roles, topk=request.topk)
            for r in results["results"]:
                sources.append({
                    "doc_id": r.get("doc_id"),
                    "article_no": r.get("article_no"),
                    "page_start": r.get("page_start"),
                    "page_end": r.get("page_end"),
                    "score": r.get("score"),
                    "excerpt": r.get("excerpt")
                })
            top_excerpts = [s["excerpt"] for s in sources[:3]]
            context_text = "\n\n".join(top_excerpts)
        except Exception as e:
            print(f"[WARN] Retriever search failed: {e}")

    # Build messages for Ollama
    ollama_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for the Article Finder project. "
                "Use the provided document context to answer clearly and concisely."
            ),
        },
        *CONV_MEMORY[conv_id],
    ]
    if context_text:
        ollama_messages.append({"role": "system", "content": f"Context:\n{context_text}"})

    # Call local model
    reply = await call_ollama_api(ollama_messages, model="phi3")

    # Store assistant reply
    CONV_MEMORY[conv_id].append({"role": "assistant", "content": reply})
    CONV_MEMORY[conv_id] = CONV_MEMORY[conv_id][-MAX_HISTORY:]

    return ChatResponse(conversation_id=conv_id, reply=reply, sources=sources)

# ----------------------------
# Conversation management
# ----------------------------
@app.get("/conversation/{conv_id}")
async def get_conversation(conv_id: str):
    return JSONResponse(content={
        "conversation_id": conv_id,
        "messages": CONV_MEMORY.get(conv_id, [])
    })

@app.post("/conversation/{conv_id}/reset")
async def reset_conversation(conv_id: str):
    CONV_MEMORY[conv_id] = []
    return JSONResponse(content={"conversation_id": conv_id, "status": "reset"})

# ----------------------------
# Serve UI (chat_ui.html)
# ----------------------------
@app.get("/")
def ui_index():
    ui_path = STATIC_DIR / "chat_ui.html"
    print(f"🔍 Looking for UI at: {ui_path}")
    if ui_path.exists():
        return FileResponse(str(ui_path), media_type="text/html")
    return {"message": f"chat_ui.html not found in {STATIC_DIR}"}
