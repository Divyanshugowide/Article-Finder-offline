"""
Enhanced FastAPI Chatbot Backend (Fast Streaming + TinyDB Persistence + Auto Context)
"""

import json
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tinydb import TinyDB, Query

from app.retrieval import Retriever


# =====================================
# CONFIGURATION
# =====================================
class ChatConfig:
    OLLAMA_BASE_URL = "http://127.0.0.1:11434"
    DEFAULT_MODEL = "qwen2.5:7b"
    FALLBACK_MODEL = "phi3"
    REQUEST_TIMEOUT = 120
    MAX_HISTORY = 10
    STREAM_DELAY = 0.0  # Instant stream


# =====================================
# Pydantic Models
# =====================================
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_role: str = "staff"
    use_documents: bool = True
    model: Optional[str] = None


# =====================================
# Initialize App and Database
# =====================================
app = FastAPI(title="Crystal Chatbot", version="3.0")

db_path = Path("data/chat_memory.json")
db_path.parent.mkdir(parents=True, exist_ok=True)
db = TinyDB(db_path)

print("✅ TinyDB persistence enabled.")

# Initialize retriever
try:
    retriever = Retriever(
        bm25_path="data/idx/bm25.pkl",
        faiss_path="data/idx/faiss.index",
        meta_path="data/idx/meta.json",
        alpha=0.45
    )
    print("✅ Retriever initialized with FAISS index.")
except Exception as e:
    print(f"⚠️ Retriever not loaded: {e}")
    retriever = None


# =====================================
# Helpers
# =====================================
def get_conversation(cid: str) -> dict:
    result = db.get(Query().conversation_id == cid)
    return result or {"conversation_id": cid, "messages": [], "created_at": datetime.now().isoformat()}


def save_conversation(conv: dict):
    db.upsert(conv, Query().conversation_id == conv["conversation_id"])


async def stream_ollama_response(messages: List[Dict[str, str]], model: str):
    """Stream text directly from Ollama (SSE style)"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.7, "max_tokens": 1200},
    }
    async with httpx.AsyncClient(timeout=ChatConfig.REQUEST_TIMEOUT) as client:
        async with client.stream("POST", f"{ChatConfig.OLLAMA_BASE_URL}/api/chat", json=payload) as response:
            if response.status_code != 200:
                yield f"data: {json.dumps({'error': 'Model connection failed'})}\n\n"
                return

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        content = data["message"]["content"]
                        yield f"data: {json.dumps({'content': content})}\n\n"
                    if data.get("done"):
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        break
                except Exception:
                    continue


# =====================================
# ROUTES
# =====================================

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/chat/conversations")
async def list_conversations():
    """Return all stored conversation summaries"""
    all_convs = db.all()
    return sorted(all_convs, key=lambda c: c.get("created_at", ""), reverse=True)


@app.get("/chat/conversations/{cid}")
async def get_conversation_route(cid: str):
    conv = get_conversation(cid)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/chat/conversations/{cid}")
async def delete_conversation(cid: str):
    db.remove(Query().conversation_id == cid)
    return {"status": "deleted", "conversation_id": cid}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Main streaming chat endpoint"""
    conv_id = request.conversation_id or str(uuid.uuid4())
    conv = get_conversation(conv_id)

    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat()
    }
    conv["messages"].append(user_message)

    # Keep conversation short
    conv["messages"] = conv["messages"][-ChatConfig.MAX_HISTORY:]

    # Build system + context prompt
    system_prompt = {
        "role": "system",
        "content": (
            "You are Crystal, a smart, friendly assistant in the Article Finder System. "
            "You can use retrieved PDF context to answer questions accurately and clearly. "
            "If no context is available, answer conversationally but relevantly."
        )
    }

    ollama_messages = [system_prompt]
    ollama_messages.extend([
        {"role": msg["role"], "content": msg["content"]}
        for msg in conv["messages"]
    ])

    # Retrieve relevant docs if needed
    sources = []
    if retriever and request.use_documents:
        try:
            results = retriever.search(request.message, roles=[request.user_role], topk=2)
            sources = results.get("results", [])[:2]
            if sources:
                context_text = "\n\n".join([
                    f"Document {s.get('doc_id', '?')}: {s.get('excerpt', '')}" for s in sources
                ])
                ollama_messages.append({"role": "system", "content": f"Relevant context:\n{context_text}"})
        except Exception as e:
            print("⚠️ Retrieval error:", e)

    # Streaming generator
    async def generate():
        yield f"data: {json.dumps({'conversation_id': conv_id, 'sources': sources})}\n\n"
        full_response = ""
        async for chunk in stream_ollama_response(ollama_messages, request.model or ChatConfig.DEFAULT_MODEL):
            yield chunk
            if "content" in chunk:
                try:
                    chunk_data = json.loads(chunk[6:])
                    if "content" in chunk_data:
                        full_response += chunk_data["content"]
                except:
                    pass

        # Save final assistant message
        if full_response.strip():
            conv["messages"].append({
                "role": "assistant",
                "content": full_response.strip(),
                "timestamp": datetime.now().isoformat(),
                "sources": sources
            })
            conv["updated_at"] = datetime.now().isoformat()
            save_conversation(conv)

    return StreamingResponse(generate(), media_type="text/event-stream")


# =====================================
# STATIC UI SERVE
# =====================================
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve UI"""
    ui = static_dir / "optimized_chat_ui.html"
    if ui.exists():
        return HTMLResponse(ui.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>UI not found — please check static/optimized_chat_ui.html</h3>", status_code=404)


# =====================================
# RUN SERVER
# =====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
