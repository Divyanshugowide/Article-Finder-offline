"""
💎 Crystal Chatbot Backend — FastAPI + Ollama + TinyDB + Streaming
Optimized Version 3.5
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tinydb import TinyDB, Query

# Try to import retriever safely
try:
    from app.retrieval import Retriever
except Exception:
    Retriever = None


# =====================================
# CONFIG
# =====================================
class ChatConfig:
    OLLAMA_BASE_URL = "http://127.0.0.1:11434"
    DEFAULT_MODEL = "qwen2.5:7b"
    REQUEST_TIMEOUT = 120
    MAX_HISTORY = 10


# =====================================
# MODELS
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
# INIT APP + DB
# =====================================
app = FastAPI(title="Crystal Chat", version="3.5")

# CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db_path = Path("data/chat_memory.json")
db_path.parent.mkdir(parents=True, exist_ok=True)
db = TinyDB(db_path)

print("✅ TinyDB persistence enabled.")

# Retriever
retriever = None
if Retriever:
    try:
        retriever = Retriever(
            bm25_path="data/idx/bm25.pkl",
            faiss_path="data/idx/faiss.index",
            meta_path="data/idx/meta.json",
            alpha=0.45,
        )
        print("✅ Retriever initialized.")
    except Exception as e:
        print(f"⚠️ Retriever not loaded: {e}")


# =====================================
# HELPERS
# =====================================
def get_conversation(cid: str) -> dict:
    conv = db.get(Query().conversation_id == cid)
    if not conv:
        conv = {
            "conversation_id": cid,
            "messages": [],
            "created_at": datetime.now().isoformat(),
        }
    return conv


def save_conversation(conv: dict):
    db.upsert(conv, Query().conversation_id == conv["conversation_id"])


async def stream_ollama(messages: List[Dict[str, str]], model: str):
    """Stream Ollama responses as SSE"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.7, "max_tokens": 1200},
    }

    async with httpx.AsyncClient(timeout=ChatConfig.REQUEST_TIMEOUT) as client:
        async with client.stream("POST", f"{ChatConfig.OLLAMA_BASE_URL}/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': 'Ollama connection failed'})}\n\n"
                return
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield f"data: {json.dumps({'content': data['message']['content']})}\n\n"
                    if data.get("done"):
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        break
                except:
                    continue


# =====================================
# ROUTES
# =====================================

@app.get("/chat/conversations")
async def list_conversations():
    """List all conversation summaries"""
    all_data = db.all()
    return [
        {
            "conversation_id": c["conversation_id"],
            "created_at": c.get("created_at"),
            "updated_at": c.get("updated_at"),
            "user_role": (
                next((m["content"] for m in c.get("messages", []) if m["role"] == "user"), "")
            )[:30],
        }
        for c in sorted(all_data, key=lambda x: x.get("updated_at", ""), reverse=True)
    ]


@app.get("/chat/conversations/{cid}")
async def get_conversation_route(cid: str):
    conv = get_conversation(cid)
    return conv


@app.delete("/chat/conversations/{cid}")
async def delete_conversation(cid: str):
    db.remove(Query().conversation_id == cid)
    return {"status": "deleted"}


@app.delete("/chat/conversations/{cid}/delete_message/{index}")
async def delete_message(cid: str, index: int):
    conv = get_conversation(cid)
    if 0 <= index < len(conv["messages"]):
        conv["messages"].pop(index)
        save_conversation(conv)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Message not found")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    conv_id = request.conversation_id or str(uuid.uuid4())
    conv = get_conversation(conv_id)

    user_msg = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat(),
    }
    conv["messages"].append(user_msg)
    conv["messages"] = conv["messages"][-ChatConfig.MAX_HISTORY :]

    system_prompt = {
        "role": "system",
        "content": "You are Crystal, a helpful AI for finding and explaining articles.",
    }

    ollama_msgs = [system_prompt] + conv["messages"]

    # Add retrieved context
    sources = []
    if retriever and request.use_documents:
        try:
            results = retriever.search(request.message, roles=[request.user_role], topk=2)
            sources = results.get("results", [])
            if sources:
                context = "\n\n".join(
                    f"Doc {s.get('doc_id')}: {s.get('excerpt', '')}" for s in sources
                )
                ollama_msgs.append({"role": "system", "content": f"Relevant context:\n{context}"})
        except Exception as e:
            print("⚠️ Retrieval error:", e)

    async def generate() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'conversation_id': conv_id, 'sources': sources})}\n\n"
        full_resp = ""
        async for chunk in stream_ollama(ollama_msgs, request.model or ChatConfig.DEFAULT_MODEL):
            yield chunk
            try:
                data = json.loads(chunk[6:])
                if "content" in data:
                    full_resp += data["content"]
            except:
                pass
        if full_resp.strip():
            conv["messages"].append(
                {
                    "role": "assistant",
                    "content": full_resp.strip(),
                    "timestamp": datetime.now().isoformat(),
                    "sources": sources,
                }
            )
            conv["updated_at"] = datetime.now().isoformat()
            save_conversation(conv)

    return StreamingResponse(generate(), media_type="text/event-stream")


# =====================================
# STATIC FILES
# =====================================
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = static_dir / "optimized_chat_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>UI file not found.</h3>")


# =====================================
# ENTRY
# =====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
