"""
💎 Crystal Chatbot Backend — v6.5
Enhanced with Stop Chat, Rename Persistence, Delete, Highlighting
"""

import json
import uuid
import re
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

# Try safe import
try:
    from app.retrieval import Retriever
except Exception:
    Retriever = None


# =======================================
# CONFIG
# =======================================
class ChatConfig:
    OLLAMA_BASE_URL = "http://127.0.0.1:11434"
    DEFAULT_MODEL = "qwen2.5:7b"
    REQUEST_TIMEOUT = 120
    MAX_HISTORY = 10


# =======================================
# MODELS
# =======================================
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_role: str = "staff"
    use_documents: bool = True
    model: Optional[str] = None


# =======================================
# APP + DB
# =======================================
app = FastAPI(title="Crystal Chat", version="6.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db_path = Path("data/chat_memory.json")
db_path.parent.mkdir(parents=True, exist_ok=True)
db = TinyDB(db_path)

stop_sessions: Dict[str, bool] = {}

print("✅ TinyDB persistence enabled.")

retriever = None
if Retriever:
   
    try:
        retriever = Retriever(
            bm25_path="data/idx/bm25.pkl",
            faiss_path="data/idx/faiss.index",
            meta_path="data/idx/meta.json",
            alpha=0.45,
        )
        if not Path("data/idx/bm25.pkl").exists():
            print("⚠️ No index found. Building from raw_pdfs...")
            retriever.build_index("data/raw_pdfs")
        print("✅ Retriever initialized.")
    except Exception as e:
        print(f"⚠️ Retriever not loaded: {e}")

        


# =======================================
# HELPERS
# =======================================
def get_conversation(cid: str) -> dict:
    conv = db.get(Query().conversation_id == cid)
    if not conv:
        conv = {
            "conversation_id": cid,
            "name": "Untitled Chat",
            "messages": [],
            "created_at": datetime.now().isoformat(),
        }
    return conv


def save_conversation(conv: dict):
    db.upsert(conv, Query().conversation_id == conv["conversation_id"])


def highlight_key_info(text: str) -> str:
    """Auto-highlight legal references."""
    patterns = {
        r"(Article\s\d+)": r"==\1==",
        r"(Section\s[\dA-Za-z.\-]+)": r"==\1==",
        r"(Page[s]?\s\d+(\s*–\s*\d+)?)": r"==\1==",
        r"(\bClause\s\d+)": r"==\1==",
    }
    for pat, repl in patterns.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


async def stream_ollama(messages: List[Dict[str, str]], model: str, cid: str):
    """Stream Ollama chat."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.7, "max_tokens": 1200},
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{ChatConfig.OLLAMA_BASE_URL}/api/chat", json=payload
        ) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': 'Ollama connection failed'})}\n\n"
                return

            async for line in resp.aiter_lines():
                if stop_sessions.get(cid, False):
                    yield f"data: {json.dumps({'stopped': True})}\n\n"
                    break
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


# =======================================
# ROUTES
# =======================================
@app.get("/chat/conversations")
async def list_conversations():
    all_data = db.all()
    return [
        {
            "conversation_id": c["conversation_id"],
            "name": c.get("name", "Untitled Chat"),
            "created_at": c.get("created_at"),
            "updated_at": c.get("updated_at"),
        }
        for c in sorted(all_data, key=lambda x: x.get("updated_at", ""), reverse=True)
    ]


@app.get("/chat/conversations/{cid}")
async def get_conversation_route(cid: str):
    return get_conversation(cid)


@app.patch("/chat/conversations/{cid}")
async def rename_conversation(cid: str, payload: dict):
    conv = get_conversation(cid)
    conv["name"] = payload.get("name", conv.get("name", "Untitled Chat"))
    conv["updated_at"] = datetime.now().isoformat()
    save_conversation(conv)
    return {"status": "renamed"}


@app.delete("/chat/conversations/{cid}")
async def delete_conversation(cid: str):
    db.remove(Query().conversation_id == cid)
    stop_sessions.pop(cid, None)
    return {"status": "deleted"}


@app.post("/chat/stop/{cid}")
async def stop_chat(cid: str):
    stop_sessions[cid] = True
    return {"status": "stopped"}

from fastapi import File, UploadFile, Form
import shutil, os

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles PDF upload and triggers retriever reindex."""
    try:
        # Ensure folder exists
        raw_pdf_dir = Path("data/raw_pdfs")
        raw_pdf_dir.mkdir(parents=True, exist_ok=True)

        # Build file path
        pdf_path = raw_pdf_dir / file.filename

        # Save the uploaded PDF
        with pdf_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Confirm file actually saved
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise Exception("File failed to save (empty or missing).")

        # If retriever is initialized, trigger reindex
        if retriever:
            try:
                print(f"📄 Received PDF: {pdf_path}")
                if hasattr(retriever, "index_pdfs"):
                    retriever.index_pdfs(str(raw_pdf_dir))
                elif hasattr(retriever, "build_index"):
                    retriever.build_index(str(raw_pdf_dir))
                print(f"✅ Indexing done for {pdf_path.name}")
                return {"status": "success", "message": f"{pdf_path.name} uploaded and indexed successfully!"}
            except Exception as e:
                print(f"⚠️ Indexing failed: {e}")
                return {"status": "error", "message": f"File uploaded but indexing failed: {e}"}
        else:
            print("⚠️ Retriever not initialized.")
            return {"status": "warning", "message": f"{pdf_path.name} uploaded, but retriever inactive."}

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return {"status": "error", "message": f"Upload failed: {str(e)}"}



@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    conv_id = request.conversation_id or str(uuid.uuid4())
    stop_sessions[conv_id] = False

    conv = get_conversation(conv_id)
    user_msg = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat(),
    }
    conv["messages"].append(user_msg)
    conv["messages"] = conv["messages"][-ChatConfig.MAX_HISTORY:]

    system_prompt = {
        "role": "system",
        "content": "You are Crystal, a professional assistant that helps users find and summarize articles from PDFs.",
    }

    ollama_msgs = [system_prompt] + conv["messages"]

    # Retrieval
    sources = []
    if retriever and request.use_documents:
        try:
            results = retriever.search(request.message, roles=[request.user_role], topk=2)
            sources = results.get("results", [])
            if sources:
                ctx = "\n\n".join(
                    f"Doc {s.get('doc_id')}: {s.get('excerpt', '')}" for s in sources
                )
                ollama_msgs.append({"role": "system", "content": f"Context:\n{ctx}"})
        except Exception as e:
            print("⚠️ Retrieval error:", e)

    async def generate() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps({'conversation_id': conv_id, 'sources': sources})}\n\n"
        full_resp = ""
        async for chunk in stream_ollama(
            ollama_msgs, request.model or ChatConfig.DEFAULT_MODEL, conv_id
        ):
            yield chunk
            try:
                data = json.loads(chunk[6:])
                if "content" in data:
                    full_resp += data["content"]
            except:
                pass

        if not stop_sessions.get(conv_id, False) and full_resp.strip():
            highlighted = highlight_key_info(full_resp.strip())
            conv["messages"].append(
                {
                    "role": "assistant",
                    "content": highlighted,
                    "timestamp": datetime.now().isoformat(),
                    "sources": sources,
                }
            )
            conv["updated_at"] = datetime.now().isoformat()
            save_conversation(conv)

    return StreamingResponse(generate(), media_type="text/event-stream")


# =======================================
# STATIC
# =======================================
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = static_dir / "optimized_chat_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>UI not found.</h3>")
