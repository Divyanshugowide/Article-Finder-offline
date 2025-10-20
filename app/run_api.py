from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import httpx
import uuid
import os
from typing import Dict, List
from app.retrieval import Retriever

# --------------------------------------------------------
# 🔹 FastAPI Initialization
# --------------------------------------------------------
app = FastAPI(title="Article Finder + Chat Assistant", version="4.2")

# --------------------------------------------------------
# 🔹 Schemas
# --------------------------------------------------------
class AskRequest(BaseModel):
    user_id: str
    roles: list[str]
    query: str
    topk: int = 5

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    conversation_id: str | None = None
    roles: list[str] = ["staff"]
    topk: int = 3

# --------------------------------------------------------
# 🔹 Retriever Setup
# --------------------------------------------------------
BM25_PATH = os.getenv("BM25_PATH", "data/idx/bm25.pkl")
FAISS_PATH = os.getenv("FAISS_PATH", "data/idx/faiss.index")
META_PATH = os.getenv("META_PATH", "data/idx/meta.json")

retriever = Retriever(BM25_PATH, FAISS_PATH, META_PATH)
print("✅ Retriever initialized successfully.")

# --------------------------------------------------------
# 🔹 Ollama & Chat Setup
# --------------------------------------------------------
CONV_MEMORY: Dict[str, List[Dict[str, str]]] = {}

@app.get("/health")
def health():
    return {"ok": True, "message": "Server running"}

@app.post("/ask")
def ask(req: AskRequest):
    return retriever.search(req.query, req.roles, req.topk)


@app.post("/chat")
async def chat(req: ChatRequest):
    conv_id = req.conversation_id or str(uuid.uuid4())
    CONV_MEMORY.setdefault(conv_id, [])
    for m in req.messages:
        CONV_MEMORY[conv_id].append(m)
    CONV_MEMORY[conv_id] = CONV_MEMORY[conv_id][-10:]

    query = req.messages[-1]["content"]
    wants_search = any(k in query.lower() for k in ["find", "search", "look", "article", "pdf", "document"])

    messages = [
        {"role": "system", "content": "You are Crystal, a friendly assistant that helps find information in PDFs or chat casually."},
        *CONV_MEMORY[conv_id],
        {"role": "user", "content": query}
    ]

    sources = []
    if wants_search:
        results = retriever.search(query, req.roles, req.topk)
        sources = results["results"]
        context = "\n\n".join([f"Document {r['doc_id']} Article {r.get('article_no','?')} Pages {r.get('page_start','?')}-{r.get('page_end','?')}: {r['excerpt']}" for r in sources[:3]])
        messages.append({"role": "system", "content": f"Context:\n{context}"})

    async def stream_response():
        OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
        payload = {"model": "phi3", "messages": messages, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", OLLAMA_URL, json=payload) as r:
                    async for line in r.aiter_lines():
                        if line.strip():
                            yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

# --------------------------------------------------------
# 🔹 UI with Floating Crystal Chat Button
# --------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Article Finder AI</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --primary: #7c3aed;
  --secondary: #a78bfa;
  --accent: #e879f9;
  --bg-gradient: linear-gradient(135deg, #ede9fe 0%, #f5f3ff 100%);
}
* { box-sizing: border-box; }
body {
  font-family: 'Poppins', sans-serif;
  background: var(--bg-gradient);
  margin: 0;
  padding: 0;
  color: #1e1b4b;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
h1 {
  text-align: center;
  margin-top: 20px;
  font-size: 2.4rem;
  font-weight: 700;
  color: var(--primary);
}
#controls {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 12px;
  margin: 20px auto;
  max-width: 900px;
}
input, select, button {
  padding: 12px 16px;
  border-radius: 12px;
  border: none;
  font-size: 1rem;
}
input, select {
  background: rgba(255,255,255,0.85);
  border: 1px solid #e5e7eb;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
button {
  background: linear-gradient(135deg, var(--primary), var(--accent));
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}
button:hover { transform: scale(1.05); box-shadow: 0 4px 12px rgba(124,58,237,0.3); }

#results {
  max-width: 900px;
  margin: 0 auto 100px;
}
.result {
  background: rgba(255,255,255,0.9);
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.result b { color: var(--primary); }

/* 💎 Floating Crystal Chat Button */
#crystalChatBtn {
  position: fixed;
  bottom: 25px;
  right: 25px;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  color: white;
  border: none;
  border-radius: 50%;
  width: 70px;
  height: 70px;
  font-size: 30px;
  cursor: pointer;
  box-shadow: 0 10px 25px rgba(124,58,237,0.4);
  transition: all 0.3s ease;
  animation: pulse 2.5s infinite;
  z-index: 50;
}
#crystalChatBtn:hover {
  transform: scale(1.1) rotate(6deg);
  box-shadow: 0 0 35px rgba(231,121,249,0.6);
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 20px rgba(124,58,237,0.4); }
  50% { box-shadow: 0 0 35px rgba(231,121,249,0.8); }
}

/* Responsive */
@media (max-width: 600px) {
  #controls { flex-direction: column; align-items: center; }
  #crystalChatBtn { width: 60px; height: 60px; font-size: 26px; }
}
</style>
</head>
<body>

<h1>🔎 Mannual Article Finder AI</h1>

<div id="controls">
  <input id="query" type="text" placeholder="Search or ask..." style="width:300px;">
  <select id="role">
    <option value="staff">Staff</option>
    <option value="legal">Legal</option>
    <option value="admin">Admin</option>
  </select>
  <button onclick="search()">Search</button>
</div>

<div id="results"></div>

<!-- 💬 Floating Chat Button -->
<button id="crystalChatBtn" title="Chat with Crystal" onclick="openCrystalChat()">💎</button>

<script>
async function openCrystalChat() {
  const chatUrl = "http://127.0.0.1:8001/";
  try {
    const res = await fetch(chatUrl, { method: "GET" });
    if (res.ok) {
      window.open(chatUrl, "_blank");
    } else {
      alert("⚠️ Crystal Chatbot is not running.\\nPlease start it by running:\\npython start_chatbot.py");
    }
  } catch (err) {
    alert("⚠️ Could not connect to Crystal Chatbot.\\nMake sure it is running using:\\npython start_chatbot.py");
  }
}

async function search() {
  const q = document.getElementById('query').value.trim();
  const role = document.getElementById('role').value;
  if (!q) return;
  const body = { user_id: 'demo', roles: [role], query: q };
  const r = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const data = await r.json();
  const div = document.getElementById('results');
  div.innerHTML = data.results.map(x => `
    <div class='result'>
      <b>${x.doc_id}</b> | Article ${x.article_no} | Pages ${x.page_start}-${x.page_end}
      <div>${x.excerpt}</div>
    </div>`).join('');
}

document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter') search();
});
</script>
</body>
</html>
    """
