from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import httpx
import uuid
import os
from typing import Dict, List, Any
from app.retrieval import Retriever

# --------------------------------------------------------
# 🔹 FastAPI Initialization
# --------------------------------------------------------
app = FastAPI(title="Article Finder + Chat Assistant", version="4.0")

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
# 🔹 Memory & Ollama Setup
# --------------------------------------------------------
CONV_MEMORY: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY = 12

async def call_ollama(messages: list[dict[str, str]], model: str = "phi3") -> str:
    """Local Ollama chat."""
    OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"Ollama error: {r.text}")
            return r.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama call failed: {e}")

# --------------------------------------------------------
# 🔹 Intent Detection (Small Talk vs Search)
# --------------------------------------------------------
def is_search_intent(text: str) -> bool:
    """Simple heuristic to detect if the user wants to search documents."""
    text = text.lower().strip()
    search_keywords = [
        "find", "search", "show", "look for", "article", "page", "where", "clause",
        "section", "in the pdf", "document", "mention", "locate"
    ]
    return any(k in text for k in search_keywords)

# --------------------------------------------------------
# 🔹 API Endpoints
# --------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "message": "Server running"}

@app.post("/ask")
def ask(req: AskRequest):
    return retriever.search(req.query, req.roles, req.topk)

from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(req: ChatRequest):
    conv_id = req.conversation_id or str(uuid.uuid4())
    CONV_MEMORY.setdefault(conv_id, [])
    for m in req.messages:
        CONV_MEMORY[conv_id].append(m)
    CONV_MEMORY[conv_id] = CONV_MEMORY[conv_id][-10:]  # keep smaller memory

    user_msgs = [m for m in req.messages if m["role"] == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message provided.")
    query = user_msgs[-1]["content"]

    wants_search = any(k in query.lower() for k in [
        "find", "search", "look", "article", "page", "where", "in pdf", "document"
    ])

    # Build message context
    messages = [
        {"role": "system", "content": "You are Crystal, a friendly assistant that helps find information in PDFs or chat casually. Speak naturally and warmly."},
        *CONV_MEMORY[conv_id],
        {"role": "user", "content": query}
    ]

    # If it's a search, add retriever context
    sources = []
    if wants_search:
        results = retriever.search(query, req.roles, req.topk)
        sources = results["results"]
        context = "\n\n".join([
            f"Document {r['doc_id']} Article {r.get('article_no','?')} Pages {r.get('page_start','?')}-{r.get('page_end','?')}: {r['excerpt']}"
            for r in sources[:3]
        ])
        messages.append({"role": "system", "content": f"Context:\n{context}"})

    async def stream_response():
        OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
        payload = {"model": "phi3", "messages": messages, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=0) as client:
                async with client.stream("POST", OLLAMA_URL, json=payload) as r:
                    async for line in r.aiter_lines():
                        if line.strip():
                            yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    # Stream back to frontend
    return StreamingResponse(stream_response(), media_type="text/event-stream")

# --------------------------------------------------------
# 🔹 UI with 3D Crystal & Lighting Effects
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
  margin: 0 auto 120px;
}
.result {
  background: rgba(255,255,255,0.9);
  border-radius: 16px;
  padding: 18px 20px;
  margin-bottom: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.result b { color: var(--primary); }

/* Chat Floating Button */
#chatBtn {
  position: fixed;
  bottom: 25px;
  right: 25px;
  width: 70px; height: 70px;
  border-radius: 50%;
  border: none;
  background: radial-gradient(circle at top left, var(--primary), var(--accent));
  color: white; font-size: 30px;
  cursor: pointer;
  box-shadow: 0 6px 18px rgba(0,0,0,0.3);
  transition: 0.3s;
  z-index: 20;
}
#chatBtn:hover { transform: scale(1.1) rotate(8deg); }

/* Chat Window */
#chatBox {
  display: none;
  flex-direction: column;
  position: fixed;
  bottom: 110px;
  right: 25px;
  width: 420px;
  height: 560px;
  background: rgba(255,255,255,0.9);
  backdrop-filter: blur(14px);
  border-radius: 22px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.25);
  overflow: hidden;
  z-index: 30;
}
#chatHeader {
  background: linear-gradient(135deg, var(--primary), var(--accent));
  color: white;
  padding: 14px;
  text-align: center;
  font-weight: 600;
  font-size: 1.1rem;
  position: relative;
}
#chatHeader::after {
  content: "";
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background: rgba(255,255,255,0.2);
}
#chatMessages {
  flex: 1;
  padding: 15px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.msg {
  max-width: 80%;
  padding: 10px 14px;
  border-radius: 16px;
  line-height: 1.45;
  animation: fadeIn 0.3s ease;
}
.user {
  align-self: flex-end;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  color: white;
  border-bottom-right-radius: 4px;
}
.assistant {
  align-self: flex-start;
  background: rgba(255,255,255,0.85);
  border-bottom-left-radius: 4px;
  border: 1px solid rgba(124,58,237,0.15);
  box-shadow: 0 2px 6px rgba(124,58,237,0.08);
}

/* Typing dots */
.typing {
  display: flex; align-items: center; gap: 4px;
}
.typing div {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--primary);
  animation: blink 1.4s infinite both;
}
.typing div:nth-child(2){animation-delay:0.2s;}
.typing div:nth-child(3){animation-delay:0.4s;}
@keyframes blink {
  0%,80%,100%{opacity:0;transform:scale(0.8);}
  40%{opacity:1;transform:scale(1);}
}
#chatInput {
  display: flex;
  padding: 10px;
  background: rgba(250,250,250,0.7);
  border-top: 1px solid #e2e8f0;
}
#chatInput input {
  flex: 1;
  padding: 10px;
  border-radius: 10px;
  border: 1px solid #d4d4d8;
  font-size: 1rem;
  background: white;
}
#chatInput button {
  margin-left: 8px;
  border-radius: 10px;
  padding: 10px 14px;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  color: white; border: none; cursor: pointer;
}
@keyframes fadeIn { from {opacity:0;transform:translateY(8px);} to {opacity:1;transform:translateY(0);} }

/* Assistant Avatar */
#assistantAvatar {
  position: absolute;
  left: 15px;
  bottom: 20px;
  width: 60px; height: 60px;
  border-radius: 50%;
  background: radial-gradient(circle at top left, var(--primary), var(--accent));
  box-shadow: 0 0 25px rgba(124,58,237,0.6);
  animation: float 3s ease-in-out infinite;
  pointer-events: none;
}
@keyframes float {
  0%,100% { transform: translateY(0); }
  50% { transform: translateY(-6px); }
}

/* Responsive */
@media (max-width: 600px) {
  #chatBox { width: 90%; height: 70%; right: 5%; bottom: 90px; }
  #controls { flex-direction: column; align-items: center; }
}
</style>
</head>
<body>

<h1>🔎 Article Finder AI</h1>

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

<!-- Floating Chat Button -->
<button id="chatBtn" onclick="toggleChat()">💬</button>

<!-- Chat Window -->
<div id="chatBox">
  <div id="chatHeader">🤖 Chat with Article Finder AI</div>
  <div id="chatMessages"></div>
  <div id="chatInput">
    <input id="chatText" placeholder="Type a message..." />
    <button onclick="sendChat()">➤</button>
  </div>
</div>

<div id="assistantAvatar"></div>

<script>
let convId=null;

function toggleChat(){
  const box=document.getElementById('chatBox');
  box.style.display=box.style.display==='flex'?'none':'flex';
}

function append(role,text){
  const div=document.createElement('div');
  div.className='msg '+role;
  if(role==='assistant') typeText(div,text);
  else div.innerHTML=text;
  document.getElementById('chatMessages').appendChild(div);
  div.scrollIntoView({behavior:'smooth',block:'end'});
}

function typeText(div,text,i=0){
  if(i<text.length){
    div.innerHTML=text.substring(0,i+1);
    setTimeout(()=>typeText(div,text,i+1),15);
  }
}

async function sendChat(){
  const input=document.getElementById('chatText');
  const text=input.value.trim();
  if(!text)return;
  append('user',text);
  input.value='';

  const loader=document.createElement('div');
  loader.className='typing assistant';
  loader.innerHTML='<div></div><div></div><div></div>';
  document.getElementById('chatMessages').appendChild(loader);
  loader.scrollIntoView({behavior:'smooth'});

  const body={messages:[{role:'user',content:text}],conversation_id:convId,roles:[document.getElementById('role').value],topk:3};
  const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();
  loader.remove();
  convId=data.conversation_id;
  append('assistant',data.reply);

  if(data.intent==='search' && data.sources?.length){
    const div=document.getElementById('results');
    div.innerHTML=data.sources.map(x=>`
      <div class='result'>
        <b>${x.doc_id}</b> | Article ${x.article_no} | Pages ${x.page_start}-${x.page_end}
        <div>${x.excerpt}</div>
      </div>`).join('');
  }
}

async function search(){
  const q=document.getElementById('query').value.trim();
  const role=document.getElementById('role').value;
  if(!q)return;
  const body={user_id:'demo',roles:[role],query:q};
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await r.json();
  const div=document.getElementById('results');
  div.innerHTML=data.results.map(x=>`
    <div class='result'>
      <b>${x.doc_id}</b> | Article ${x.article_no} | Pages ${x.page_start}-${x.page_end}
      <div>${x.excerpt}</div>
    </div>`).join('');
}

document.getElementById('chatText').addEventListener('keydown',e=>{ if(e.key==='Enter')sendChat(); });
document.getElementById('query').addEventListener('keydown',e=>{ if(e.key==='Enter')search(); });
</script>
</body>
</html>
    """
