from fastapi import FastAPI
from pydantic import BaseModel
import os
from app.retrieval import Retriever

class AskRequest(BaseModel):
    user_id: str
    roles: list[str]
    query: str
    topk: int = 5

app = FastAPI()

BM25_PATH = os.getenv("BM25_PATH", "data/idx/bm25.pkl")
FAISS_PATH = os.getenv("FAISS_PATH", "data/idx/mE5.faiss")
META_PATH = os.getenv("META_PATH", "data/idx/meta.json")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

retriever = Retriever(BM25_PATH, FAISS_PATH, META_PATH, MODEL_NAME)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
def ask(req: AskRequest):
    return retriever.search(req.query, req.roles, req.topk)

from fastapi.responses import HTMLResponse

from fastapi.responses import HTMLResponse
import time

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Article Finder</title>
        <style>
            body {
                font-family: "Segoe UI", Arial, sans-serif;
                background: #f9fafb;
                color: #111;
                margin: 40px auto;
                width: 90%;
                max-width: 800px;
            }
            h1 { color: #2563eb; }
            input, button, select {
                padding: 10px;
                font-size: 1rem;
                border: 1px solid #ccc;
                border-radius: 6px;
                margin-right: 6px;
            }
            button {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                transition: 0.2s;
            }
            button:hover { background: #1e40af; }
            .result {
                margin-top: 20px;
                background: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .doc { font-weight: bold; color: #1d4ed8; }
            .score { color: gray; font-size: 0.9rem; margin-bottom: 6px; }
            #loader {
                display: none;
                border: 5px solid #f3f3f3;
                border-top: 5px solid #2563eb;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            #controls { margin-bottom: 10px; }
            #info { font-size: 0.9rem; color: gray; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>🔎 Article Finder</h1>
        <p>Search across your legal or technical PDFs. Choose your role to apply access filters.</p>

        <div id="controls">
            <input id="query" type="text" placeholder="Type your question or word..." style="width:50%" />
            <select id="role">
                <option value="staff">Staff</option>
                <option value="legal">Legal</option>
                <option value="admin">Admin</option>
            </select>
            <button onclick="search()">Search</button>
            <button onclick="clearResults()" style="background:#9ca3af;">Clear</button>
        </div>

        <div id="loader"></div>
        <div id="info"></div>
        <div id="results"></div>

        <script>
        async function search() {
            const q = document.getElementById('query').value.trim();
            const role = document.getElementById('role').value;
            const resDiv = document.getElementById('results');
            const loader = document.getElementById('loader');
            const info = document.getElementById('info');

            if (!q) {
                resDiv.innerHTML = '<p style="color:red;">Please enter a query.</p>';
                return;
            }

            resDiv.innerHTML = '';
            info.innerHTML = '';
            loader.style.display = 'block';  // Show spinner
            const startTime = performance.now();

            try {
                const body = { user_id: "demo", roles: [role], query: q };
                const r = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await r.json();
                const endTime = performance.now();
                const duration = ((endTime - startTime) / 1000).toFixed(2);

                loader.style.display = 'none';  // Hide spinner

                if (!data.results || data.results.length === 0) {
                    resDiv.innerHTML = '<p>No results found.</p>';
                    return;
                }

                let html = `<h2>Answer</h2><p>${highlight(data.answer, q)}</p><h2>Results</h2>`;
                for (const res of data.results) {
                    html += `<div class="result">
                        <div class="doc">${res.doc_id} | Article ${res.article_no}</div>
                        <div class="score">Score: ${res.score.toFixed(2)} | Pages ${res.page_start}-${res.page_end}</div>
                        <div>${highlight(res.excerpt, q)}</div>
                    </div>`;
                }
                resDiv.innerHTML = html;
                info.innerHTML = `${data.results.length} result(s) found in ${duration}s`;
            } catch (err) {
                loader.style.display = 'none';
                resDiv.innerHTML = '<p style="color:red">Error: ' + err + '</p>';
            }
        }

        function clearResults() {
            document.getElementById('query').value = '';
            document.getElementById('results').innerHTML = '';
            document.getElementById('info').innerHTML = '';
        }

        function highlight(text, word) {
            if (!word) return text;
            const pattern = new RegExp(word, "gi");
            return text.replace(pattern, (m) => `<b style='background:yellow'>${m}</b>`);
        }
        </script>
    </body>
    </html>
    """
