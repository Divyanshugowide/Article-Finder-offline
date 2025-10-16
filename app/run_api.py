from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from app.retrieval import Retriever


# --------------------------------------------------------
# 🔹 FastAPI App Initialization
# --------------------------------------------------------
app = FastAPI(title="Article Finder API", version="2.0")


# --------------------------------------------------------
# 🔹 Request Schema
# --------------------------------------------------------
class AskRequest(BaseModel):
    user_id: str
    roles: list[str]
    query: str
    topk: int = 5


# --------------------------------------------------------
# 🔹 Environment Paths / Config
# --------------------------------------------------------
BM25_PATH = os.getenv("BM25_PATH", "data/idx/bm25.pkl")
FAISS_PATH = os.getenv("FAISS_PATH", "data/idx/mE5.faiss")
META_PATH = os.getenv("META_PATH", "data/idx/meta.json")

# Initialize the hybrid retriever
retriever = Retriever(BM25_PATH, FAISS_PATH, META_PATH)


# --------------------------------------------------------
# 🔹 API Routes
# --------------------------------------------------------
@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "message": "Server is running!"}


@app.post("/ask")
def ask(req: AskRequest):
    """Main retrieval endpoint."""
    return retriever.search(req.query, req.roles, req.topk)


# --------------------------------------------------------
# 🔹 Frontend (UI)
# --------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    """Modern animated UI for Article Finder."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Article Finder</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: "Poppins", sans-serif;
                background: radial-gradient(circle at 10% 20%, #f9fafc, #e0e7ff 90%);
                color: #111;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                padding: 20px;
            }

            /* Title and Intro */
            h1 {
                font-size: 2.8rem;
                background: linear-gradient(90deg, #2563eb, #1d4ed8, #60a5fa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                letter-spacing: 1px;
                margin-bottom: 10px;
                text-align: center;
                animation: fadeIn 1.5s ease;
            }
            #intro {
                font-size: 1.3rem;
                color: #2563eb;
                font-weight: 600;
                text-align: center;
                margin-bottom: 20px;
                animation: typing 3s steps(30, end), blink 0.8s infinite;
                white-space: nowrap;
                overflow: hidden;
                border-right: 3px solid #2563eb;
                width: 0;
            }
            #subtitle {
                color: #555;
                font-size: 1.2rem;
                margin-bottom: 30px;
                text-align: center;
                animation: fadeIn 2s ease;
            }

            @keyframes fadeIn {
                from {opacity: 0; transform: translateY(-10px);}
                to {opacity: 1; transform: translateY(0);}
            }
            @keyframes typing {
                from { width: 0; }
                to { width: 320px; }
            }
            @keyframes blink {
                0%, 100% { border-color: transparent; }
                50% { border-color: #2563eb; }
            }

            /* Search Controls */
            #controls {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 10px;
                margin-bottom: 25px;
            }
            input, select, button {
                padding: 12px 16px;
                font-size: 1rem;
                border-radius: 10px;
                border: 1px solid #d1d5db;
                transition: all 0.3s;
            }
            input {
                width: 60%;
                min-width: 250px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            }
            input:focus {
                border-color: #2563eb;
                box-shadow: 0 2px 10px rgba(37,99,235,0.3);
            }
            select {
                background: white;
                cursor: pointer;
            }
            button {
                background: linear-gradient(135deg, #2563eb, #1e40af);
                color: white;
                border: none;
                cursor: pointer;
                font-weight: 600;
                transition: 0.3s;
                box-shadow: 0 4px 10px rgba(37,99,235,0.3);
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 15px rgba(37,99,235,0.4);
            }
            button:disabled {
                background: #9ca3af;
                cursor: not-allowed;
            }

            /* Loader */
            #loader {
                display: none;
                border: 6px solid #f3f3f3;
                border-top: 6px solid #2563eb;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin: 40px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            /* Results */
            #results {
                display: flex;
                flex-direction: column;
                gap: 20px;
                margin-top: 20px;
                width: 100%;
                max-width: 900px;
            }
            .result {
                background: rgba(255,255,255,0.8);
                backdrop-filter: blur(8px);
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }
            .result:hover {
                transform: translateY(-5px) scale(1.01);
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }
            .doc {
                font-weight: 700;
                color: #1d4ed8;
                font-size: 1.1rem;
                margin-bottom: 5px;
            }
            .score {
                color: #666;
                font-size: 0.9rem;
                margin-bottom: 10px;
            }
            #info {
                margin-top: 10px;
                color: #555;
                font-size: 0.9rem;
                text-align: center;
            }

            mark {
                padding: 2px 4px;
                border-radius: 4px;
            }

            @media (max-width: 600px) {
                input { width: 100%; }
                #intro { width: 240px; }
            }
        </style>
    </head>

    <body>
        <h1>🔎 Article Finder</h1>
        <div id="intro">Let's Find Words...</div>
        <p id="subtitle">Powered by AI + Hybrid Retrieval (BM25 + FAISS)</p>

        <div id="controls">
            <input id="query" type="text" placeholder="Type your question or keyword..." />
            <select id="role">
                <option value="staff">Staff</option>
                <option value="legal">Legal</option>
                <option value="admin">Admin</option>
            </select>
            <button id="searchBtn" onclick="search()">Search</button>
            <button onclick="clearResults()" style="background:#6b7280;">Clear</button>
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
            const searchBtn = document.getElementById('searchBtn');

            if (!q) {
                resDiv.innerHTML = '<p style="color:red;">Please enter a query.</p>';
                return;
            }

            resDiv.innerHTML = '';
            info.innerHTML = '';
            loader.style.display = 'block';
            searchBtn.disabled = true;
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

                loader.style.display = 'none';
                searchBtn.disabled = false;

                if (!data.results || data.results.length === 0) {
                    resDiv.innerHTML = '<p>No results found.</p>';
                    return;
                }

                let html = `<h2 style="color:#1e40af;margin-bottom:10px;">Top Answer</h2>
                            <div class="result">${data.answer}</div>
                            <h2 style="margin-top:25px;color:#1e40af;">All Results</h2>`;
                for (const res of data.results) {
                    html += `
                        <div class="result">
                            <div class="doc">${res.doc_id} | Article ${res.article_no}</div>
                            <div class="score">Score: ${res.score.toFixed(2)} | Page ${res.page_start}-${res.page_end}</div>
                            <div>${res.excerpt}</div>
                        </div>`;
                }
                resDiv.innerHTML = html;
                info.innerHTML = `${data.results.length} result(s) found in ${duration}s`;

                window.scrollTo({ top: resDiv.offsetTop, behavior: 'smooth' });
            } catch (err) {
                loader.style.display = 'none';
                searchBtn.disabled = false;
                resDiv.innerHTML = '<p style="color:red">Error: ' + err + '</p>';
            }
        }

        function clearResults() {
            document.getElementById('query').value = '';
            document.getElementById('results').innerHTML = '';
            document.getElementById('info').innerHTML = '';
        }
        </script>
    </body>
    </html>
    """
# --------------------------------------------------------
# End of File
# --------------------------------------------------------


