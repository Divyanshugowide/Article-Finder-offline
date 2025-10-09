

# 🧠 Article Finder — Offline-First

A lightweight, **offline-capable document search engine** that retrieves English excerpts with citations and supports **role-based access control (RBAC)**.
Built for environments with limited or no internet access.

---

## 🚀 Quick Start

### **1️⃣ Environment Setup**

Create and activate a Python virtual environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### **2️⃣ Add Your PDFs**

Place all PDF files inside:

```
data/raw_pdfs/
```

> ⚠️ Do **not** commit PDFs to version control (they may contain sensitive or large data).

---

### **3️⃣ Build the Processing Pipeline**

Run the following scripts in sequence:

```bash
python scripts/01_chunk_pdfs.py
python scripts/02_build_bm25.py
python scripts/03_build_faiss.py
```

After running successfully, you should see:

```
✅ data/processed/chunks.jsonl
✅ data/idx/bm25.pkl
✅ data/idx/mE5.faiss
✅ data/idx/meta.json
```

---

### **4️⃣ Start the API Server**

Run:

```bash
uvicorn app.run_api:app --host 0.0.0.0 --port 8000
```

---

### **5️⃣ Example Query**

Use `curl` to send a query:

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" \
  -d '{"user_id":"demo","roles":["staff"],"query":"What is the operator\'s liability limit?"}'
```

Example response:

```json
{
  "answer": "The operator’s liability limit is defined under Section 10.2.3...",
  "results": [
    {
      "doc_id": "Transport_Regulations.pdf",
      "page_start": 85,
      "page_end": 87,
      "roles": ["staff", "legal"]
    }
  ]
}
```

---

## 🧩 Offline Model Setup

By default, this project uses the model
`sente­nce-transformers/all-MiniLM-L6-v2`
for FAISS embeddings.

To ensure full **offline functionality**, use one of the following methods:

---

### **Option 1 — Cache the Model (simplest)**

Before going offline for the first time, cache the model locally:

```bash
# Linux / macOS
export TRANSFORMERS_CACHE=$(pwd)/models_cache
mkdir -p models_cache

# Windows PowerShell
set TRANSFORMERS_CACHE=./models_cache
mkdir models_cache
```

Then build FAISS once:

```bash
python scripts/03_build_faiss.py
```

✅ The model will now be stored in `./models_cache`
and automatically reused offline later.

---

### **Option 2 — Fully Local Model (no internet ever needed)**

1. **Download the model manually** (on any machine with internet):
   🔗 [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

   Use either:

   ```bash
   git lfs install
   git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
   ```

   or click **Download repository** on the Hugging Face page.

2. **Move it to your offline environment:**

   ```
   models/all-MiniLM-L6-v2/
   ```

3. **Update your code** (in `scripts/03_build_faiss.py`):

   ```python
   from sentence_transformers import SentenceTransformer
   import os

   LOCAL_MODEL_PATH = os.getenv("MODEL_NAME", "models/all-MiniLM-L6-v2")
   model = SentenceTransformer(LOCAL_MODEL_PATH)
   ```

✅ Now, FAISS will build embeddings using your local model — no internet connection required.

---

### **Option 3 — Use BM25 Only (no embeddings)**

If you prefer a completely neural-free setup:

```bash
python scripts/01_chunk_pdfs.py
python scripts/02_build_bm25.py
```

This uses classic keyword-based BM25 ranking — 100% offline.
(FAISS step can be skipped.)

---

## 🧱 Project Structure

```
article-finder/
├── app/
│   ├── run_api.py          # FastAPI server
│   ├── utils.py
│   └── __init__.py
├── data/
│   ├── raw_pdfs/           # Input PDFs
│   ├── processed/          # Chunks and normalized text
│   └── idx/                # BM25 + FAISS indices
├── models/
│   └── all-MiniLM-L6-v2/   # Local model for offline use
├── scripts/
│   ├── 01_chunk_pdfs.py
│   ├── 02_build_bm25.py
│   ├── 03_build_faiss.py
│   ├── 04_query_cli.py
│   └── 05_api.py
├── requirements.txt
└── README.md
```

---

## 🧪 Self-Test Checklist

| Step        | Command                                                      | Expected Output                   |
| ----------- | ------------------------------------------------------------ | --------------------------------- |
| Chunk PDFs  | `python scripts/01_chunk_pdfs.py`                            | `data/processed/chunks.jsonl`     |
| Build BM25  | `python scripts/02_build_bm25.py`                            | `data/idx/bm25.pkl`               |
| Build FAISS | `python scripts/03_build_faiss.py`                           | `data/idx/mE5.faiss`, `meta.json` |
| CLI Query   | `python scripts/04_query_cli.py --query "..." --roles staff` | Answer + Chunks                   |
| API Test    | `curl http://127.0.0.1:8000/health`                          | `{"ok": true}`                    |

---

## 🔒 RBAC (Role-Based Access Control)

1. Rename a file to include `"restricted"` (e.g., `Contract_restricted.pdf`)
2. Rebuild indices
3. Query as:

   ```json
   {"roles": ["staff"]}
   ```

   → restricted doc **hidden**
4. Query as:

   ```json
   {"roles": ["legal"]}
   ```

   → restricted doc **visible**

---
 💡 Tip

Once downloaded, **zip the `models/all-MiniLM-L6-v2/` folder**
and reuse it for all future offline deployments — no re-downloads needed.

---

### ✅ Example End-to-End (Offline Setup)

```bash
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Place PDFs in data/raw_pdfs/
# 3. Ensure model is saved locally in models/all-MiniLM-L6-v2/

# 4. Build pipeline fully offline
python scripts/01_chunk_pdfs.py
python scripts/02_build_bm25.py
python scripts/03_build_faiss.py

# 5. Launch API
uvicorn app.run_api:app --host 0.0.0.0 --port 8000
```

Now you have a **completely offline, role-aware article finder** running locally 🎯
