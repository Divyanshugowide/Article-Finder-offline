# README2 — Updated Setup for Article Finder + Crystal Chat

## Overview
FastAPI services with hybrid retrieval (BM25 + FAISS) over PDFs and an enhanced chat API. This guide reflects the updated requirements, Dockerfile, and .dockerignore.

## Requirements
- Python 3.11+
- pip

Install deps:

```bash
pip install -r requirements.txt
```

## Data layout
- Place PDFs in `data/raw_pdfs/`

## Build indices (one-time or when PDFs change)
Run in order:

```bash
python scripts/01_chunk_pdfs.py
python scripts/02_build_bm25.py
python scripts/03_build_faiss.py
```

This produces:
- `data/processed/chunks.jsonl`
- `data/idx/bm25.pkl`
- `data/idx/faiss.index`
- `data/idx/meta.json`

## Run locally
- Search API (default):

```bash
uvicorn app.run_api:app --host 0.0.0.0 --port 8000 --reload
```

- Enhanced chat API (with TinyDB + upload):

```bash
uvicorn app.enhanced_chat:app --host 0.0.0.0 --port 8001 --reload
```

Upload PDFs at runtime (enhanced chat):
- Endpoint: `POST /upload/pdf` (form file: `file`)

## Docker
Build image:

```bash
docker build -t article-finder .
```

Run Search API (default `APP_MODULE=app.run_api:app`, `PORT=8000`):

```bash
docker run --rm -p 8000:8000 \
  -v ${PWD}/data:/app/data \
  article-finder
```

Run Enhanced Chat instead:

```bash
docker run --rm -p 8001:8001 \
  -e APP_MODULE=app.enhanced_chat:app \
  -e PORT=8001 \
  -v ${PWD}/data:/app/data \
  article-finder
```

Notes:
- The container expects indices in `/app/data/idx`. Volume-mount your local `data/` folder.
- If you use Ollama locally, ensure the host’s `127.0.0.1:11434` is reachable. For remote hosts, update URLs in the code.

## What changed
- requirements: trimmed unused deps; added `python-multipart` for uploads.
- Dockerfile: installs from `requirements.txt`, exposes `APP_MODULE`/`PORT` envs.
- .dockerignore: excludes `data/*` and typical Python/IDE junk for smaller images.
