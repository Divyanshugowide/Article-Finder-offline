SELFTEST.md — Quick verification checklist

1) Chunking
- Run: `python scripts/01_chunk_pdfs.py`
- Check: `data/processed/chunks.jsonl` exists and has JSON lines with keys:
  doc_id, article_no, page_start, page_end, text, norm_text, roles

2) Indexing
- Run:
  - `python scripts/02_build_bm25.py`
  - `python scripts/03_build_faiss.py`
- Check: files in `data/idx/`:
  - `bm25.pkl`
  - `mE5.faiss`
  - `meta.json`

3) CLI test
- `python scripts/04_query_cli.py --query "Which body is responsible for inspections?" --roles staff`

4) RBAC test
- Rename a PDF to include `restricted` in filename (e.g., `Contract_restricted.pdf`), rebuild indices, verify results disappear for `roles=["staff"]` and appear for `roles=["legal"]`.

5) API test
- Start uvicorn and:
  - `curl http://127.0.0.1:8000/health` -> `{"ok": true}`
  - POST `/ask` returns JSON with `answer` and `results` list.
