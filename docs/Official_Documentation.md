# Article Finder — Official Project Documentation

Author: Divyanshu  
Date: 2025-10-08  
Version: 1.0

Abstract
Article Finder is an offline-first document search system that indexes local PDF documents, builds both lexical (BM25) and semantic (FAISS) indices, and serves an HTTP API with role-based access control (RBAC). 
The project targets constrained or disconnected environments where internet access is partial or unavailable. This document presents the problem statement, objectives, approach and methodology, architecture, implementation details, build/run instructions, testing and validation procedures, security considerations, limitations, and future work.

1. Introduction
Modern teams often need to search across policy, legal, and technical PDFs without relying on cloud services.
 Article Finder addresses this by:
  (1) extracting structured chunks from PDFs, 
  (2) indexing content using a hybrid approach, and 
  (3) serving an API and simple web UI for users with different roles. The system emphasizes privacy, portability, and offline usability.

2. Objectives
- Provide an offline-capable search engine for local PDFs.
- Combine keyword accuracy (BM25) with semantic recall (embeddings + FAISS).
- Enforce basic access control using filename-driven RBAC.
- Offer both a programmatic API and a minimal browser UI.
- Keep the pipeline simple and reproducible via scripts.

3. Data Sources and Constraints
- Inputs: PDF documents placed in data/raw_pdfs/.
- Outputs:
  - data/processed/chunks.jsonl: normalized, chunked records with metadata and roles.
  - data/idx/bm25.pkl: pickled BM25 index plus chunk metadata.
  - data/idx/mE5.faiss: FAISS index built from Sentence-Transformer embeddings.
  - data/idx/meta.json: JSON metadata for all chunks.
- Constraints: No external network dependency at runtime; embedding model must be cached or available locally for FAISS step.

4. Approach and Methodology
4.1 PDF parsing and chunking
- Extraction: Uses PyMuPDF (fitz) to extract page-level text.
- Chunking: Detects article/section headings via regex (e.g., “Article 10”, “Section IV”) and aggregates text into coherent chunks with page ranges.
- Metadata: Captures doc_id, article_no, page_start/end, raw text, normalized text.

4.2 Text normalization
- Lowercasing and quotation normalization.
- Whitespace collapsing to stabilize tokenization and embeddings.

4.3 Index construction
- BM25 (lexical): Tokenizes normalized text into terms; builds BM25Okapi index for exact/keyword relevance.
- Embeddings + FAISS (semantic): Encodes normalized text with sentence-transformers (default: sentence-transformers/all-MiniLM-L6-v2), L2-normalized, and indexed in FAISS (inner product).

4.4 Hybrid ranking
- Query is normalized, tokenized, encoded.
- Compute BM25 scores and FAISS similarities against all chunks.
- Normalize both score arrays to [0, 1].
- Fuse via weighted sum: fused = alpha * vec_norm + (1 - alpha) * bm25_norm (default alpha = 0.4).
- Enforce keyword overlap by default for precision (prevents purely semantic but off-topic matches).

4.5 Role-Based Access Control (RBAC)
- Policy: Filename-based convention.
  - If filename contains “restricted” (case-insensitive): roles = ["legal", "admin"].
  - Otherwise: roles = ["staff", "legal", "admin"].
- Enforcement: At query time, filter out chunks whose roles do not intersect with the user’s roles.

4.6 Offline-first model strategy
- Option A (cache): Set TRANSFORMERS_CACHE and run FAISS build once to cache the model.
- Option B (fully local): Download model repo to models/all-MiniLM-L6-v2 and set MODEL_NAME to that path.
- Option C (BM25-only): Skip FAISS step to run 100% offline without any model files.

5. System Architecture
5.1 Components
- app/run_api.py: FastAPI service exposing GET /health, POST /ask, and a minimal HTML UI (GET /).
- app/retrieval.py: Retriever class implementing hybrid BM25+FAISS ranking with RBAC and keyword-overlap enforcement.
- app/normalize.py: Text normalization utilities.
- scripts/01_chunk_pdfs.py: PDF ingestion, chunking, normalization, role tagging.
- scripts/02_build_bm25.py: BM25 index construction.
- scripts/03_build_faiss.py: Embeddings and FAISS index construction.
- scripts/04_query_cli.py: Command-line querying for quick checks and automation.
- conf/glossary_en.json: Example synonyms (ready for future query expansion).

5.2 Runtime flow
- Ingestion: PDFs → chunks.jsonl (with roles and normalized text).
- Indexing: chunks.jsonl → bm25.pkl and mE5.faiss (+ meta.json).
- Serving: FastAPI loads indices and model; POST /ask processes queries.
- Query processing: normalize + tokenize + encode → BM25 + FAISS → fuse → RBAC + keyword filter → results.

5.3 Configuration (Environment Variables)
- BM25_PATH (default: data/idx/bm25.pkl)
- FAISS_PATH (default: data/idx/mE5.faiss)
- META_PATH (default: data/idx/meta.json)
- MODEL_NAME (default: sentence-transformers/all-MiniLM-L6-v2 or a local folder)
- TRANSFORMERS_CACHE (optional local cache directory)

6. Implementation Details
6.1 Key responsibilities
- Chunking: Regex-driven article/section detection (adjustable) and page-span tracking.
- Hybrid retrieval: Score normalization and weighted fusion; strict overlap option for precision.
- RBAC: Lightweight filename convention to guard restricted content.
- Fallback: Exact substring match if fused ranking yields no results.

6.2 API design
- GET /health → {"ok": true}
- POST /ask
  - Request
    ```json path=null start=null
    {
      "user_id": "demo",
      "roles": ["staff"],
      "query": "What is the operator's liability limit?",
      "topk": 5
    }
    ```
  - Response
    ```json path=null start=null
    {
      "answer": "The operator’s liability limit is defined under Section 10.2.3...",
      "results": [
        {
          "doc_id": "Transport_Regulations.pdf",
          "article_no": "3",
          "page_start": 85,
          "page_end": 87,
          "score": 0.84,
          "roles": ["staff", "legal", "admin"],
          "excerpt": "..."
        }
      ]
    }
    ```

6.3 CLI usage
```bash path=null start=null
python scripts/04_query_cli.py --query "operator liability limit" --roles staff --topk 5
```

7. Build and Run Instructions (Windows PowerShell)
7.1 Environment setup
```powershell path=null start=null
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

7.2 Add PDFs
Place files into data/raw_pdfs/ (do not commit sensitive PDFs).

7.3 Build pipeline
```powershell path=null start=null
python scripts/01_chunk_pdfs.py
python scripts/02_build_bm25.py
# Optional (requires model, cache or local folder):
python scripts/03_build_faiss.py
```
Expected artifacts:
- data/processed/chunks.jsonl
- data/idx/bm25.pkl
- data/idx/mE5.faiss (if FAISS built)
- data/idx/meta.json (if FAISS built)

7.4 Run API server
```powershell path=null start=null
uvicorn app.run_api:app --host 0.0.0.0 --port 8000
```

7.5 Example API query
```bash path=null start=null
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" \
  -d '{"user_id":"demo","roles":["staff"],"query":"What is the operator\'s liability limit?"}'
```

8. Testing and Validation
8.1 Self-test checklist
- Chunk PDFs → data/processed/chunks.jsonl
- Build BM25 → data/idx/bm25.pkl
- Build FAISS → data/idx/mE5.faiss, data/idx/meta.json
- CLI query → Returns ranked results with excerpts
- API health → GET /health returns {"ok": true}

8.2 RBAC validation
- Rename a PDF to include “restricted”.
- Rebuild indices.
- Query with roles=["staff"] → restricted chunks hidden.
- Query with roles=["legal"] → restricted chunks visible.

8.3 Practical checks
- Precision: Verify that enabling keyword overlap improves topicality of results.
- Latency: Measure time from request to response (the UI page displays a simple timing).

9. Security, Privacy, and Compliance
- Data locality: All processing and serving occur locally; PDFs are not uploaded.
- Version control hygiene: Keep PDFs out of Git; artifacts can be regenerated.
- RBAC scope: Filename-based RBAC is lightweight; for stronger guarantees, consider per-chunk policies or authenticated users.

10. Limitations
- Chunking relies on regex patterns and may miss complex structures.
- RBAC is derived from filenames, not document contents.
- No integrated authentication or audit logging in the API.
- No quantitative IR metrics (MAP/NDCG) reported; evaluation is manual in this version.

11. Future Work
- Smarter segmentation (TOC-based, layout-aware chunking).
- Query expansion (use conf/glossary_en.json) and synonym-aware ranking.
- Stronger RBAC: Per-chunk labels, user auth, and signed tokens.
- Caching and batching for lower latency on repeated queries.
- Optional reranker (e.g., cross-encoder) in a fully offline pipeline.
- Evaluation suite with labeled queries and retrieval metrics.

12. References and Credits
- Sentence-Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- Rank-BM25: https://pypi.org/project/rank-bm25/
- PyMuPDF: https://pymupdf.readthedocs.io/
- FastAPI: https://fastapi.tiangolo.com/

Appendix A: File/Module Summary
- app/run_api.py: API endpoints and minimal HTML UI.
- app/retrieval.py: Hybrid search and RBAC filtering.
- app/normalize.py: Normalization helpers.
- scripts/01_chunk_pdfs.py: PDF → chunks.jsonl with roles.
- scripts/02_build_bm25.py: BM25 index build.
- scripts/03_build_faiss.py: Embeddings + FAISS build.
- scripts/04_query_cli.py: CLI for quick queries.
- conf/glossary_en.json: Example glossary for future query expansion.
