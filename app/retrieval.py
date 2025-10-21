import os
import re
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from app.normalize import normalize_text
from PyPDF2 import PdfReader


class Retriever:
    def __init__(self, bm25_path, faiss_path, meta_path, alpha=0.3):
        """
        💎 Universal Hybrid Retriever (BM25 + FAISS)
        Combines lexical + semantic similarity with distinct color highlights:
          🟡 BM25 → exact keyword match
          🟢 FAISS → semantically similar (non-literal)
        """
        self.alpha = alpha
        self.faiss_topk = 50
        self.semantic_threshold = 0.35
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        print("⚙️ Initializing Universal Hybrid Retriever (BM25 + FAISS)...")

        # Model
        self.model = SentenceTransformer(self.model_name)

        self.bm25_path = Path(bm25_path)
        self.faiss_path = Path(faiss_path)
        self.meta_path = Path(meta_path)

        # Try loading indices if available
        if self.bm25_path.exists() and self.faiss_path.exists() and self.meta_path.exists():
            self._load_indexes()
        else:
            print("⚠️ Index files not found — please build the index first.")

    # ------------------------------------------------------------------
    def _load_indexes(self):
        """Load BM25, FAISS, and metadata from disk."""
        with open(self.bm25_path, "rb") as f:
            bm25_obj = pickle.load(f)
        self.bm25 = bm25_obj["bm25"]
        self.meta = bm25_obj["meta"]

        self.index = faiss.read_index(str(self.faiss_path))

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta_json = json.load(f)

        print(f"✅ Loaded BM25, FAISS, and metadata ({len(self.meta_json)} chunks).")

    # ------------------------------------------------------------------
    def build_index(self, pdf_dir: str):
        """
        🔄 Build new hybrid index from PDFs in a directory.
        - Extract text from PDFs
        - Chunk it
        - Build BM25 and FAISS indices
        """
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        print(f"📚 Building new index from {pdf_dir}...")
        docs = []
        meta = []

        for pdf_file in pdf_dir.glob("*.pdf"):
            print(f"📖 Reading: {pdf_file.name}")
            try:
                reader = PdfReader(str(pdf_file))
                full_text = ""
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    text = normalize_text(text)
                    if len(text.strip()) > 50:  # skip empty pages
                        full_text += text + "\n"

                # Chunking
                chunks = self._chunk_text(full_text, chunk_size=500, overlap=80)
                for idx, chunk in enumerate(chunks):
                    meta.append({
                        "doc_id": pdf_file.stem,
                        "page_start": 1,
                        "page_end": len(reader.pages),
                        "chunk_id": idx,
                        "text": chunk,
                        "norm_text": normalize_text(chunk),
                        "roles": self._assign_roles_from_filename(pdf_file.name)
                    })
                    docs.append(chunk)
            except Exception as e:
                print(f"⚠️ Error reading {pdf_file.name}: {e}")

        if not docs:
            print("❌ No valid PDF text found. Index not created.")
            return

        # BM25
        tokenized_corpus = [normalize_text(d).split() for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)

        # FAISS
        print("🧠 Encoding embeddings for FAISS...")
        embeddings = self.model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))

        # Save all
        print("💾 Saving indexes to disk...")
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "meta": meta}, f)
        faiss.write_index(index, str(self.faiss_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Index built successfully! {len(meta)} chunks indexed.")

        # Reload indexes into memory
        self._load_indexes()

    # ------------------------------------------------------------------
    def index_pdfs(self, pdf_dir: str):
     """Simple reindex method to refresh embeddings or tokenization."""
     print(f"🔄 Rebuilding index from {pdf_dir} ...")
     self.build_index(pdf_dir)


    # ------------------------------------------------------------------
    def _chunk_text(self, text, chunk_size=500, overlap=100):
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # ------------------------------------------------------------------
    def _normalize_top(self, scores, top_n=10):
        scores = np.array(scores)
        if len(scores) == 0:
            return np.zeros_like(scores)
        top_mean = np.mean(np.sort(scores)[-top_n:])
        if top_mean == 0:
            return np.zeros_like(scores)
        return np.clip(scores / (top_mean + 1e-8), 0, 1)

    def _normalize_01(self, scores):
        scores = np.array(scores)
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # ------------------------------------------------------------------
    def search(self, query, roles, topk=5):
        """Perform hybrid retrieval with RBAC."""
        q_norm = normalize_text(query)
        tokens = q_norm.split()
        query_lower = query.lower().strip()

        bm25_scores = np.array(self.bm25.get_scores(tokens))
        bm25_norm = self._normalize_top(bm25_scores)

        q_emb = self.model.encode([q_norm], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q_emb.astype("float32"), self.faiss_topk)

        vec_scores = np.zeros(len(self.meta_json))
        for idx, score in zip(I[0], D[0]):
            if score >= self.semantic_threshold:
                vec_scores[idx] = score
        vec_norm = self._normalize_01(vec_scores)

        fused = self.alpha * vec_norm + (1 - self.alpha) * bm25_norm

        for i, chunk in enumerate(self.meta_json):
            text = chunk["norm_text"]
            overlap = sum(tok in text for tok in tokens)
            if overlap == 0:
                fused[i] -= 0.2
            if query_lower in text:
                fused[i] += 0.2

        ranked_idx = np.argsort(-fused)
        results = []

        for i in ranked_idx:
            if fused[i] < 0.05:
                continue

            chunk = self.meta_json[i]
            chunk_roles = chunk.get("roles", self._assign_roles_from_filename(chunk.get("doc_id", "")))

            if not set(roles).intersection(set(chunk_roles)):
                continue

            excerpt = self._highlight_keywords(
                chunk["text"][:700],
                bm25_tokens=tokens,
                faiss_query_emb=q_emb[0]
            )

            results.append({
                "doc_id": chunk["doc_id"],
                "page_start": chunk.get("page_start", 1),
                "page_end": chunk.get("page_end", 1),
                "score": round(float(fused[i]), 3),
                "roles": chunk_roles,
                "excerpt": excerpt
            })

            if len(results) >= topk:
                break

        if not results:
            print("⚠️ No strong hybrid result — fallback to BM25.")
            top_idx = np.argsort(-bm25_scores)[:topk]
            for i in top_idx:
                chunk = self.meta_json[i]
                excerpt = self._highlight_keywords(
                    chunk["text"][:700],
                    bm25_tokens=tokens,
                    faiss_query_emb=q_emb[0]
                )
                results.append({
                    "doc_id": chunk["doc_id"],
                    "score": round(float(bm25_norm[i]), 3),
                    "roles": chunk.get("roles", []),
                    "excerpt": excerpt
                })

        return {
            "answer": results[0]["excerpt"] if results else "❌ No relevant section found.",
            "results": results
        }

    # ------------------------------------------------------------------
    def _highlight_keywords(self, text, bm25_tokens, faiss_query_emb=None):
        bm25_tokens_clean = {tok.lower() for tok in bm25_tokens if tok.strip()}
        highlighted = text

        for tok in bm25_tokens_clean:
            pattern = re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f"<mark style='background:yellow;font-weight:bold;'>{m.group(0)}</mark>",
                highlighted
            )

        if faiss_query_emb is not None:
            try:
                words = list(set(re.findall(r"\b\w+\b", text)))
                candidate_words = [w for w in words if w.lower() not in bm25_tokens_clean]
                if not candidate_words:
                    return highlighted

                word_embs = self.model.encode(candidate_words, convert_to_numpy=True, normalize_embeddings=True)
                sims = np.dot(word_embs, faiss_query_emb.T).flatten()
                top_indices = sims.argsort()[-8:][::-1]
                top_semantic_words = {candidate_words[i].lower() for i in top_indices if sims[i] > 0.35}

                for w in top_semantic_words:
                    pattern = re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
                    highlighted = pattern.sub(
                        lambda m: f"<mark style='background:lightgreen;font-weight:bold;'>{m.group(0)}</mark>",
                        highlighted
                    )
            except Exception as e:
                print(f"[WARN] Semantic highlighting failed: {e}")

        return highlighted

    # ------------------------------------------------------------------
    def _assign_roles_from_filename(self, filename):
        if not filename:
            return ["staff", "legal", "admin"]
        if "restricted" in filename.lower():
            return ["legal", "admin"]
        return ["staff", "legal", "admin"]
