import os
import re
import json
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from app.normalize import normalize_text


class Retriever:
    def __init__(self, bm25_path, faiss_path, meta_path, alpha=0.3):
        """
        🚀 Universal Hybrid Retriever (BM25 + FAISS)
        Combines lexical + semantic similarity with distinct color highlights:
          🟡 BM25 → exact keyword match
          🟢 FAISS → semantically similar (non-literal)
        """
        self.alpha = alpha
        self.faiss_topk = 50
        self.semantic_threshold = 0.35
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        print("⚙️ Initializing Universal Hybrid Retriever (BM25 + FAISS)...")

        # --- Load SentenceTransformer model ---
        self.model = SentenceTransformer(self.model_name)

        # --- Load BM25 index ---
        with open(bm25_path, "rb") as f:
            bm25_obj = pickle.load(f)
        self.bm25 = bm25_obj["bm25"]
        self.meta = bm25_obj["meta"]

        # --- Load FAISS index ---
        self.index = faiss.read_index(faiss_path)

        # --- Load metadata ---
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta_json = json.load(f)

    # ------------------------------------------------------------------
    def _normalize_top(self, scores, top_n=10):
        """Scale BM25 scores relative to top N values for stability."""
        scores = np.array(scores)
        if len(scores) == 0:
            return np.zeros_like(scores)
        top_mean = np.mean(np.sort(scores)[-top_n:])
        if top_mean == 0:
            return np.zeros_like(scores)
        return np.clip(scores / (top_mean + 1e-8), 0, 1)

    def _normalize_01(self, scores):
        """Normalize array to [0, 1] range."""
        scores = np.array(scores)
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # ------------------------------------------------------------------
    def search(self, query, roles, topk=5):
        """
        Perform hybrid retrieval:
          - BM25 (keyword-based)
          - FAISS (semantic-based)
          - Blended + role filtered
        """
        q_norm = normalize_text(query)
        tokens = q_norm.split()
        query_lower = query.lower().strip()

        # --- BM25 keyword scoring ---
        bm25_scores = np.array(self.bm25.get_scores(tokens))
        bm25_norm = self._normalize_top(bm25_scores)

        # --- Semantic scoring ---
        q_emb = self.model.encode([q_norm], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q_emb.astype("float32"), self.faiss_topk)

        vec_scores = np.zeros(len(self.meta_json))
        for idx, score in zip(I[0], D[0]):
            if score >= self.semantic_threshold:
                vec_scores[idx] = score
        vec_norm = self._normalize_01(vec_scores)

        # --- Weighted fusion ---
        fused = self.alpha * vec_norm + (1 - self.alpha) * bm25_norm

        # --- Penalize unrelated chunks ---
        for i, chunk in enumerate(self.meta_json):
            text = chunk["norm_text"]
            overlap = sum(tok in text for tok in tokens)
            if overlap == 0:
                fused[i] -= 0.2
            if query_lower in text:
                fused[i] += 0.2

        # --- Sort by final fusion score ---
        ranked_idx = np.argsort(-fused)
        results = []

        for i in ranked_idx:
            if fused[i] < 0.05:
                continue

            chunk = self.meta_json[i]
            chunk_roles = chunk.get("roles", self._assign_roles_from_filename(chunk.get("doc_id", "")))

            # RBAC
            if not set(roles).intersection(set(chunk_roles)):
                continue

            # Highlight both BM25 + FAISS distinctly
            excerpt = self._highlight_keywords(
                chunk["text"][:700],
                bm25_tokens=tokens,
                faiss_query_emb=q_emb[0]
            )

            results.append({
                "doc_id": chunk["doc_id"],
                "article_no": chunk.get("article_no", "Unknown"),
                "page_start": chunk.get("page_start", 1),
                "page_end": chunk.get("page_end", 1),
                "score": round(float(fused[i]), 3),
                "roles": chunk_roles,
                "excerpt": excerpt
            })

            if len(results) >= topk:
                break

        # --- Fallback if no hybrid result ---
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
                    "article_no": chunk.get("article_no", "Unknown"),
                    "page_start": chunk.get("page_start", 1),
                    "page_end": chunk.get("page_end", 1),
                    "score": round(float(bm25_norm[i]), 3),
                    "roles": chunk.get("roles", []),
                    "excerpt": excerpt
                })

        # --- Return result ---
        return {
            "answer": results[0]["excerpt"] if results else "❌ No relevant section found.",
            "results": results
        }

    # ------------------------------------------------------------------
    def _highlight_keywords(self, text, bm25_tokens, faiss_query_emb=None):
        """
        🟡 BM25 → Yellow (exact keyword matches)
        🟢 FAISS → Green (semantically similar but NOT literal matches)
        """
        bm25_tokens_clean = {tok.lower() for tok in bm25_tokens if tok.strip()}
        highlighted = text

        # --- 1️⃣ BM25 keyword highlighting (Yellow) ---
        for tok in bm25_tokens_clean:
            pattern = re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE)
            highlighted = pattern.sub(
                lambda m: f"<mark style='background:yellow;font-weight:bold;'>{m.group(0)}</mark>",
                highlighted
            )

        # --- 2️⃣ FAISS semantic highlighting (Green, non-literal only) ---
        if faiss_query_emb is not None:
            try:
                # Extract all unique words from text
                words = list(set(re.findall(r"\b\w+\b", text)))
                if not words:
                    return highlighted

                # Exclude literal BM25 matches
                candidate_words = [w for w in words if w.lower() not in bm25_tokens_clean]
                if not candidate_words:
                    return highlighted

                # Encode candidates & compute cosine similarity
                word_embs = self.model.encode(candidate_words, convert_to_numpy=True, normalize_embeddings=True)
                sims = np.dot(word_embs, faiss_query_emb.T).flatten()

                # Top 8 semantically closest non-literal words
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
        """Assign roles dynamically from filename for RBAC."""
        if not filename:
            return ["staff", "legal", "admin"]
        if "restricted" in filename.lower():
            return ["legal", "admin"]
        return ["staff", "legal", "admin"]



