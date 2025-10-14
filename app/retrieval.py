# import os
# import re
# import json
# import pickle
# import numpy as np
# import faiss
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer
# from app.normalize import normalize_text


# class Retriever:
#     def __init__(self, bm25_path, faiss_path, meta_path, model_name=None, alpha=0.2):
#         """
#         Hybrid retriever (BM25 + FAISS) with RBAC and exact-match filtering.
#         alpha -> weight for FAISS vector similarity (lower = stricter keyword match)
#         """
#         self.alpha = alpha

#         # --- Load Sentence Transformer model ---
#         self.model_name = model_name or os.getenv("MODEL_NAME", "models/all-MiniLM-L6-v2")
#         if not os.path.exists(self.model_name):
#             print(f"[WARN] Model path '{self.model_name}' not found — using default online model.")
#         self.model = SentenceTransformer(self.model_name)

#         # --- Load BM25 index ---
#         with open(bm25_path, "rb") as f:
#             bm25_obj = pickle.load(f)
#         self.bm25 = bm25_obj["bm25"]
#         self.meta = bm25_obj["meta"]

#         # --- Load FAISS index ---
#         self.index = faiss.read_index(faiss_path)

#         # --- Load metadata (chunk info with roles) ---
#         with open(meta_path, "r", encoding="utf-8") as f:
#             self.meta_json = json.load(f)

#     # ----------------------------------------------------------------------
#     def _normalize_scores(self, scores):
#         """Normalize array to [0, 1] range safely."""
#         scores = np.array(scores)
#         if len(scores) == 0 or scores.max() == scores.min():
#             return np.zeros_like(scores)
#         return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

#     # ----------------------------------------------------------------------
#     def search(self, query, roles, topk=5, mode="hybrid"):
#         """
#         Perform hybrid (BM25 + FAISS) search with RBAC filtering.
#         Modes:
#           - "hybrid"  : semantic + keyword (default)
#           - "exact"   : strict word match only
#           - "semantic": ignore BM25, use embeddings only
#         """
#         q_norm = normalize_text(query)
#         tokens = q_norm.split()

#         # --- BM25 scoring ---
#         bm25_scores = np.array(self.bm25.get_scores(tokens))

#         # --- FAISS vector scoring ---
#         q_emb = self.model.encode([q_norm], convert_to_numpy=True, normalize_embeddings=True)
#         D, I = self.index.search(q_emb.astype("float32"), 50)

#         vec_scores = np.zeros(len(self.meta_json))
#         for idx, score in zip(I[0], D[0]):
#             vec_scores[idx] = score

#         # --- Combine scores according to mode ---
#         if mode == "exact":
#             fused = bm25_scores
#         elif mode == "semantic":
#             fused = vec_scores
#         else:  # hybrid
#             bm25_norm = self._normalize_scores(bm25_scores)
#             vec_norm = self._normalize_scores(vec_scores)
#             fused = self.alpha * vec_norm + (1 - self.alpha) * bm25_norm

#         # --- Bonus boost for exact word presence ---
#         for i, chunk in enumerate(self.meta_json):
#             if query.lower() in chunk["norm_text"]:
#                 fused[i] += 0.3  # push literal matches higher

#         # --- Sort descending by fused score ---
#         ranked_idx = np.argsort(-fused)

#         results = []
#         query_lower = query.lower().strip()

#         for i in ranked_idx:
#             chunk = self.meta_json[i]
#             chunk_roles = chunk.get("roles", self._assign_roles_from_filename(chunk.get("doc_id", "")))

#             # --- RBAC filtering ---
#             if not set(roles).intersection(set(chunk_roles)):
#                 continue

#             # --- Enforce strict keyword presence for hybrid/exact modes ---
#             if mode in ("exact", "hybrid"):
#                 if not any(tok in chunk["norm_text"] for tok in tokens):
#                     continue

#             # --- Skip weak scores ---
#             if fused[i] <= 0.05:
#                 continue

#             # --- Highlight the keyword(s) ---
#             excerpt = self._highlight_keywords(chunk["text"][:500], tokens)

#             results.append({
#                 "doc_id": chunk["doc_id"],
#                 "article_no": chunk["article_no"],
#                 "page_start": chunk["page_start"],
#                 "page_end": chunk["page_end"],
#                 "score": round(float(fused[i]), 3),
#                 "roles": chunk_roles,
#                 "excerpt": excerpt
#             })

#             if len(results) >= topk:
#                 break

#         # --- Fallback: literal substring search if nothing found ---
#         if not results:
#             for chunk in self.meta_json:
#                 chunk_roles = chunk.get("roles", self._assign_roles_from_filename(chunk.get("doc_id", "")))
#                 if not set(roles).intersection(set(chunk_roles)):
#                     continue
#                 if query_lower in chunk["norm_text"]:
#                     excerpt = self._highlight_keywords(chunk["text"][:500], [query_lower])
#                     results.append({
#                         "doc_id": chunk["doc_id"],
#                         "article_no": chunk["article_no"],
#                         "page_start": chunk["page_start"],
#                         "page_end": chunk["page_end"],
#                         "score": 1.0,
#                         "roles": chunk_roles,
#                         "excerpt": excerpt
#                     })
#                     break

#         return {
#             "answer": results[0]["excerpt"] if results else "❌ No relevant article found.",
#             "results": results
#         }

#     # ----------------------------------------------------------------------
#     def _highlight_keywords(self, text, tokens):
#         """Highlight all matched tokens in yellow for UI."""
#         for tok in tokens:
#             if not tok.strip():
#                 continue
#             pattern = re.compile(re.escape(tok), re.IGNORECASE)
#             text = pattern.sub(lambda m: f"<mark style='background:yellow;font-weight:bold;'>{m.group(0)}</mark>", text)
#         return text

#     # ----------------------------------------------------------------------
#     def _assign_roles_from_filename(self, filename: str):
#         """Assign roles based on filename convention."""
#         if "restricted" in filename.lower():
#             return ["legal", "admin"]
#         return ["staff", "legal", "admin"]


# code 2 alpha = 0.7
# import os
# import re
# import json
# import pickle
# import numpy as np
# import faiss
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer
# from app.normalize import normalize_text


# class Retriever:
#     def __init__(self, bm25_path, faiss_path, meta_path, alpha=0.5):
#         """
#         ⚖️ Stable Mode — Balanced between semantic accuracy and keyword precision.
#         Uses BM25 for exact matches and FAISS for meaning-based matches.
#         alpha → controls balance between semantic (FAISS) and keyword (BM25)
#         """
#         self.alpha = alpha
#         self.faiss_topk = 50
#         self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

#         print("⚖️ Running in STABLE Hybrid Mode")

#         # Load Sentence Transformer model
#         if not os.path.exists(self.model_name):
#             print(f"[WARN] Model '{self.model_name}' not found locally — using default online model.")
#         self.model = SentenceTransformer(self.model_name)

#         # Load BM25 index
#         with open(bm25_path, "rb") as f:
#             bm25_obj = pickle.load(f)
#         self.bm25 = bm25_obj["bm25"]
#         self.meta = bm25_obj["meta"]

#         # Load FAISS index
#         self.index = faiss.read_index(faiss_path)

#         # Load metadata (chunk info with roles)
#         with open(meta_path, "r", encoding="utf-8") as f:
#             self.meta_json = json.load(f)

#     # ----------------------------------------------------------------------
#     def _normalize_scores(self, scores):
#         """Normalize scores safely to [0,1] range."""
#         scores = np.array(scores)
#         if len(scores) == 0 or scores.max() == scores.min():
#             return np.zeros_like(scores)
#         return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

#     # ----------------------------------------------------------------------
#     def search(self, query, roles, topk=5):
#         """
#         Perform hybrid BM25 + FAISS retrieval with RBAC filtering.
#         Steps:
#          1. Compute BM25 and FAISS scores.
#          2. Normalize and fuse them.
#          3. Filter results based on user roles.
#          4. Highlight matching keywords.
#         """
#         q_norm = normalize_text(query)
#         tokens = q_norm.split()

#         # --- BM25 Keyword Scoring ---
#         bm25_scores = np.array(self.bm25.get_scores(tokens))

#         # --- FAISS Semantic Scoring ---
#         q_emb = self.model.encode([q_norm], convert_to_numpy=True, normalize_embeddings=True)
#         D, I = self.index.search(q_emb.astype("float32"), self.faiss_topk)

#         vec_scores = np.zeros(len(self.meta_json))
#         for idx, score in zip(I[0], D[0]):
#             vec_scores[idx] = score

#         # --- Combine Scores ---
#         bm25_norm = self._normalize_scores(bm25_scores)
#         vec_norm = self._normalize_scores(vec_scores)
#         fused = self.alpha * vec_norm + (1 - self.alpha) * bm25_norm

#         # --- Boost exact text matches ---
#         for i, chunk in enumerate(self.meta_json):
#             if query.lower() in chunk["norm_text"]:
#                 fused[i] += 0.2  # boost for literal matches

#         # --- Rank results descending ---
#         ranked_idx = np.argsort(-fused)
#         results = []
#         query_lower = query.lower().strip()

#         for i in ranked_idx:
#             chunk = self.meta_json[i]
#             chunk_roles = chunk.get("roles", self._assign_roles_from_filename(chunk.get("doc_id", "")))

#             # Role-based filtering
#             if not set(roles).intersection(set(chunk_roles)):
#                 continue

#             # Skip low-confidence matches
#             if fused[i] < 0.05:
#                 continue

#             # Highlight matching keywords
#             excerpt = self._highlight_keywords(chunk["text"][:600], tokens)

#             results.append({
#                 "doc_id": chunk["doc_id"],
#                 "article_no": chunk["article_no"],
#                 "page_start": chunk["page_start"],
#                 "page_end": chunk["page_end"],
#                 "score": round(float(fused[i]), 3),
#                 "roles": chunk_roles,
#                 "excerpt": excerpt
#             })

#             if len(results) >= topk:
#                 break

#         # --- Fallback to BM25 if nothing found ---
#         if not results:
#             print("⚠️ No semantic results found, falling back to keyword search...")
#             for i in np.argsort(-bm25_norm)[:topk]:
#                 chunk = self.meta_json[i]
#                 if not set(roles).intersection(set(chunk["roles"])):
#                     continue
#                 excerpt = self._highlight_keywords(chunk["text"][:600], tokens)
#                 results.append({
#                     "doc_id": chunk["doc_id"],
#                     "article_no": chunk["article_no"],
#                     "page_start": chunk["page_start"],
#                     "page_end": chunk["page_end"],
#                     "score": round(float(bm25_norm[i]), 3),
#                     "roles": chunk["roles"],
#                     "excerpt": excerpt
#                 })

#         return {
#             "answer": results[0]["excerpt"] if results else "❌ No relevant article found.",
#             "results": results
#         }

#     # ----------------------------------------------------------------------
#     def _highlight_keywords(self, text, tokens):
#         """Highlight keywords in yellow for frontend display."""
#         for tok in tokens:
#             if not tok.strip():
#                 continue
#             pattern = re.compile(re.escape(tok), re.IGNORECASE)
#             text = pattern.sub(lambda m: f"<mark style='background:yellow;font-weight:bold;'>{m.group(0)}</mark>", text)
#         return text

#     # ----------------------------------------------------------------------
#     def _assign_roles_from_filename(self, filename: str):
#         """Assign RBAC roles based on file naming convention."""
#         if "restricted" in filename.lower():
#             return ["legal", "admin"]
#         return ["staff", "legal", "admin"]


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
    def __init__(self, bm25_path, faiss_path, meta_path, alpha=0.5):
        """
        ⚖️ Improved Stable Mode — Balanced hybrid retrieval.
        Fixes previous issue where semantic dominance caused wrong matches.
        alpha -> blend weight between FAISS (semantic) and BM25 (keyword)
        Lower alpha = stronger keyword influence.
        """
        self.alpha = alpha
        self.faiss_topk = 50
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        print("⚖️ Running Improved STABLE Hybrid Mode (Balanced + Keyword Aware)")

        # Load sentence transformer
        if not os.path.exists(self.model_name):
            print(f"[WARN] Model '{self.model_name}' not found locally — using default online model.")
        self.model = SentenceTransformer(self.model_name)

        # Load BM25
        with open(bm25_path, "rb") as f:
            bm25_obj = pickle.load(f)
        self.bm25 = bm25_obj["bm25"]
        self.meta = bm25_obj["meta"]

        # Load FAISS index
        self.index = faiss.read_index(faiss_path)

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta_json = json.load(f)

    # ------------------------------------------------------------------
    def _normalize_scores(self, scores):
        """Normalize array to [0,1] safely."""
        scores = np.array(scores)
        if len(scores) == 0 or scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # ------------------------------------------------------------------
    def search(self, query, roles, topk=5):
        """Perform balanced hybrid retrieval with role filtering and keyword boosting."""
        q_norm = normalize_text(query)
        tokens = q_norm.split()
        query_lower = query.lower().strip()

        # --- BM25 keyword scoring ---
        bm25_scores = np.array(self.bm25.get_scores(tokens))

        # --- FAISS semantic scoring ---
        q_emb = self.model.encode([q_norm], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q_emb.astype("float32"), self.faiss_topk)
        vec_scores = np.zeros(len(self.meta_json))
        for idx, score in zip(I[0], D[0]):
            vec_scores[idx] = score

        # --- Normalize both ---
        bm25_norm = self._normalize_scores(bm25_scores)
        vec_norm = self._normalize_scores(vec_scores)

        # --- Blend scores (Balanced Hybrid) ---
        fused = self.alpha * vec_norm + (1 - self.alpha) * bm25_norm

        # --- Extra boost for strong legal keywords ---
        important_keywords = ["criminal", "penalty", "inspection", "violation", "fine", "offence", "law", "license"]
        for i, chunk in enumerate(self.meta_json):
            for word in important_keywords:
                if word in chunk["norm_text"]:
                    fused[i] += 0.15  # boost relevant legal terms
            if query_lower in chunk["norm_text"]:
                fused[i] += 0.25  # strong literal match boost

        # --- Rank descending ---
        ranked_idx = np.argsort(-fused)
        results = []

        for i in ranked_idx:
            chunk = self.meta_json[i]
            chunk_roles = chunk.get("roles", self._assign_roles_from_filename(chunk.get("doc_id", "")))

            # RBAC filtering
            if not set(roles).intersection(set(chunk_roles)):
                continue

            # Skip weak matches
            if fused[i] < 0.05:
                continue

            # Highlight results
            excerpt = self._highlight_keywords(chunk["text"][:700], tokens)

            results.append({
                "doc_id": chunk["doc_id"],
                "article_no": chunk["article_no"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "score": round(float(fused[i]), 3),
                "roles": chunk_roles,
                "excerpt": excerpt
            })

            if len(results) >= topk:
                break

        # --- Fallback if no result ---
        if not results:
            print("⚠️ No semantic results — falling back to keyword search.")
            for i in np.argsort(-bm25_norm)[:topk]:
                chunk = self.meta_json[i]
                if not set(roles).intersection(set(chunk.get("roles", []))):
                    continue
                excerpt = self._highlight_keywords(chunk["text"][:700], tokens)
                results.append({
                    "doc_id": chunk["doc_id"],
                    "article_no": chunk["article_no"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "score": round(float(bm25_norm[i]), 3),
                    "roles": chunk.get("roles", []),
                    "excerpt": excerpt
                })

        return {
            "answer": results[0]["excerpt"] if results else "❌ No relevant article found.",
            "results": results
        }

    # ------------------------------------------------------------------
    def _highlight_keywords(self, text, tokens):
        """Highlight all query tokens in the text."""
        for tok in tokens:
            if not tok.strip():
                continue
            pattern = re.compile(re.escape(tok), re.IGNORECASE)
            text = pattern.sub(
                lambda m: f"<mark style='background:yellow;font-weight:bold;'>{m.group(0)}</mark>",
                text
            )
        return text

    # ------------------------------------------------------------------
    def _assign_roles_from_filename(self, filename: str):
        """Assign access roles based on filename convention."""
        if "restricted" in filename.lower():
            return ["legal", "admin"]
        return ["staff", "legal", "admin"]
