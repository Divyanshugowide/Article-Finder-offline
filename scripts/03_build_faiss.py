import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
model_name = "sentence-transformers/all-MiniLM-L6-v2"
chunks_path = "data/processed/chunks.jsonl"
out_dir = "data/idx"
os.makedirs(out_dir, exist_ok=True)
faiss_path = os.path.join(out_dir, "faiss.index")
meta_path = os.path.join(out_dir, "meta.json")

# ----------------------------------------------------------------------
def load_chunks():
    """Load preprocessed text chunks."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_faiss(batch_size=512):
    """Build FAISS vector index with batch embedding encoding."""
    print(f"⚙️ Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"📥 Loading chunks from {chunks_path}")
    chunks = load_chunks()
    texts = [c["norm_text"] for c in chunks]
    print(f"📄 Total chunks: {len(texts)}")

    # --- Compute embeddings in batches ---
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔹 Encoding embeddings"):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")

    # --- Build FAISS Index ---
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, faiss_path)
    print(f"✅ FAISS index saved to {faiss_path}")
    print(f"📊 Total vectors: {index.ntotal} | Dim: {d}")

    # --- Save metadata ---
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"🧾 Metadata saved to {meta_path}")


if __name__ == "__main__":
    try:
        build_faiss()
    except Exception as e:
        print("❌ FAISS build failed:", str(e))
