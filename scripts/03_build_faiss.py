import json, os, faiss, numpy as np
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
chunks_path = "data/processed/chunks.jsonl"

os.makedirs("data/idx", exist_ok=True)

def main():
    chunks = [json.loads(line) for line in open(chunks_path, "r", encoding="utf-8")]
    texts = [c["norm_text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, "data/idx/mE5.faiss")
    with open("data/idx/meta.json", "w") as f:
        json.dump(chunks, f)
    print("✅ FAISS index built and saved.")

if __name__ == "__main__":
    main()
