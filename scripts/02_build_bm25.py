import os
import json
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from app.normalize import normalize_text

# Input / Output paths
chunks_path = "data/processed/chunks.jsonl"
out_dir = "data/idx"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "bm25.pkl")
vocab_path = os.path.join(out_dir, "bm25_vocab.txt")


def load_chunks():
    """Load chunk records from preprocessed JSONL."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_bm25():
    """Build and save BM25 index for text chunks."""
    print(f"⚙️ Loading chunks from {chunks_path} ...")
    chunks = load_chunks()
    print(f"📄 Loaded {len(chunks)} chunks.")

    corpus = []
    for c in tqdm(chunks, desc="🔹 Normalizing & tokenizing"):
        text = normalize_text(c["text"])
        tokens = [t for t in text.split() if len(t) > 2]
        corpus.append(tokens)

    print("🔍 Building BM25 index ...")
    bm25 = BM25Okapi(corpus)

    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "meta": chunks}, f)
    print(f"✅ BM25 index saved to {out_path}")

    # Optional — Save vocabulary for inspection
    vocab = sorted(set(word for tokens in corpus for word in tokens))
    with open(vocab_path, "w", encoding="utf-8") as vf:
        vf.write("\n".join(vocab))
    print(f"🧾 Vocabulary saved to {vocab_path}")


if __name__ == "__main__":
    try:
        build_bm25()
    except Exception as e:
        print("❌ BM25 build failed:", str(e))
