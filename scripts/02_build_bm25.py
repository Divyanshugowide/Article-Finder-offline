import json, pickle, os
from rank_bm25 import BM25Okapi
from app.normalize import normalize_text

chunks_path = "data/processed/chunks.jsonl"
out_path = "data/idx/bm25.pkl"

os.makedirs("data/idx", exist_ok=True)

def load_chunks():
    with open(chunks_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    chunks = load_chunks()
    corpus = [c["norm_text"].split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    pickle.dump({"bm25": bm25, "meta": chunks}, open(out_path, "wb"))
    print("✅ BM25 index saved to", out_path)

if __name__ == "__main__":
    main()
