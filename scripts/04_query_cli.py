#!/usr/bin/env python3
"""04_query_cli.py
Simple CLI to run a query against the built indices.
"""

import os
import argparse
from app.retrieval import Retriever

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--roles", nargs="+", default=["staff"])
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    BM25_PATH = os.environ.get("BM25_PATH", "data/idx/bm25.pkl")
    FAISS_PATH = os.environ.get("FAISS_PATH", "data/idx/mE5.faiss")
    META_PATH = os.environ.get("META_PATH", "data/idx/meta.json")
    MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    retriever = Retriever(BM25_PATH, FAISS_PATH, META_PATH, MODEL_NAME)
    out = retriever.search(args.query, args.roles, topk=args.topk)
    print("\nANSWER:\n", out["answer"])
    print("\nRESULTS:")
    for r in out["results"]:
        print(f'- {r["doc_id"]} | Article {r["article_no"]} | pages {r["page_start"]}-{r["page_end"]} | score={r["score"]}')
        print("  excerpt:", r["excerpt"][:300].replace("\n", " "), "\n")

if __name__ == '__main__':
    main()
