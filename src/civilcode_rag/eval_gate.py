import sys

import chromadb
from sentence_transformers import SentenceTransformer

from civilcode_rag.eval_questions import EVAL_QUESTIONS

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "civil_code_articles"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
TOP_K = 3
MIN_HIT_RATE = 0.75


def main() -> int:
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    hits = 0
    for item in EVAL_QUESTIONS:
        embedding = model.encode(f"query: {item['question']}").tolist()
        results = collection.query(query_embeddings=[embedding], n_results=TOP_K)
        retrieved = [m["article_number"] for m in results["metadatas"][0]]
        hits += int(item["expected_article"] in retrieved)

    hit_rate = hits / len(EVAL_QUESTIONS)
    print(f"hit_rate@{TOP_K} = {hit_rate:.2%} (threshold {MIN_HIT_RATE:.0%})")

    if hit_rate < MIN_HIT_RATE:
        print("Eval gate FAILED")
        return 1
    print("Eval gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())