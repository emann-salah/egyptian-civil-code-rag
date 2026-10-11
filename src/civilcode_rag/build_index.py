import json

import chromadb
from sentence_transformers import SentenceTransformer

ARTICLES_PATH = "data/processed/articles.json"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "civil_code_articles"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def build_index():
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        articles = json.load(f)
    print(f"[info] loaded {len(articles)} articles")

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = [str(a["article_number"]) for a in articles]
    documents = [a["text_ar"] for a in articles]  
    
    to_embed = [f"passage: {text}" for text in documents]
    metadatas = [
        {
            "article_number": a["article_number"],
            "citation": a["citation"],
            "is_repealed": a["is_repealed"],
        }
        for a in articles
    ]

    print("[info] computing embeddings...")
    embeddings = model.encode(to_embed, show_progress_bar=True).tolist()

    print("[info] writing to Chroma...")
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"[info] indexed {collection.count()} articles into {CHROMA_PATH}")


if __name__ == "__main__":
    build_index()