from typing import Annotated

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "civil_code_articles"

app = FastAPI()

model = SentenceTransformer("intfloat/multilingual-e5-base")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "healthy", "documents_indexed": collection.count()}


@app.post("/ask")
def ask(request: AskRequest) -> AskResponse:
    query_embedding = model.encode(f"query: {request.question}").tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    sources = [m["citation"] for m in metadatas]
    answer = documents[0] if documents else "لم يتم العثور على نص ذي صلة."

    return AskResponse(answer=answer, sources=sources)
