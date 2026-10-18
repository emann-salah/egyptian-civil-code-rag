import os
from typing import Annotated

import chromadb
import groq
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, StringConstraints
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "civil_code_articles"
LLM_MODEL = "qwen/qwen3.8-27b"

SYSTEM_PROMPT = (
    "أنت مساعد قانوني. أجب فقط اعتماداً على المواد المرفقة، واذكر رقم المادة. "
    "إذا لم تكفِ المواد للإجابة قل: لا أجد في المواد المرفقة إجابة."
)

app = FastAPI()

model = SentenceTransformer("intfloat/multilingual-e5-base")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)
llm = groq.Groq(api_key=os.environ["GROQ_API_KEY"])


class AskRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


def retrieve(question: str) -> tuple[list[str], list[str]]:
    query_embedding = model.encode(f"query: {question}").tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Sources come from retrieval metadata, never from the LLM.
    sources = [metadata["citation"] for metadata in metadatas]

    return documents, sources


def build_messages(
    question: str,
    documents: list[str],
    sources: list[str],
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"{source}:\n{document}"
        for source, document in zip(sources, documents)
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"المواد:\n{context}\n\nالسؤال: {question}",
        },
    ]


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_indexed": collection.count(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    documents, sources = retrieve(request.question)

    try:
        response = llm.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.1,
            messages=build_messages(
                request.question,
                documents,
                sources,
            ),
        )
    except groq.APIError as e:
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable, retry shortly",
        ) from e

    answer = response.choices[0].message.content or ""

    return AskResponse(
        answer=answer,
        sources=sources,
    )


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    documents, sources = retrieve(request.question)

    try:
        stream = llm.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.1,
            messages=build_messages(
                request.question,
                documents,
                sources,
            ),
            stream=True,
        )
    except groq.APIError as e:
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable, retry shortly",
        ) from e

    def tokens():
        try:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            yield "\n\nSources: " + ", ".join(sources)

        except groq.APIError:
            yield "\n\n[Error: LLM streaming failed.]"

    return StreamingResponse(
        tokens(),
        media_type="text/plain; charset=utf-8",
    )