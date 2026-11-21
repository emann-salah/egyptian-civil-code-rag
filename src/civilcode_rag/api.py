import logging
import os
from typing import Annotated

import chromadb
import groq
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, StringConstraints
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "civil_code_articles"
EMBED_MODEL = "intfloat/multilingual-e5-base"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
TOP_K = 3

SYSTEM_PROMPT = (
    "أنت مساعد قانوني. أجب فقط اعتماداً على المواد المرفقة، واذكر رقم المادة. "
    "إذا لم تكفِ المواد للإجابة قل: لا أجد في المواد المرفقة إجابة."
)


class Retriever:
    def __init__(self, chroma_path: str = CHROMA_PATH, top_k: int = TOP_K):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        self.top_k = top_k

    def retrieve(self, question: str) -> tuple[list[str], list[str]]:
        embedding = self.model.encode(f"query: {question}").tolist()
        results = self.collection.query(
            query_embeddings=[embedding], n_results=self.top_k
        )
        documents = results["documents"][0]
        # Sources come from retrieval metadata, never from the LLM.
        sources = [m["citation"] for m in results["metadatas"][0]]
        return documents, sources


class LLMClient:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        key = os.getenv("GROQ_API_KEY", "").strip()
        self.client = groq.Groq(api_key=key) if key else None

    def _require_client(self) -> groq.Groq:
        if self.client is None:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY is not set on the server",
            )
        return self.client

    def _map_error(self, e: groq.APIError) -> HTTPException:
        if isinstance(e, groq.AuthenticationError):
            logger.error("Groq rejected the API key")
            return HTTPException(503, "LLM auth failed: check GROQ_API_KEY")
        if isinstance(e, groq.NotFoundError):
            logger.error("Groq model not found: %s", self.model)
            return HTTPException(503, f"LLM model not found: {self.model}")
        if isinstance(e, groq.RateLimitError):
            logger.warning("Groq rate limit hit")
            return HTTPException(429, "LLM rate limit, retry shortly")
        logger.exception("Groq API error")
        return HTTPException(503, "LLM service unavailable, retry shortly")

    def complete(self, messages: list[dict[str, str]]) -> str:
        client = self._require_client()
        try:
            r = client.chat.completions.create(
                model=self.model, temperature=0.1, messages=messages
            )
        except groq.APIError as e:
            raise self._map_error(e) from e
        return r.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, str]]):
        client = self._require_client()
        try:
            return client.chat.completions.create(
                model=self.model, temperature=0.1, messages=messages, stream=True
            )
        except groq.APIError as e:
            raise self._map_error(e) from e


class AskRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ]


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


def build_messages(
    question: str, documents: list[str], sources: list[str]
) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"{source}:\n{document}" for source, document in zip(sources, documents)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"المواد:\n{context}\n\nالسؤال: {question}"},
    ]


app = FastAPI()
retriever = Retriever()
llm = LLMClient()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_indexed": retriever.collection.count(),
        "llm_configured": llm.client is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    documents, sources = retriever.retrieve(request.question)
    answer = llm.complete(build_messages(request.question, documents, sources))
    return AskResponse(answer=answer, sources=sources)


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    documents, sources = retriever.retrieve(request.question)
    stream = llm.stream(build_messages(request.question, documents, sources))

    def tokens():
        try:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            yield "\n\nSources: " + ", ".join(sources)
        except groq.APIError:
            logger.exception("Groq streaming failed")
            yield "\n\n[Error: LLM streaming failed.]"

    return StreamingResponse(tokens(), media_type="text/plain; charset=utf-8")