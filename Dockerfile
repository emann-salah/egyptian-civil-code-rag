FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --default-timeout=120 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -e .

COPY data/processed/articles.json data/processed/articles.json
COPY data/chroma_db/ data/chroma_db/

EXPOSE 8000

CMD ["uvicorn", "civilcode_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]