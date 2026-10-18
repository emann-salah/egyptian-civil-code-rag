

# Egyptian Civil Code — Bilingual Semantic Search & LLM API

A bilingual Arabic-English semantic search and question-answering system for the Egyptian Civil Code (Law No. 131 of 1948).

The project combines legal document processing, multilingual embeddings, vector search, and a FastAPI service to retrieve relevant legal articles and generate context-grounded answers using a Groq-hosted large language model.

The system is designed to provide article-level citations derived from retrieved document metadata, making legal information easier to search and explore in both Arabic and English.

## Features

- **Bilingual Legal Search:** Retrieve relevant legal articles using Arabic or English questions.
- **PDF Ingestion:** Extract and process articles from a bilingual, two-column legal PDF.
- **Arabic Text Processing:** Normalize Arabic text and handle right-to-left extraction, numbering, and paragraph structure.
- **Corpus Validation:** Validate article records, detect duplicates, and identify missing or incomplete records.
- **Reproducible Data Pipeline:** Use DVC to orchestrate corpus extraction, validation, and indexing.
- **Multilingual Embeddings:** Encode queries using `intfloat/multilingual-e5-base`.
- **Vector Search:** Store and retrieve legal articles using ChromaDB.
- **LLM-Powered Answers:** Use the Groq API to generate answers based on retrieved legal context.
- **Article-Level Sources:** Build the API's source list from retrieval metadata rather than relying on the LLM to generate citation identifiers.
- **REST API:** Expose health-check, question-answering, and streaming endpoints through FastAPI.
- **Experiment Tracking:** Compare embedding configurations and track experiments with MLflow.
- **Automated Testing and CI:** Use pytest, Ruff, and GitHub Actions.
- **Containerization:** Run the service with Docker Compose.
- **Load Testing:** Use Locust to measure API performance under concurrent workloads.

## Architecture

```text
Bilingual Egyptian Civil Code PDF
                |
                v
         PDF Extraction
                |
                v
      Structured Article Corpus
                |
                v
       Corpus Validation
                |
                v
          DVC Pipeline
                |
                v
    Multilingual Embeddings
                |
                v
             ChromaDB
                |
                v
            FastAPI
                |
          User Question
                |
                v
     Retrieve Top 3 Articles
                |
                v
       Groq-hosted LLM
                |
                v
      Answer + Source Metadata
```

The retrieval component supplies relevant article text and metadata to the LLM as context. The API derives its source list from the retrieved records, keeping citation identifiers separate from the model-generated answer.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| PDF Processing | pdfplumber |
| Data Validation | Pydantic, pytest |
| Data Pipeline | DVC |
| Embeddings | `intfloat/multilingual-e5-base` |
| Vector Database | ChromaDB |
| API Framework | FastAPI, Uvicorn |
| LLM Integration | Groq API |
| Experiment Tracking | MLflow |
| Containerization | Docker, Docker Compose |
| Load Testing | Locust |
| Code Quality | Ruff |
| CI | GitHub Actions |

## Repository Structure

```text
src/civilcode_rag/
  schemas.py            Article data models
  api.py                FastAPI endpoints and LLM integration
  build_index.py        Embedding generation and ChromaDB indexing
  run_experiments.py    Embedding model comparison
  eval_questions.py     Retrieval evaluation questions
  ingest/
    arabic_utils.py     Arabic text normalization
    extract_pdf.py      PDF extraction and article processing

tests/
  test_corpus.py        Corpus validation tests

dvc.yaml                Data pipeline configuration
locustfile.py           Load-testing configuration
docker-compose.yml      Container orchestration
reports/                Experiment and load-test reports
```

## Getting Started

### Requirements

- Python 3.12
- Docker Desktop with Docker Compose
- The source PDF at the path configured in the DVC pipeline
- A Groq API key for LLM-powered question answering

### 1. Clone the Repository

```bash
git clone https://github.com/emann-salah/egyptian-civil-code-rag.git
cd egyptian-civil-code-rag
```

### 2. Create the Python Environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -e . dvc pytest
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e . dvc pytest
```

### 3. Build the Corpus and Vector Index

```bash
dvc repro
```

This runs the configured data pipeline to extract the source document, validate the article corpus, and build the vector index.

The first run may take longer because the embedding model needs to be downloaded. Ensure that the source PDF is available at the configured path.

### 4. Configure the Groq API

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Keep this file out of version control and never commit API keys or other secrets.

### 5. Start the Application

```bash
docker compose up --build
```

Once the application has started, the API endpoints are available at:

- Health check: http://localhost:8000/health
- Interactive API documentation: http://localhost:8000/docs

## API Endpoints

### Health Check

```http
GET /health
```

Returns the service status and the number of documents indexed in ChromaDB.

Example response:

```json
{
  "status": "healthy",
  "documents_indexed": 1148
}
```

The document count is an example based on the recorded corpus configuration; the actual count depends on the generated index.

### Ask a Legal Question

```http
POST /ask
Content-Type: application/json
```

Example request:

```json
{
  "question": "هل يجوز للقاضي أن يحكم إذا لم يوجد نص قانوني ينطبق؟"
}
```

Example response structure:

```json
{
  "answer": "Generated answer based on the retrieved legal context.",
  "sources": [
    "Egyptian Civil Code, Article 1",
    "Egyptian Civil Code, Article 2",
    "Egyptian Civil Code, Article 3"
  ]
}
```

The response illustrates the API format, not an actual answer or a guarantee that these specific articles will be retrieved.

The endpoint retrieves the top three articles using multilingual embeddings, supplies their text to the LLM, and returns the generated answer with source identifiers derived from retrieval metadata.

### Streaming Answers

```http
POST /ask/stream
Content-Type: application/json
```

The streaming endpoint returns generated answer text incrementally and appends the retrieved source identifiers after generation.

The endpoint uses FastAPI's `StreamingResponse` to stream text to the client.

## Corpus and Data Quality

The source document is a bilingual, two-column PDF, with English on the left and Arabic on the right.

The ingestion pipeline includes:

- Right-to-left Arabic text extraction and normalization.
- Restoration of digit sequences and paragraph numbering.
- Expansion of repealed article ranges into individual records.
- Validation of article numbers and duplicate records.
- Detection of empty text and documented article coverage gaps.

The documented corpus covers 1,148 of 1,149 article numbers. Article 1022 is missing, and 12 records are documented as lacking Arabic text.

The source edition and amendment status should be verified before using the corpus for current legal research.

## Embedding Evaluation

The project compares three embedding configurations using Hit Rate@3 on a manually curated set of 17 questions.

| Embedding Configuration | Hit Rate@3 |
|---|---:|
| `paraphrase-multilingual-MiniLM-L12-v2` | 47.06% (8/17) |
| `multilingual-e5-base` with query/passage prefixes | 82.35% (14/17) |
| `multilingual-e5-base` without prefixes | 88.24% (15/17) |

Hit Rate@3 measures how often the expected article appears among the top three retrieved results.

The best observed result was 88.24% for `multilingual-e5-base` without prefixes on this evaluation set. Since the dataset contains only 17 questions, the results should be interpreted as an initial comparison rather than a definitive benchmark.

MLflow is used to track experiments and model configurations.

## Performance Testing

The project includes a Locust workload for the `/ask` endpoint.

A previously recorded test used 50 concurrent users over 60 seconds in a Docker environment running on a laptop CPU.

| Metric | Recorded Result |
|---|---:|
| Requests | 561 |
| p50 latency | 2.6 s |
| p95 latency | 3.9 s |
| p99 latency | 7.1 s |

These measurements describe the recorded test environment and workload. They are not a production capacity guarantee, and performance may differ when external LLM inference is involved.

## Testing and Continuous Integration

The project includes:

- Corpus validation tests using pytest.
- Ruff linting and pytest checks in GitHub Actions.
- DVC stages for extraction, validation, and indexing.
- A Locust workload for API load testing.

To run the data pipeline and corpus tests:

```bash
dvc repro
pytest -v
```

The corpus tests depend on the generated article corpus being available.

## Design Considerations

- **Retrieval-grounded generation:** The LLM receives retrieved article text as context.
- **Metadata-derived sources:** Source identifiers come from retrieved records rather than model-generated identifiers.
- **Reproducibility:** DVC manages the data pipeline, while MLflow tracks embedding experiments.
- **Separation of concerns:** Ingestion, indexing, API serving, and evaluation are organized into separate modules.
- **Bilingual processing:** Arabic text normalization and multilingual embeddings support search across Arabic and English queries.
