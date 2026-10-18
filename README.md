

# Egyptian Civil Code — Bilingual Semantic Search & LLM API

A bilingual Arabic-English semantic search system for the Egyptian Civil Code (Law No. 131 of 1948). The project combines document processing, multilingual embeddings, vector search, and a FastAPI service to retrieve relevant legal articles with article-level citations.

The project is being developed in stages, from reproducible legal document ingestion and semantic retrieval to LLM-powered answer generation.

## Project Status

**Current stage:** Semantic retrieval API with initial LLM integration.

- [x] PDF extraction and structured article corpus
- [x] Corpus validation with pytest
- [x] Reproducible pipeline with DVC
- [x] Multilingual embeddings and ChromaDB indexing
- [x] FastAPI endpoints for health checks and question answering
- [x] Embedding model comparison using MLflow
- [x] Docker Compose setup
- [x] Locust load-test configuration
- [x] GitHub Actions CI configuration
- [x] Initial Groq LLM integration in the API
- [ ] End-to-end validation of LLM-generated answers
- [ ] Streaming endpoint validation
- [ ] Further evaluation of answer grounding and citation quality

## Architecture

```text
Egyptian Civil Code PDF
          |
          v
    PDF Extraction
          |
          v
      articles.json
          |
          v
 Corpus Validation (pytest)
          |
          v
   DVC Pipeline
          |
          v
  multilingual-e5-base
          |
          v
       ChromaDB
          |
          v
       FastAPI
          |
          v
     User Question
          |
          v
   Top 3 Retrieved Articles
          |
          v
    Groq-hosted LLM
          |
          v
 Answer + Article Citations
```

The retrieval component supplies the article text and metadata to the LLM. The API derives its source list from the retrieved article metadata rather than asking the LLM to invent citation identifiers.

## Tech Stack

- **Language:** Python 3.12
- **Document processing:** pdfplumber and custom Arabic text normalization
- **Data validation:** Pydantic and pytest
- **Pipeline orchestration:** DVC
- **Embeddings:** `intfloat/multilingual-e5-base`
- **Vector database:** ChromaDB
- **API:** FastAPI and Uvicorn
- **LLM integration:** Groq API
- **Experiment tracking:** MLflow
- **Containerization:** Docker and Docker Compose
- **Load testing:** Locust
- **Code quality and CI:** Ruff, pytest, and GitHub Actions

## Repository Structure

```text
src/civilcode_rag/
  schemas.py            Article data models
  api.py                FastAPI and LLM integration
  build_index.py        Embedding generation and ChromaDB indexing
  run_experiments.py    Embedding model comparison
  eval_questions.py     Evaluation questions
  ingest/               PDF extraction and Arabic text processing

tests/
  test_corpus.py        Corpus validation tests

dvc.yaml                Data pipeline
locustfile.py           Load-test configuration
docker-compose.yml      Container orchestration
reports/                Experiment and load-test reports
```

## Quick Start

### Requirements

- Python 3.12
- Docker Desktop with Docker Compose
- Access to the source PDF
- A Groq API key for LLM-powered question answering

### Clone the repository

```bash
git clone https://github.com/emann-salah/egyptian-civil-code-rag.git
cd egyptian-civil-code-rag
```

### Set up the Python environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -e . dvc pytest
```

On Linux or macOS:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e . dvc pytest
```

### Build the corpus and vector index

```bash
dvc repro
```

The pipeline extracts the bilingual PDF, validates the article corpus, and builds the vector index. The first run downloads the embedding model and may take some time.

Ensure that the source PDF is available at the path configured by the pipeline.

### Configure the LLM

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Keep `.env` out of version control. Never commit API keys or other secrets.

### Start the API

```bash
docker compose up --build
```

Wait until the application reports that startup is complete.

The API is then available at:

- Health check: `http://localhost:8000/health`
- API documentation: `http://localhost:8000/docs`

## API Usage

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "documents_indexed": 1148
}
```

The document count reflects the current local index and may vary depending on the corpus and indexing configuration.

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
  "answer": "LLM-generated answer based on retrieved legal articles.",
  "sources": [
    "Egyptian Civil Code, Article 1",
    "Egyptian Civil Code, Article 2",
    "Egyptian Civil Code, Article 3"
  ]
}
```

The answer above illustrates the response structure; it is not an actual legal answer or a claim about which articles this question retrieves.

The API retrieves the top three articles using multilingual embeddings and supplies their text to the LLM as context. The `sources` field is derived from retrieval metadata.

### Streaming Answers

```http
POST /ask/stream
Content-Type: application/json
```

The streaming endpoint returns generated text incrementally and appends the retrieved source identifiers at the end of the response.

The endpoint is part of the initial LLM integration and remains subject to end-to-end validation.

## Corpus and Data Quality

The source document is a bilingual, two-column PDF, with English on the left and Arabic on the right.

The ingestion pipeline addresses:

- Right-to-left Arabic text extraction and normalization.
- Restoration of digit sequences and paragraph numbering.
- Expansion of repealed article ranges into individual records.
- Validation of article numbers, duplicate records, empty text, and documented coverage gaps.

The documented corpus coverage is 1,148 of 1,149 article numbers. Article 1022 is missing, and 12 records are documented as lacking Arabic text.

The source document's edition and amendment status should be verified before relying on the corpus for current legal advice.

## Embedding Evaluation

The project compares three embedding configurations using Hit Rate@3 on a hand-written set of 17 questions.

| Embedding configuration | Hit Rate@3 |
|---|---:|
| paraphrase-multilingual-MiniLM-L12-v2 | 47.06% (8/17) |
| multilingual-e5-base with query/passage prefixes | 82.35% (14/17) |
| multilingual-e5-base without prefixes | 88.24% (15/17) |

Hit Rate@3 measures how often the expected article appears among the three retrieved results. These scores measure retrieval performance on this small evaluation set, not legal answer accuracy.

The no-prefix configuration scored highest in this experiment, but the difference between the two E5 variants is based on only one question. More evaluation is needed before drawing strong conclusions.

Experiment tracking and model configuration registration are handled through MLflow.

## Performance Testing

The project includes a Locust load-test configuration for the `/ask` endpoint.

A previously recorded test used 50 concurrent users for 60 seconds in a Docker environment on a laptop CPU:

| Metric | Recorded result |
|---|---:|
| Requests | 561 |
| p50 latency | 2.6 s |
| p95 latency | 3.9 s |
| p99 latency | 7.1 s |

These figures describe the recorded test environment and workload; they should not be interpreted as a production capacity guarantee. The results should be re-measured after LLM integration because external model latency changes the request profile.

## Tests and CI

The project includes:

- Corpus validation tests with pytest.
- Ruff linting and pytest in GitHub Actions.
- DVC stages for extraction, validation, and indexing.
- A Locust workload for API performance testing.

The corpus tests require the generated article corpus, so run the DVC pipeline before executing the full corpus test suite.

```bash
dvc repro
pytest -v
```

## Limitations and Next Steps

- Expand the evaluation set with more Arabic and English questions.
- Validate generated answers against the retrieved legal text.
- Test citation relevance and factual grounding.
- Verify streaming behavior and API error handling.
- Address Arabic text normalization issues and evaluate their impact on retrieval.
- Reassess latency after LLM integration.
- Verify the source edition and legal amendment status.
- Add end-to-end tests for the complete retrieval-to-generation pipeline.

