import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
import mlflow
from sentence_transformers import SentenceTransformer

from civilcode_rag.eval_questions import EVAL_QUESTIONS

ARTICLES_PATH = "data/processed/articles.json"
COLLECTION_NAME_PREFIX = "exp"
ARTIFACTS_DIR = Path("artifacts")
BEST_MODEL = "intfloat/multilingual-e5-base"
TOP_K = 3


@dataclass
class ExperimentConfig:
    name: str
    model_name: str = BEST_MODEL
    use_e5_prefix: bool = True
    chunk_strategy: str = "whole"
    chunk_size: int = 0
    chunk_overlap: int = 0


EXPERIMENTS = [
    ExperimentConfig("minilm_noprefix", "paraphrase-multilingual-MiniLM-L12-v2", False),
    ExperimentConfig("e5_prefix_whole", BEST_MODEL, True),
    ExperimentConfig("e5_noprefix_whole", BEST_MODEL, False),
    ExperimentConfig("e5_fixed_256_ov0", chunk_strategy="fixed", chunk_size=256, chunk_overlap=0),
    ExperimentConfig("e5_fixed_256_ov64", chunk_strategy="fixed", chunk_size=256, chunk_overlap=64),
    ExperimentConfig("e5_fixed_128_ov32", chunk_strategy="fixed", chunk_size=128, chunk_overlap=32),
    ExperimentConfig("e5_fixed_512_ov64", chunk_strategy="fixed", chunk_size=512, chunk_overlap=64),
    ExperimentConfig("e5_clause_384", chunk_strategy="clause", chunk_size=384),
]


def split_fixed(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(size - overlap, 1)
    pieces = []
    for i in range(0, len(words), step):
        window = words[i : i + size]
        if window:
            pieces.append(" ".join(window))
        if i + size >= len(words):
            break
    return pieces


def split_clause(text: str, max_words: int) -> list[str]:
    parts = re.split(r"\n+|(?<=\s)(?=[\d٠-٩]+\s*[-–)])", text)
    parts = [p.strip() for p in parts if p.strip()]
    chunks, current, count = [], [], 0
    for p in parts:
        n = len(p.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(p)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def make_pieces(text: str, cfg: ExperimentConfig) -> list[str]:
    if cfg.chunk_strategy == "fixed":
        return split_fixed(text, cfg.chunk_size, cfg.chunk_overlap)
    if cfg.chunk_strategy == "clause":
        return split_clause(text, cfg.chunk_size)
    return [text]


def build_temp_collection(cfg: ExperimentConfig):
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        articles = json.load(f)

    model = SentenceTransformer(cfg.model_name)
    client = chromadb.Client()
    collection = client.create_collection(
        name=f"{COLLECTION_NAME_PREFIX}_{cfg.name}_{int(time.time())}"
    )

    ids, texts, metadatas = [], [], []
    for a in articles:
        for i, piece in enumerate(make_pieces(a["text_ar"], cfg)):
            ids.append(f"{a['article_number']}_{i}")
            texts.append(piece)
            metadatas.append({"article_number": a["article_number"], "chunk_index": i})

    to_embed = [f"passage: {t}" for t in texts] if cfg.use_e5_prefix else texts
    embeddings = model.encode(
        to_embed, show_progress_bar=True, convert_to_numpy=True
    ).tolist()
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    words = [len(t.split()) for t in texts]
    stats = {"n_chunks": len(texts), "avg_chunk_words": sum(words) / len(words)}
    return model, collection, stats


def evaluate(model, collection, cfg: ExperimentConfig, k: int = TOP_K):
    if not EVAL_QUESTIONS:
        raise ValueError("EVAL_QUESTIONS must not be empty.")

    hits, rows = 0, []
    fetch = min(k * 5, collection.count())

    for item in EVAL_QUESTIONS:
        q = f"query: {item['question']}" if cfg.use_e5_prefix else item["question"]
        emb = model.encode(q).tolist()
        res = collection.query(query_embeddings=[emb], n_results=fetch)

        retrieved = []
        for m in res["metadatas"][0]:
            n = m["article_number"]
            if n not in retrieved:
                retrieved.append(n)
            if len(retrieved) == k:
                break

        hit = item["expected_article"] in retrieved
        hits += int(hit)
        rows.append(
            {
                "question": item["question"],
                "expected_article": item["expected_article"],
                "retrieved_articles": retrieved,
                "hit": hit,
            }
        )
    return hits / len(EVAL_QUESTIONS), rows


def run_experiment(cfg: ExperimentConfig) -> tuple[str, float]:
    with mlflow.start_run(run_name=cfg.name) as run:
        mlflow.log_params(
            {
                "embedding_model": cfg.model_name,
                "uses_e5_prefix": cfg.use_e5_prefix,
                "chunk_strategy": cfg.chunk_strategy,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "top_k": TOP_K,
                "n_eval_questions": len(EVAL_QUESTIONS),
            }
        )

        t0 = time.time()
        model, collection, stats = build_temp_collection(cfg)
        build_seconds = time.time() - t0

        hit_rate, rows = evaluate(model, collection, cfg)

        mlflow.log_metric("hit_rate_at_3", hit_rate)
        mlflow.log_metric("n_chunks", stats["n_chunks"])
        mlflow.log_metric("avg_chunk_words", stats["avg_chunk_words"])
        mlflow.log_metric("build_seconds", build_seconds)

        ARTIFACTS_DIR.mkdir(exist_ok=True)
        results_path = ARTIFACTS_DIR / f"{cfg.name}_results.json"
        config_path = ARTIFACTS_DIR / f"{cfg.name}_config.json"
        results_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        config_path.write_text(
            json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(config_path))

        print(f"[{cfg.name}] hit_rate@3 = {hit_rate:.2%} | chunks = {stats['n_chunks']}")
        return run.info.run_id, hit_rate


if __name__ == "__main__":
    mlflow.set_experiment("civilcode-chunking-and-embeddings")

    results = [run_experiment(cfg) for cfg in EXPERIMENTS]
    best_run_id, best_score = max(results, key=lambda r: r[1])

    client = mlflow.MlflowClient()
    model_registry_name = "civilcode-embedding-config"

    try:
        client.create_registered_model(model_registry_name)
    except mlflow.exceptions.MlflowException:
        pass

    version = client.create_model_version(
        name=model_registry_name,
        source=f"runs:/{best_run_id}/",
        run_id=best_run_id,
        description=f"Best config: hit_rate@3={best_score:.2%}",
    )
    client.set_registered_model_alias(model_registry_name, "production", version.version)

    print(
        f"[registry] Registered {model_registry_name} "
        f"v{version.version} (hit_rate@3={best_score:.2%})"
    )