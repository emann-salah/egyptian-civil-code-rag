import json

import chromadb
import mlflow
from sentence_transformers import SentenceTransformer

from civilcode_rag.eval_questions import EVAL_QUESTIONS

ARTICLES_PATH = "data/processed/articles.json"
COLLECTION_NAME_PREFIX = "exp"


def build_temp_collection(
    model_name: str,
    use_e5_prefix: bool,
):
    """Build a fresh in-memory Chroma collection for one experiment run."""
    with open(ARTICLES_PATH, encoding="utf-8") as file:
        articles = json.load(file)

    model = SentenceTransformer(model_name)

    client = chromadb.Client()

    collection_name = (
        f"{COLLECTION_NAME_PREFIX}_"
        f"{model_name.replace('/', '_')}_"
        f"{use_e5_prefix}"
    )

    collection = client.create_collection(name=collection_name)

    texts = [article["text_ar"] for article in articles]

    if use_e5_prefix:
        passage_texts = [f"passage: {text}" for text in texts]
    else:
        passage_texts = texts

    ids = [str(article["article_number"]) for article in articles]

    metadatas = [
        {"article_number": article["article_number"]}
        for article in articles
    ]

    embeddings = model.encode(
        passage_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return model, collection


def evaluate(
    model,
    collection,
    use_e5_prefix: bool,
    k: int = 3,
) -> float:
    """
    Calculate Hit Rate @ k.

    Hit Rate @ k = fraction of evaluation questions for which
    the expected article appears among the top-k retrieved results.
    """
    if not EVAL_QUESTIONS:
        raise ValueError("EVAL_QUESTIONS must not be empty.")

    hits = 0

    for item in EVAL_QUESTIONS:
        question = item["question"]

        if use_e5_prefix:
            question = f"query: {question}"

        query_embedding = model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
        )

        retrieved_numbers = [
            metadata["article_number"]
            for metadata in results["metadatas"][0]
        ]

        if item["expected_article"] in retrieved_numbers:
            hits += 1

    return hits / len(EVAL_QUESTIONS)


def run_experiment(
    model_name: str,
    use_e5_prefix: bool = False,
) -> tuple[str, float]:
    """Run one embedding experiment and log its results to MLflow."""
    run_name = f"{model_name}|prefix={use_e5_prefix}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("embedding_model", model_name)
        mlflow.log_param("uses_e5_prefix", use_e5_prefix)
        mlflow.log_param("k", 3)
        mlflow.log_param("n_eval_questions", len(EVAL_QUESTIONS))

        model, collection = build_temp_collection(
            model_name=model_name,
            use_e5_prefix=use_e5_prefix,
        )

        hit_rate = evaluate(
            model=model,
            collection=collection,
            use_e5_prefix=use_e5_prefix,
            k=3,
        )

        mlflow.log_metric("hit_rate_at_3", hit_rate)

        print(
            f"[{model_name}] "
            f"prefix={use_e5_prefix} "
            f"hit_rate@3 = {hit_rate:.2%}"
        )

        return run.info.run_id, hit_rate


if __name__ == "__main__":
    experiments = [
        ("paraphrase-multilingual-MiniLM-L12-v2", False),
        ("intfloat/multilingual-e5-base", True),
        ("intfloat/multilingual-e5-base", False),
    ]

    results = [
        run_experiment(model_name, use_e5_prefix)
        for model_name, use_e5_prefix in experiments
    ]

    # Select the experiment with the highest Hit Rate @ 3.
    best_run_id, best_score = max(
        results,
        key=lambda result: result[1],
    )

    client = mlflow.MlflowClient()
    model_registry_name = "civilcode-embedding-config"

    try:
        client.create_registered_model(model_registry_name)
    except mlflow.exceptions.MlflowException:
        # The registered model may already exist.
        pass

    version = client.create_model_version(
        name=model_registry_name,
        source=f"runs:/{best_run_id}/",
        run_id=best_run_id,
        description=f"Best config: hit_rate@3={best_score:.2%}",
    )

    client.set_registered_model_alias(
        model_registry_name,
        "production",
        version.version,
    )

    print(
        f"[registry] Registered {model_registry_name} "
        f"v{version.version} "
        f"(hit_rate@3={best_score:.2%})"
    )
