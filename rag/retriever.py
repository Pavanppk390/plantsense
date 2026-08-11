"""
PlantSense — Retriever
Queries the Chroma vector store for the most semantically similar
maintenance logs given a natural-language query.
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Anchor paths to the repo root (this file lives in rag/, so parent.parent
# is the repo root) instead of the current working directory — this makes
# the app work correctly no matter where it's launched from (Colab,
# Streamlit Cloud, a different machine, etc.)
REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(REPO_ROOT / "rag" / "chroma_db")
LOGS_PATH = REPO_ROOT / "data" / "maintenance_logs_full.json"
COLLECTION_NAME = "maintenance_logs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load once, reuse across queries — loading the model per-call would be slow
_embedder = SentenceTransformer(EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

try:
    _collection = _client.get_collection(COLLECTION_NAME)
except Exception:
    # chroma_db/ is gitignored (derived data) — if it doesn't exist yet
    # (e.g. first run on a fresh deployment), build it automatically from
    # the committed maintenance_logs_full.json instead of crashing.
    import json

    with open(LOGS_PATH) as f:
        _logs = json.load(f)

    def _build_embedding_text(log: dict) -> str:
        return (
            f"Unit: {log['unit']}\n"
            f"Issue: {log['reported_issue']}\n"
            f"Root cause: {log['root_cause']}"
        )

    _texts = [_build_embedding_text(log) for log in _logs]
    _embeddings = _embedder.encode(_texts).tolist()

    _collection = _client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "PlantSense synthetic maintenance logs"},
    )
    _collection.add(
        ids=[log["log_id"] for log in _logs],
        embeddings=_embeddings,
        documents=_texts,
        metadatas=[
            {
                "log_id": log["log_id"],
                "unit": log["unit"],
                "date": log["date"],
                "subsystem": log["subsystem"],
                "action_taken": log["action_taken"],
                "resolution_time": log["resolution_time"],
            }
            for log in _logs
        ],
    )


def retrieve_similar_incidents(query: str, top_k: int = 3) -> list:
    """
    Given a natural-language query, returns the top_k most semantically
    similar maintenance logs, including their metadata.
    """
    query_embedding = _embedder.encode([query]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    # Reshape Chroma's output into a cleaner list of dicts
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "log_id": results["ids"][0][i],
            "distance": results["distances"][0][i],  # lower = more similar
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return retrieved


def print_results(query: str, results: list):
    print(f"\nQuery: \"{query}\"")
    print("=" * 60)
    for r in results:
        print(f"[{r['log_id']}] (distance: {r['distance']:.3f}, subsystem: {r['metadata']['subsystem']})")
        print(r["document"])
        print(f"Action taken: {r['metadata']['action_taken']}")
        print("-" * 60)


if __name__ == "__main__":
    test_queries = [
        "vibration issue on a bearing",
        "compressor efficiency dropping over time",
        "fan blade damage",
        "unexpected engine shutdown",
    ]

    for q in test_queries:
        results = retrieve_similar_incidents(q, top_k=3)
        print_results(q, results)
