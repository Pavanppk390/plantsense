"""
PlantSense — Embed Store
Embeds the maintenance log knowledge base and stores it in a persistent
Chroma vector database for retrieval by the RAG/agent layer.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

LOGS_PATH = "maintenance_logs_full.json"
CHROMA_PERSIST_DIR = "chroma_db"  # saved to disk, survives across runs
COLLECTION_NAME = "maintenance_logs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, strong general-purpose
                                       # sentence embedding model — good default


def build_embedding_text(log: dict) -> str:
    """
    Combines the semantically meaningful fields into one text block for
    embedding. Fields left out (action_taken, resolution_time, date, log_id)
    are kept as metadata instead — useful to retrieve, not useful to embed.
    """
    return (
        f"Unit: {log['unit']}\n"
        f"Issue: {log['reported_issue']}\n"
        f"Root cause: {log['root_cause']}"
    )


def main():
    print("Loading logs from", LOGS_PATH)
    with open(LOGS_PATH) as f:
        logs = json.load(f)
    print(f"Loaded {len(logs)} logs.")

    print("Loading embedding model:", EMBEDDING_MODEL)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Building embedding texts...")
    texts = [build_embedding_text(log) for log in logs]

    print("Computing embeddings for all logs (this may take ~10-30s)...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    print("Setting up Chroma persistent client...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Start clean each run — avoids duplicate entries if you rerun this script
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet, that's fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "PlantSense synthetic maintenance logs"},
    )

    print("Adding documents to Chroma collection...")
    collection.add(
        ids=[log["log_id"] for log in logs],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "log_id": log["log_id"],
                "unit": log["unit"],
                "date": log["date"],
                "subsystem": log["subsystem"],
                "action_taken": log["action_taken"],
                "resolution_time": log["resolution_time"],
            }
            for log in logs
        ],
    )

    print(f"\nDone. {collection.count()} logs embedded and stored in '{CHROMA_PERSIST_DIR}'.")
    print("Collection name:", COLLECTION_NAME)


if __name__ == "__main__":
    main()
