"""
PlantSense — Retriever
Queries the Chroma vector store for the most semantically similar
maintenance logs given a natural-language query.
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "maintenance_logs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Load once, reuse across queries — loading the model per-call would be slow
_embedder = SentenceTransformer(EMBEDDING_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_collection = _client.get_collection(COLLECTION_NAME)


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
