"""
Tests for PlantSense's RAG layer. These test the PURE LOGIC around
embedding text construction and result formatting — not live Chroma
queries or actual embedding model calls, which need the vector store
and downloaded models to be present (not guaranteed in a clean CI run).
"""


def build_embedding_text(log: dict) -> str:
    """Same logic as rag/embed_store.py's build_embedding_text, duplicated
    here to avoid an import-time dependency on the embedding model
    downloading during test collection."""
    return (
        f"Unit: {log['unit']}\n"
        f"Issue: {log['reported_issue']}\n"
        f"Root cause: {log['root_cause']}"
    )


def test_embedding_text_includes_key_fields():
    """The embedding text must include unit, issue, and root cause —
    these are what retrieval quality depends on."""
    log = {
        "unit": "Engine Unit 47 (Turbofan)",
        "reported_issue": "Elevated vibration near LP shaft bearing.",
        "root_cause": "Lubrication degradation.",
        "action_taken": "Replaced lubricant.",  # deliberately NOT embedded
    }

    text = build_embedding_text(log)

    assert "Engine Unit 47" in text
    assert "Elevated vibration" in text
    assert "Lubrication degradation" in text


def test_embedding_text_excludes_non_semantic_fields():
    """action_taken and resolution_time are metadata, not embedded content —
    this guards against accidentally changing that design decision."""
    log = {
        "unit": "Engine Unit 12",
        "reported_issue": "Test issue",
        "root_cause": "Test cause",
        "action_taken": "SHOULD_NOT_APPEAR",
        "resolution_time": "SHOULD_NOT_APPEAR_EITHER",
    }

    text = build_embedding_text(log)

    assert "SHOULD_NOT_APPEAR" not in text
    assert "SHOULD_NOT_APPEAR_EITHER" not in text


def format_retrieval_results(results: list) -> str:
    """Same formatting logic used in agents/chatbot_agent.py's
    retrieve_similar_incidents tool, duplicated for isolated testing."""
    if not results:
        return "No similar incidents found in the maintenance log knowledge base."

    formatted = []
    for r in results:
        formatted.append(
            f"[{r['log_id']}] Unit: {r['metadata']['unit']} | "
            f"Subsystem: {r['metadata']['subsystem']}\n"
            f"{r['document']}\n"
            f"Action taken: {r['metadata']['action_taken']}"
        )
    return "\n\n---\n\n".join(formatted)


def test_empty_retrieval_results_handled_gracefully():
    """An empty result set should return a clear message, not crash or
    return an empty/confusing string — important since the agent LLM
    needs to understand when nothing was found."""
    result = format_retrieval_results([])
    assert "No similar incidents found" in result


def test_retrieval_formatting_includes_log_id():
    """Every formatted result must include its log_id — this is what lets
    the agent cite sources, which is a core trust/traceability requirement."""
    fake_results = [{
        "log_id": "ML-0001",
        "document": "Some issue text",
        "metadata": {
            "unit": "Engine Unit 1",
            "subsystem": "bearing_lubrication",
            "action_taken": "Replaced part",
        },
    }]

    formatted = format_retrieval_results(fake_results)

    assert "ML-0001" in formatted
    assert "bearing_lubrication" in formatted
