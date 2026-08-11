"""
PlantSense — Chatbot Agent (LangGraph)

A ReAct-style agent that can call retrieve_similar_incidents to answer
questions about maintenance history, grounded in the RAG knowledge base.

This is step 1 of the agent layer: ONE tool, wired end-to-end and tested,
before adding query_sensor_data and get_current_predictions.
"""

import getpass
import os
import sys
import json
from pathlib import Path

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain.agents import create_agent as _create_agent
    _PROMPT_KWARG = "system_prompt"  # new API (langchain.agents, LangGraph v1.0+)
except ImportError:
    from langgraph.prebuilt import create_react_agent as _create_agent
    _PROMPT_KWARG = "prompt"  # old API (langgraph.prebuilt, pre-v1.0)

# This file lives in agents/, so parent.parent is the repo root. Add rag/
# to the path so `from retriever import ...` resolves regardless of the
# current working directory the app was launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "rag"))

from retriever import retrieve_similar_incidents as _retrieve_similar_incidents

with open(REPO_ROOT / "models" / "fleet_state.json") as f:
    _FLEET_STATE = json.load(f)

try:
    with open(REPO_ROOT / "models" / "sensor_summary.json") as f:
        _SENSOR_SUMMARY = json.load(f)
except FileNotFoundError:
    _SENSOR_SUMMARY = {}


# --- Tool definition ---
# The docstring below is NOT just documentation for humans — the LLM reads
# this description to decide WHEN to call this tool. Be specific.
@tool
def retrieve_similar_incidents(query: str) -> str:
    """
    Searches the maintenance log knowledge base for past incidents similar
    to the given query. Use this when the user asks about a specific issue,
    symptom, or wants to know if something similar has happened before
    (e.g. "why is this engine flagged", "have we seen this vibration pattern
    before", "what usually causes compressor efficiency loss").

    Args:
        query: A natural-language description of the issue or symptom to
               search for (e.g. "elevated bearing vibration").

    Returns:
        A formatted string containing the top 3 most similar past incidents,
        including root cause and action taken.
    """
    results = _retrieve_similar_incidents(query, top_k=3)

    if not results:
        return "No similar incidents found in the maintenance log knowledge base."

    formatted = []
    for r in results:
        formatted.append(
            f"[{r['log_id']}] Unit: {r['metadata']['unit']} | "
            f"Subsystem: {r['metadata']['subsystem']}\n"
            f"{r['document']}\n"
            f"Action taken: {r['metadata']['action_taken']}\n"
            f"Resolution: {r['metadata']['resolution_time']}"
        )
    return "\n\n---\n\n".join(formatted)


@tool
def get_current_predictions(engine_id: str = "") -> str:
    """
    Returns the current model predictions (remaining useful life and anomaly
    status) for a specific engine, or a fleet-wide summary if no engine_id
    is given. Use this when the user asks about an engine's current health,
    status, or wants to know which engines need attention right now
    (e.g. "what's the status of engine 12", "which engines are critical",
    "how much life does unit 47 have left").

    Args:
        engine_id: The engine's unit number as a string (e.g. "12"). Leave
                   empty to get a fleet-wide summary instead.

    Returns:
        A formatted string with RUL, anomaly score, and severity status.
    """
    if engine_id and engine_id in _FLEET_STATE:
        s = _FLEET_STATE[engine_id]
        return (
            f"Engine {s['engine_id']}: severity={s['severity']}, "
            f"predicted RUL={s['predicted_rul']} cycles, "
            f"anomaly_score={s['anomaly_score']} (threshold={s['anomaly_threshold']}, "
            f"flagged={'yes' if s['is_anomalous'] else 'no'})"
        )

    if engine_id:
        return f"No data found for engine {engine_id}. Valid engine IDs range from 1 to 100."

    # Fleet-wide summary
    critical = [s for s in _FLEET_STATE.values() if s["severity"] == "critical"]
    watch = [s for s in _FLEET_STATE.values() if s["severity"] == "watch"]
    healthy = [s for s in _FLEET_STATE.values() if s["severity"] == "healthy"]

    critical_ids = ", ".join(str(s["engine_id"]) for s in critical[:10])
    return (
        f"Fleet summary ({len(_FLEET_STATE)} engines): "
        f"{len(critical)} critical, {len(watch)} watch, {len(healthy)} healthy.\n"
        f"Critical engine IDs: {critical_ids}"
        f"{' (and more)' if len(critical) > 10 else ''}"
    )


@tool
def query_sensor_data(engine_id: str) -> str:
    """
    Returns the most significant sensor deviations for a specific engine's
    most recent operating window. Use this when the user asks WHY an engine
    is flagged, or wants to know which specific sensors are behaving
    abnormally (e.g. "why is engine 12 flagged", "what sensors are unusual
    on unit 47", "what's driving this anomaly").

    Args:
        engine_id: The engine's unit number as a string (e.g. "12").

    Returns:
        A formatted string listing the top sensor deviations, sorted by
        magnitude (values are standardized — 0 is the training-set average,
        positive/negative indicates direction and size of deviation).
    """
    if not _SENSOR_SUMMARY:
        return "Sensor summary data not available. Run compute_sensor_summary first."

    if engine_id not in _SENSOR_SUMMARY:
        return f"No sensor data found for engine {engine_id}."

    readings = _SENSOR_SUMMARY[engine_id][:5]  # top 5 most deviating sensors
    lines = [
        f"{r['sensor']}: {r['avg_scaled_value']:+.3f} std devs from training average"
        for r in readings
    ]
    return f"Top sensor deviations for engine {engine_id}:\n" + "\n".join(lines)


def build_agent(api_key: str = None):
    """Builds and returns a ready-to-use LangGraph ReAct agent.

    Pass api_key explicitly (e.g. from a Streamlit input) when running in an
    app context. Falls back to an interactive prompt only when run as a
    standalone script.
    """
    if api_key is None:
        api_key = getpass.getpass("Enter your Gemini API key: ")
    os.environ["GOOGLE_API_KEY"] = api_key

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",  # same reliable free-tier model as generation
        temperature=0.3,  # lower than generation — we want grounded, consistent
                           # answers here, not creative variety
    )

    tools = [retrieve_similar_incidents, get_current_predictions, query_sensor_data]

    system_prompt = """You are PlantSense, an AI assistant for turbofan engine
predictive maintenance. You help engineers understand equipment health by
answering questions about sensor predictions and past maintenance history.

You have three tools:
- get_current_predictions: current RUL and anomaly status for an engine or the whole fleet
- query_sensor_data: which specific sensors are driving an engine's anomaly status
- retrieve_similar_incidents: past maintenance logs similar to a described issue

For questions about current engine status, use get_current_predictions first.
If asked WHY an engine is flagged, follow up with query_sensor_data to identify
the driving sensors, then use retrieve_similar_incidents to find historically
similar cases and ground your explanation in real precedent. Always cite the
log_id of any incident you reference. Be concise and technical — you're
talking to an engineer, not writing a report."""

    agent = _create_agent(llm, tools, **{_PROMPT_KWARG: system_prompt})
    return agent


def extract_text(content) -> str:
    """
    Gemini responses via langchain-google-genai return content as a list of
    structured blocks (with internal metadata like a grounding 'signature'),
    not a plain string. This pulls out just the human-readable text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "\n".join(p for p in parts if p)
    return str(content)


if __name__ == "__main__":
    agent = build_agent()

    test_questions = [
        "What's the fleet status right now? Any engines I should worry about?",
        "Why is engine 12 flagged? What's driving it, and have we seen anything similar before?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}\nUser: {q}\n{'='*60}")
        response = agent.invoke({"messages": [{"role": "user", "content": q}]})
        final_message = response["messages"][-1]
        print(f"PlantSense: {extract_text(final_message.content)}")
