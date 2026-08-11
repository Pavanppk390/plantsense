"""
PlantSense — Shift Report Agent (Autonomous)

Unlike chatbot_agent.py (reactive — waits for a question), this agent runs
on its own: it gathers current fleet state, identifies what's worth
reporting, grounds explanations in historical precedent via RAG, and
generates a natural-language shift handover report — without a user prompt.

This is the "autonomous/scheduled agent" half of the two-surface design:
initiative + a schedule, not a response to a question.
"""

import json
import getpass
import os
import sys
from pathlib import Path
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# This file lives in agents/, so parent.parent is the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "rag"))

from retriever import retrieve_similar_incidents

FLEET_STATE_PATH = REPO_ROOT / "models" / "fleet_state.json"
SENSOR_SUMMARY_PATH = REPO_ROOT / "models" / "sensor_summary.json"
REPORT_LOG_PATH = REPO_ROOT / "agents" / "shift_reports.json"

MAX_CRITICAL_DETAILED = 5  # detail this many critical engines fully, summarize the rest


def gather_report_data() -> dict:
    """
    Deterministically gathers everything needed for the report — no LLM
    involved yet. This mirrors a real monitoring pipeline: structured data
    collection happens first, language generation happens last.
    """
    with open(FLEET_STATE_PATH) as f:
        fleet_state = json.load(f)
    with open(SENSOR_SUMMARY_PATH) as f:
        sensor_summary = json.load(f)

    critical = [s for s in fleet_state.values() if s["severity"] == "critical"]
    watch = [s for s in fleet_state.values() if s["severity"] == "watch"]
    healthy = [s for s in fleet_state.values() if s["severity"] == "healthy"]

    # Sort critical engines by lowest RUL first — most urgent first
    critical.sort(key=lambda s: s["predicted_rul"])

    detailed_engines = []
    for engine in critical[:MAX_CRITICAL_DETAILED]:
        engine_id = str(engine["engine_id"])
        top_sensors = sensor_summary.get(engine_id, [])[:3]

        # Ground this engine's situation in historical precedent
        sensor_desc = ", ".join(f"{s['sensor']} ({s['avg_scaled_value']:+.2f})" for s in top_sensors)
        query = f"engine issue with deviations in {sensor_desc}"
        similar = retrieve_similar_incidents(query, top_k=2)

        detailed_engines.append({
            "engine_id": engine["engine_id"],
            "predicted_rul": engine["predicted_rul"],
            "anomaly_score": engine["anomaly_score"],
            "top_sensors": top_sensors,
            "similar_incidents": [
                {
                    "log_id": s["log_id"],
                    "subsystem": s["metadata"]["subsystem"],
                    "root_cause": s["metadata"].get("root_cause", ""),
                }
                for s in similar
            ],
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "total_engines": len(fleet_state),
        "critical_count": len(critical),
        "watch_count": len(watch),
        "healthy_count": len(healthy),
        "detailed_critical_engines": detailed_engines,
        "additional_critical_count": max(0, len(critical) - MAX_CRITICAL_DETAILED),
        "watch_engine_ids": [w["engine_id"] for w in watch],
    }


def draft_report(data: dict, llm) -> str:
    """
    The ONE point where an LLM is used — synthesizing gathered data into a
    natural-language shift handover report. Everything the model needs is
    already in `data`; this is generation/formatting, not further tool use.
    """
    system_prompt = """You are PlantSense's autonomous shift-report generator.
Given structured fleet monitoring data, write a concise, professional shift
handover report — the way an experienced engineer would write one for the
next shift. Be terse and factual, not chatty. Prioritize the most urgent
engines first. Cite log_ids when referencing historical precedent. End with
a clear action recommendation. Do not invent data not present in the input."""

    user_prompt = f"""Generate a shift handover report from this fleet data:

{json.dumps(data, indent=2)}

Format: a short overview line, then a section per detailed critical engine
(RUL, driving sensors, historical precedent, recommended action), then a
one-line note on watch-list engines, then a closing summary line."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    if isinstance(response.content, list):
        return "\n".join(b.get("text", "") for b in response.content if isinstance(b, dict))
    return response.content


def generate_shift_report(llm) -> dict:
    """Full pipeline: gather data -> draft report -> log it."""
    data = gather_report_data()
    report_text = draft_report(data, llm)

    report_entry = {
        "timestamp": data["timestamp"],
        "summary_counts": {
            "critical": data["critical_count"],
            "watch": data["watch_count"],
            "healthy": data["healthy_count"],
        },
        "report_text": report_text,
    }

    # Append to the report log (this is what the Shift Reports tab will read from)
    try:
        with open(REPORT_LOG_PATH) as f:
            log = json.load(f)
    except FileNotFoundError:
        log = []

    log.append(report_entry)
    with open(REPORT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    return report_entry


def build_llm(api_key: str = None) -> ChatGoogleGenerativeAI:
    """Builds the LLM used for report synthesis. Accepts an explicit api_key
    for app contexts (e.g. Streamlit), or prompts interactively if run as a
    standalone script."""
    if api_key is None:
        api_key = getpass.getpass("Enter your Gemini API key: ")
    os.environ["GOOGLE_API_KEY"] = api_key

    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.4,  # slightly more than the chatbot — some natural
                           # report-writing variety, still grounded
    )


if __name__ == "__main__":
    llm = build_llm()

    print("Generating shift report (no user prompt — this is autonomous)...\n")
    entry = generate_shift_report(llm)

    print("=" * 60)
    print(f"SHIFT REPORT — {entry['timestamp']}")
    print("=" * 60)
    print(entry["report_text"])
    print(f"\nSaved to {REPORT_LOG_PATH}")
