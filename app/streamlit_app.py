"""
PlantSense — Streamlit App

Two tabs:
- Ask PlantSense: chat interface backed by the LangGraph ReAct agent
- Shift Reports: browsable log of autonomously generated shift reports,
  with a button to trigger a new one on demand (simulating a scheduled run)
"""

import json
import streamlit as st

from chatbot_agent import build_agent, extract_text
from shift_report_agent import build_llm, generate_shift_report, REPORT_LOG_PATH

st.set_page_config(page_title="PlantSense", page_icon="✈️", layout="wide")


# --- Cached resource loading ---
# @st.cache_resource ensures the agent/models load ONCE per session, not on
# every rerun (Streamlit reruns the whole script on every interaction —
# without this, you'd reload the embedding model and reconnect the LLM on
# every single chat message, which would be slow and wasteful).
@st.cache_resource
def get_chat_agent(api_key: str):
    return build_agent(api_key=api_key)


@st.cache_resource
def get_report_llm(api_key: str):
    return build_llm(api_key=api_key)


def load_reports() -> list:
    try:
        with open(REPORT_LOG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# --- Sidebar: API key input ---
st.sidebar.title("✈️ PlantSense")
st.sidebar.caption("Agentic predictive maintenance demo")

api_key = st.sidebar.text_input("Gemini API Key", type="password")

if not api_key:
    st.sidebar.info("Enter your Gemini API key to activate the agents.")
    st.title("Welcome to PlantSense")
    st.write("Enter your Gemini API key in the sidebar to get started.")
    st.stop()  # halts execution here until a key is provided — avoids
               # errors from trying to build agents with no key

# --- Tabs ---
tab_chat, tab_reports = st.tabs(["💬 Ask PlantSense", "📋 Shift Reports"])


# ============ TAB 1: Ask PlantSense (reactive chat) ============
with tab_chat:
    st.subheader("Ask PlantSense")
    st.caption(
        "Ask about engine status, sensor anomalies, or historical maintenance "
        "incidents. The agent decides which tools to use based on your question."
    )

    agent = get_chat_agent(api_key)

    # Session state holds chat history across reruns (Streamlit reruns the
    # script on every interaction, so without session_state the conversation
    # would reset on every message)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("e.g. Why is engine 34 flagged?")

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
                final_message = response["messages"][-1]
                answer = extract_text(final_message.content)
                st.markdown(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ============ TAB 2: Shift Reports (autonomous feed) ============
with tab_reports:
    st.subheader("Shift Reports")
    st.caption(
        "Autonomously generated handover reports — no question needed. "
        "In production this would run on a schedule; click below to trigger one now."
    )

    if st.button("🔄 Generate New Shift Report"):
        with st.spinner("Gathering fleet state and drafting report..."):
            report_llm = get_report_llm(api_key)
            entry = generate_shift_report(report_llm)
        st.success(f"New report generated at {entry['timestamp']}")
        st.rerun()  # refresh the page so the new report appears in the feed below

    reports = load_reports()

    if not reports:
        st.info("No shift reports yet. Click the button above to generate the first one.")
    else:
        # Newest first
        for report in reversed(reports):
            counts = report["summary_counts"]
            header = (
                f"{report['timestamp']} — "
                f"{counts['critical']} critical, {counts['watch']} watch, {counts['healthy']} healthy"
            )
            with st.expander(header, expanded=(report is reports[-1])):
                st.markdown(report["report_text"])
