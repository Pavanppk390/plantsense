# PlantSense

**Agentic predictive maintenance for turbofan engines** — combining deep learning (PyTorch LSTM + TensorFlow Autoencoder) with a RAG-grounded, tool-using LLM agent layer (LangGraph) for interactive diagnostics and autonomous shift reporting.

Built as an extension of classical predictive-maintenance work ([Predictive-Maintenance repo](https://github.com/Pavanppk390/Predictive-Maintenance)) into deep learning and agentic GenAI — grounded in real power-plant rotating-equipment experience (turbine/bearing/lubrication failure modes), retargeted to the NASA CMAPSS turbofan dataset for a consistent, well-documented benchmark.

---

## What it does

Two agentic surfaces, sharing one prediction pipeline:

- **💬 Ask PlantSense** — a reactive, tool-using chatbot. Ask about engine status, why something is flagged, or historical precedent. The agent decides which tools to call and in what order.
- **📋 Shift Reports** — an autonomous agent. No question needed — it pulls current fleet state, identifies what's urgent, grounds its explanation in historical maintenance logs, and drafts a shift handover report, the way an experienced engineer would.

## Architecture

```
Sensor Data (NASA CMAPSS FD001)
        │
        ▼
┌───────────────────────────────────┐
│  Prediction Layer                 │
│  • PyTorch LSTM  → RUL prediction │
│  • TF Autoencoder → anomaly score │
└───────────────────────────────────┘
        │
        ▼
   Fleet State (per-engine RUL, anomaly score, severity)
        │
   ┌────┴──────────────────────┐
   ▼                           ▼
Chatbot Agent              Shift Report Agent
(LangGraph ReAct,          (deterministic data
 3 tools, ~free-form Q&A)   gathering + single
                             LLM synthesis call)
        │                           │
        └───────────┬───────────────┘
                     ▼
          RAG layer (100 maintenance logs,
          Chroma + sentence-transformers)
                     │
                     ▼
              Streamlit App
     (Ask PlantSense tab + Shift Reports tab)
```

## Key results

| Component | Result |
|---|---|
| LSTM (RUL prediction) | Test RMSE **24.98** (FD001), down from 30.66 after fixing a discovered overestimation bias |
| Autoencoder (anomaly detection) | ~90 cycles of advance warning on validated test case, calibrated via 95th-percentile threshold on held-out healthy data |
| RAG knowledge base | 100 maintenance logs — 14 hand-written (grounded in real power-plant failure-mode experience), 86 LLM-generated from those seeds, balanced across 6 subsystems |
| Chatbot agent | Correctly self-corrects false premises in questions (verified against live model output before answering) rather than hallucinating a justification |

### A real bug found and fixed: RUL overestimation bias
Initial LSTM training (plain MSE loss) produced a systematic bias: 69/100 test engines had their remaining life **overestimated** — the dangerous direction for a maintenance system, since it risks missing a real failure window. Diagnosed via a real/predicted error-direction breakdown, then fixed with a custom asymmetric loss (2× penalty on overestimation), which improved **both** the safety-relevant bias (mean error +12.69 → +6.38) and overall RMSE (30.66 → 24.98).

## Tech stack

- **Deep learning:** PyTorch (LSTM), TensorFlow/Keras (Autoencoder)
- **GenAI/Agents:** LangGraph, LangChain, Google Gemini (`gemini-3.1-flash-lite`)
- **RAG:** sentence-transformers, ChromaDB
- **App:** Streamlit
- **Testing:** pytest (15 tests covering pipeline logic, model shapes, RAG formatting)
- **Data:** NASA CMAPSS Turbofan Degradation dataset (FD001)

## Repo structure

```
plantsense/
├── data/              # CMAPSS raw data, seed + generated maintenance logs
├── pipeline/          # data rebuild/generation scripts
├── models/            # LSTM, Autoencoder, fleet state computation
├── rag/               # embedding + Chroma vector store + retriever
├── agents/            # chatbot agent (LangGraph) + shift report agent
├── app/                # Streamlit app
├── tests/             # pytest unit tests
├── notebooks/         # exploratory / development notebooks
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

1. Download CMAPSS FD001 data (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) into `data/raw/`
2. Run `pipeline/rebuild_state.py` to rebuild the feature pipeline (RUL capping, constant-sensor removal, scaling)
3. Run `rag/embed_store.py` to build the Chroma vector store from `data/maintenance_logs_full.json` (not committed — regenerated on demand, ~15 seconds)
4. Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
5. Run the app: `streamlit run app/streamlit_app.py`

## Known limitations / honest notes

- **Synthetic maintenance logs**: 86 of the 100 knowledge-base logs are LLM-generated (Gemini), seeded from 14 hand-written examples grounded in real rotating-equipment failure modes. Clearly not real incident data — used to demonstrate the RAG pipeline, not as a claim of authentic maintenance history.
- **Simulated "live" data**: no real streaming sensor feed — the CMAPSS test set stands in for "engines currently being monitored."
- **RUL predictions can exceed the training cap** (125 cycles) since the model output is unconstrained regression — acceptable since the actionable signal ("healthy, not urgent") doesn't change at that range, but worth knowing if reproducing this work.
- **Anomaly flagging isn't required to be persistent** before surfacing in the current implementation — a production system would want consecutive-flag confirmation to avoid false-alarm fatigue.
- **Some architectural choices trade testability for simplicity** (e.g. modules loading data at import time) — noted directly in `tests/` rather than hidden.

## Author

Pavan Kumar Punna — Electronics and communication Engineer (IIT-ISM Dhanbad) transitioning into Data Science, with prior operational experience in thermal power plant equipment monitoring and predictive maintenance ([Jindal Power Limited](https://www.linkedin.com/in/pavan-kumarpunna-50535319b)).

