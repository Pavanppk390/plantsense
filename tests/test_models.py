"""
Tests for PlantSense's model components. These test ARCHITECTURE SHAPES and
LOGIC using dummy inputs/models — not the actual trained .pth/.keras files,
so they run fast, need no GPU, and don't depend on large model artifacts
being present in CI.
"""

import torch
import torch.nn as nn


class RULPredictor(nn.Module):
    """Same architecture as models/current_state.py — duplicated here so
    this test file has no import-time dependency on files that load real
    data/models on import."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.fc(last_hidden).squeeze(-1)


def test_lstm_output_shape():
    """Given a batch of sequences, the model should output exactly one
    RUL prediction per sequence — no matter the batch size."""
    model = RULPredictor(input_size=14)
    dummy_input = torch.randn(8, 30, 14)  # batch=8, seq_len=30, features=14

    output = model(dummy_input)

    assert output.shape == (8,)


def test_lstm_handles_single_example():
    """Batch size of 1 shouldn't break anything (a common edge case bug)."""
    model = RULPredictor(input_size=14)
    dummy_input = torch.randn(1, 30, 14)

    output = model(dummy_input)

    assert output.shape == (1,)


def classify_severity(rul: float, is_anomalous: bool) -> str:
    """Standalone version of the severity heuristic from
    models/current_state.py, extracted for isolated testing."""
    if rul < 30 or (is_anomalous and rul < 60):
        return "critical"
    elif rul < 60 or is_anomalous:
        return "watch"
    else:
        return "healthy"


def test_severity_critical_low_rul():
    assert classify_severity(rul=10, is_anomalous=False) == "critical"


def test_severity_critical_anomalous_and_moderate_rul():
    assert classify_severity(rul=45, is_anomalous=True) == "critical"


def test_severity_watch_moderate_rul():
    assert classify_severity(rul=45, is_anomalous=False) == "watch"


def test_severity_watch_anomalous_but_high_rul():
    assert classify_severity(rul=100, is_anomalous=True) == "watch"


def test_severity_healthy():
    assert classify_severity(rul=120, is_anomalous=False) == "healthy"
