"""
PlantSense — Current State
Loads the trained LSTM (RUL) and Autoencoder (anomaly) models, runs them on
the test engines, and exposes a clean "current fleet state" data structure.

This simulates what a live monitoring pipeline would produce — since we
don't have a real streaming feed, the test set stands in for "engines
currently being monitored."
"""

import numpy as np
import torch
import torch.nn as nn
from tensorflow import keras

SEQUENCE_LENGTH = 30
RUL_CAP = 125
ANOMALY_THRESHOLD = 0.24833283701059583  # from Phase 2 — 95th percentile on healthy val data


# --- Must match the architecture used during training, or state_dict load fails ---
class RULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        out = self.fc(last_hidden)
        return out.squeeze(-1)


def load_models(lstm_path: str, autoencoder_path: str, input_size: int):
    """Loads both trained models, ready for inference."""
    lstm_model = RULPredictor(input_size=input_size)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu"))
    lstm_model.eval()

    autoencoder = keras.models.load_model(autoencoder_path)

    return lstm_model, autoencoder


def compute_fleet_state(
    X_test: np.ndarray,
    test_engine_ids: list,
    lstm_model,
    autoencoder,
) -> dict:
    """
    Runs both models on the test set and returns a dict keyed by engine ID,
    containing RUL prediction and anomaly score/flag for each engine.
    This dict is what the agent tools will query against.
    """
    # --- LSTM RUL predictions ---
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        rul_predictions = lstm_model(X_test_tensor).numpy()

    # --- Autoencoder anomaly scores ---
    n_timesteps, n_features = X_test.shape[1], X_test.shape[2]
    X_test_flat = X_test.reshape(X_test.shape[0], n_timesteps * n_features)
    reconstructions = autoencoder.predict(X_test_flat, verbose=0)
    reconstruction_error = np.mean(np.square(X_test_flat - reconstructions), axis=1)

    fleet_state = {}
    for i, engine_id in enumerate(test_engine_ids):
        rul = float(rul_predictions[i])
        anomaly_score = float(reconstruction_error[i])
        is_anomalous = bool(anomaly_score > ANOMALY_THRESHOLD)  # cast from numpy.bool_

        # Simple severity heuristic — combines both model outputs into one
        # human-readable status, consistent with the flat-then-decline RUL
        # logic established back in Phase 1.
        if rul < 30 or (is_anomalous and rul < 60):
            severity = "critical"
        elif rul < 60 or is_anomalous:
            severity = "watch"
        else:
            severity = "healthy"

        fleet_state[str(engine_id)] = {
            "engine_id": int(engine_id),  # cast from numpy.int64 for JSON serialization
            "predicted_rul": round(rul, 1),
            "anomaly_score": round(anomaly_score, 4),
            "anomaly_threshold": round(ANOMALY_THRESHOLD, 4),
            "is_anomalous": is_anomalous,
            "severity": severity,
        }

    return fleet_state


def compute_sensor_summary(
    X_test: np.ndarray,
    test_engine_ids: list,
    feature_cols: list,
) -> dict:
    """
    For each engine, summarizes its most recent sensor window as the mean
    scaled value per sensor, sorted by magnitude of deviation from 0 (the
    training-set average, since features were standardized). This lets the
    agent explain WHICH sensors are driving a given engine's status, not
    just report the final RUL/anomaly numbers.
    """
    sensor_summary = {}
    for i, engine_id in enumerate(test_engine_ids):
        window = X_test[i]  # shape: (sequence_length, n_features)
        mean_per_sensor = window.mean(axis=0)  # average over the time window

        sensor_readings = [
            {"sensor": feature_cols[j], "avg_scaled_value": round(float(mean_per_sensor[j]), 3)}
            for j in range(len(feature_cols))
        ]
        # Sort by absolute deviation from 0 — largest deviations first
        sensor_readings.sort(key=lambda s: abs(s["avg_scaled_value"]), reverse=True)

        sensor_summary[str(engine_id)] = sensor_readings

    return sensor_summary


if __name__ == "__main__":
    # Quick standalone test — assumes X_test, test_engine_ids already exist
    # as .npy files, OR you run this inline in the same notebook session
    # where those variables are already in memory (see instructions).
    print("This module is meant to be imported, not run standalone,")
    print("unless you've saved X_test/test_engine_ids to disk first.")
    print("See the accompanying instructions for how to build fleet_state.json")
