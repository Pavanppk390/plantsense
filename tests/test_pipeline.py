"""
Tests for PlantSense's data pipeline logic (RUL capping, constant sensor
detection). These test the underlying LOGIC using small synthetic data,
not the actual CMAPSS files — so they run fast and need no external data.
"""

import pandas as pd
import numpy as np


def cap_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    """Standalone version of the RUL capping logic from rebuild_state.py,
    extracted here so it's testable in isolation."""
    max_cycles = df.groupby("unit_number")["time_cycles"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = (df["max_cycle"] - df["time_cycles"]).clip(upper=rul_cap)
    return df.drop("max_cycle", axis=1)


def find_constant_sensors(df: pd.DataFrame, sensor_cols: list) -> list:
    """Standalone version of the constant-sensor detection logic."""
    return [col for col in sensor_cols if df[col].std() == 0]


def test_rul_starts_high_and_ends_at_zero():
    """RUL should be highest at cycle 1 and exactly 0 at the engine's last cycle."""
    df = pd.DataFrame({
        "unit_number": [1] * 10,
        "time_cycles": list(range(1, 11)),
    })
    result = cap_rul(df, rul_cap=125)

    assert result.iloc[0]["RUL"] == 9   # cycle 1: RUL = 10 - 1 = 9
    assert result.iloc[-1]["RUL"] == 0  # last cycle: RUL = 0


def test_rul_capping_applies():
    """RUL should never exceed the cap, even for long-lived engines."""
    df = pd.DataFrame({
        "unit_number": [1] * 200,
        "time_cycles": list(range(1, 201)),
    })
    result = cap_rul(df, rul_cap=125)

    assert result["RUL"].max() == 125
    assert result.iloc[0]["RUL"] == 125  # cycle 1 of a 200-cycle engine should be capped


def test_rul_is_per_engine_not_global():
    """Each engine's RUL should be computed relative to ITS OWN max cycle,
    not a shared/global one — this catches the classic multi-engine bug."""
    df = pd.DataFrame({
        "unit_number": [1, 1, 2, 2, 2],
        "time_cycles": [1, 2, 1, 2, 3],
    })
    result = cap_rul(df, rul_cap=125)

    engine_1_last = result[(result["unit_number"] == 1) & (result["time_cycles"] == 2)]
    engine_2_last = result[(result["unit_number"] == 2) & (result["time_cycles"] == 3)]

    assert engine_1_last["RUL"].values[0] == 0
    assert engine_2_last["RUL"].values[0] == 0  # engine 2's own last cycle, not engine 1's


def test_constant_sensor_detection():
    """Sensors with zero variance should be correctly identified for removal."""
    df = pd.DataFrame({
        "sensor_a": [1.0, 1.0, 1.0, 1.0],   # constant — should be flagged
        "sensor_b": [1.0, 2.0, 3.0, 4.0],   # varies — should NOT be flagged
        "sensor_c": [5.0, 5.0, 5.0, 5.0],   # constant — should be flagged
    })

    constant = find_constant_sensors(df, ["sensor_a", "sensor_b", "sensor_c"])

    assert set(constant) == {"sensor_a", "sensor_c"}
    assert "sensor_b" not in constant
