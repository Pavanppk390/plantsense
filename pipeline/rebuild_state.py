"""
PlantSense — Rebuild Pipeline State (fresh session)

Reconstructs feature_cols, the fitted scaler, X_test, and test_engine_ids
from scratch, so current_state.py can run without needing the original
training-session variables still in memory.

Run this BEFORE current_state.py in a fresh Colab session.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SEQUENCE_LENGTH = 30
RUL_CAP = 125

# --- 1. Load raw data ---
index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

train = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=col_names)
test = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=col_names)
rul_test = pd.read_csv('RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])

# --- 2. Compute train RUL labels (capped) ---
max_cycles = train.groupby('unit_number')['time_cycles'].max().rename('max_cycle')
train = train.merge(max_cycles, on='unit_number', how='left')
train['RUL'] = (train['max_cycle'] - train['time_cycles']).clip(upper=RUL_CAP)
train.drop('max_cycle', axis=1, inplace=True)

# --- 3. Drop known constant sensors (confirmed in Phase 1) ---
constant_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                     'sensor_16', 'sensor_18', 'sensor_19']
feature_cols = [c for c in sensor_names if c not in constant_sensors]
print("Feature count:", len(feature_cols))  # should be 14

# --- 4. Fit scaler on train, transform both train and test ---
scaler = StandardScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

# --- 5. Build test sequences (last 30 cycles per engine) ---
def create_test_sequences(df, sequence_length, feature_cols):
    sequences = []
    engine_ids = []
    for engine_id in df['unit_number'].unique():
        engine_df = df[df['unit_number'] == engine_id].reset_index(drop=True)
        data = engine_df[feature_cols].values
        if len(engine_df) < sequence_length:
            padding = np.repeat(data[0:1], sequence_length - len(data), axis=0)
            data = np.concatenate([padding, data], axis=0)
        last_seq = data[-sequence_length:]
        sequences.append(last_seq)
        engine_ids.append(engine_id)
    return np.array(sequences), engine_ids

X_test, test_engine_ids = create_test_sequences(test, SEQUENCE_LENGTH, feature_cols)

print("X_test shape:", X_test.shape)  # should be (100, 30, 14)
print("Number of test engines:", len(test_engine_ids))
print("Rebuild complete — feature_cols, scaler, X_test, test_engine_ids all ready.")
