"""
Neural network model architecture module for TurboLife AI.
Defines the Deep LSTM regression architecture for Remaining Useful Life (RUL) estimation.
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, Optional

# Ensure config is imported first to set KERAS_HOME and environment variables
import config

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units_1: int = config.LSTM_UNITS_1,
    lstm_units_2: int = config.LSTM_UNITS_2,
    dropout_rate: float = config.DROPOUT_RATE,
    dense_units: int = config.DENSE_UNITS,
    learning_rate: float = config.LEARNING_RATE,
) -> tf.keras.Model:
    """
    Builds and compiles a Deep 2-Layer LSTM regression neural network.
    
    Architecture:
      1. Input Layer: (sequence_length=30, num_features)
      2. LSTM Layer 1: 64 units, returns sequences, with Dropout
      3. LSTM Layer 2: 32 units, aggregates sequence into embedding, with Dropout
      4. Fully-Connected Dense Layer: 32 units, ReLU activation
      5. Output Layer: 1 unit, Linear activation (Predicts scalar RUL)
      
    Loss: Mean Squared Error (MSE)
    Metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
    """
    model = Sequential(
        [
            Input(shape=input_shape, name="sensor_sequence_input"),
            LSTM(
                units=lstm_units_1,
                return_sequences=True,
                dropout=dropout_rate,
                name="lstm_layer_1",
            ),
            LSTM(
                units=lstm_units_2,
                return_sequences=False,
                dropout=dropout_rate,
                name="lstm_layer_2",
            ),
            Dense(units=dense_units, activation="relu", name="dense_features"),
            Dropout(dropout_rate / 2, name="dense_dropout"),
            Dense(units=1, activation="linear", name="rul_output"),
        ],
        name="TurboLife_LSTM_Regressor",
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mae", RootMeanSquaredError(name="rmse")],
    )

    logging.info(f"Initialized LSTM model with input shape {input_shape}.")
    return model


def save_model_metadata(
    filepath: os.PathLike = config.METADATA_FILE,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Saves model training metadata, active sensor features, and metrics to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Model metadata written to {filepath}")


def load_model_metadata(
    filepath: os.PathLike = config.METADATA_FILE,
) -> Dict[str, Any]:
    """Loads model metadata from JSON."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found at: {filepath}")
    with open(filepath, "r") as f:
        metadata = json.load(f)
    return metadata


def load_trained_model(
    filepath: os.PathLike = config.MODEL_FILE,
) -> tf.keras.Model:
    """Loads saved Keras LSTM model."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trained model not found at: {filepath}")
    model = load_model(filepath)
    logging.info(f"Successfully loaded trained model from {filepath}")
    return model
