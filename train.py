"""
End-to-end training and evaluation script for TurboLife AI.
Loads NASA C-MAPSS dataset, trains Deep LSTM regressor, logs performance metrics,
evaluates on test fleet with official ground truth, and generates diagnostic charts.
"""

import os
import sys
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure config is loaded first for sandbox environment setup
import config
import preprocess
import model as model_module

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def plot_training_history(history, save_path=config.TRAINING_HISTORY_PLOT):
    """Generates and saves professional training and validation loss/metric curves."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    # 1. Loss (MSE) Curve
    axes[0].plot(epochs_range, history.history["loss"], label="Train Loss (MSE)", color="#2563eb", lw=2)
    axes[0].plot(epochs_range, history.history["val_loss"], label="Val Loss (MSE)", color="#dc2626", lw=2, linestyle="--")
    axes[0].set_title("Training vs Validation Loss (MSE)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Mean Squared Error", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, alpha=0.3)

    # 2. MAE Curve
    axes[1].plot(epochs_range, history.history["mae"], label="Train MAE", color="#059669", lw=2)
    axes[1].plot(epochs_range, history.history["val_mae"], label="Val MAE", color="#d97706", lw=2, linestyle="--")
    axes[1].set_title("Training vs Validation Mean Absolute Error (MAE)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("MAE (Cycles)", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved training history chart to {save_path}")


def plot_actual_vs_predicted(
    engine_ids, actual_ruls, predicted_ruls, save_path=config.ACTUAL_VS_PREDICTED_PLOT
):
    """Generates and saves publication-quality evaluation comparison plots on test fleet."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Sort engines by Actual RUL for clear comparison
    sorted_indices = np.argsort(actual_ruls)
    s_actual = np.array(actual_ruls)[sorted_indices]
    s_pred = np.array(predicted_ruls)[sorted_indices]
    s_engines = np.array(engine_ids)[sorted_indices]
    x_axis = np.arange(len(s_actual))

    # 1. Line Comparison
    ax1.plot(x_axis, s_actual, label="Actual Ground Truth RUL", color="#10b981", lw=2.5)
    ax1.plot(x_axis, s_pred, label="LSTM Predicted RUL", color="#f59e0b", lw=2, linestyle="--", marker="o", markersize=3)
    ax1.set_title("Test Fleet: Actual vs Predicted Remaining Useful Life", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Engine Trajectory (Ranked by Life Remaining)", fontsize=11)
    ax1.set_ylabel("RUL (Flight Cycles)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. Scatter / Regression Diagnostic
    ax2.scatter(actual_ruls, predicted_ruls, color="#2563eb", alpha=0.7, edgecolors="k", s=40, label="Engines")
    max_val = max(max(actual_ruls), max(predicted_ruls)) + 10
    ax2.plot([0, max_val], [0, max_val], color="#ef4444", lw=2, linestyle="--", label="Ideal Perfect Prediction (y=x)")
    ax2.set_title("Actual vs Predicted RUL Correlation", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Actual Ground Truth RUL (Cycles)", fontsize=11)
    ax2.set_ylabel("Predicted RUL (Cycles)", fontsize=11)
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved evaluation comparison plot to {save_path}")


def main():
    print("=" * 70)
    print(" TurboLife AI – Turbofan Engine RUL Prediction Training Pipeline ")
    print("=" * 70)

    # Step 1: Ensure Dataset is ready
    preprocess.ensure_dataset_available()

    # Step 2: Load raw training data
    print("\n[1/6] Loading CMAPSS Training Dataset...")
    train_raw = preprocess.load_cmapss_file(config.TRAIN_DATA_FILE)

    # Step 3: Compute Piecewise Capped RUL Target
    print(f"[2/6] Computing Target RUL with piecewise linear upper bound ({config.RUL_CAP} cycles)...")
    train_with_rul = preprocess.add_rul_target(train_raw, rul_cap=config.RUL_CAP)

    # Step 4: Identify Informative Sensors & Filter Static Channels
    print("[3/6] Analyzing sensor variance and filtering constant channels...")
    informative_features = preprocess.identify_informative_features(
        train_with_rul, variance_threshold=config.VARIANCE_THRESHOLD
    )
    print(f"      Selected {len(informative_features)} active dynamic features: {informative_features}")

    # Step 5: Split Engines and Scale Features
    print(f"[4/6] Splitting engines into Train / Validation sets ({int((1-config.VAL_SPLIT_RATIO)*100)}% / {int(config.VAL_SPLIT_RATIO*100)}%)...")
    train_df, val_df = preprocess.train_val_split_by_engine(
        train_with_rul, val_ratio=config.VAL_SPLIT_RATIO, random_state=config.RANDOM_STATE
    )

    print("      Fitting MinMaxScaler on training engines...")
    scaler = preprocess.fit_scaler(train_df, informative_features)
    preprocess.save_scaler(scaler, config.SCALER_FILE)

    train_scaled = preprocess.scale_features(train_df, scaler, informative_features)
    val_scaled = preprocess.scale_features(val_df, scaler, informative_features)

    # Construct 3D Sliding Window Sequences
    print(f"      Generating sliding temporal sequences of {config.SEQUENCE_LENGTH} cycles...")
    X_train, y_train = preprocess.generate_sequences(
        train_scaled, sequence_length=config.SEQUENCE_LENGTH, feature_cols=informative_features
    )
    X_val, y_val = preprocess.generate_sequences(
        val_scaled, sequence_length=config.SEQUENCE_LENGTH, feature_cols=informative_features
    )
    print(f"      Training Sequences: {X_train.shape}, Labels: {y_train.shape}")
    print(f"      Validation Sequences: {X_val.shape}, Labels: {y_val.shape}")

    # Step 6: Build & Train LSTM Network
    print("\n[5/6] Initializing Deep LSTM Neural Network...")
    input_shape = (config.SEQUENCE_LENGTH, len(informative_features))
    model = model_module.build_lstm_model(
        input_shape=input_shape,
        lstm_units_1=config.LSTM_UNITS_1,
        lstm_units_2=config.LSTM_UNITS_2,
        dropout_rate=config.DROPOUT_RATE,
        dense_units=config.DENSE_UNITS,
        learning_rate=config.LEARNING_RATE,
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(config.MODEL_FILE), monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    ]

    print(f"\nStarting model training for up to {config.EPOCHS} epochs (Batch Size: {config.BATCH_SIZE})...\n")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Plot training loss & metrics
    plot_training_history(history)

    # Step 7: Evaluate on Test Fleet with Official Ground Truth
    print("\n[6/6] Evaluating on Test Fleet (FD001)...")
    test_raw = preprocess.load_cmapss_file(config.TEST_DATA_FILE)
    test_scaled = preprocess.scale_features(test_raw, scaler, informative_features)
    
    X_test, test_engine_ids, test_cycles = preprocess.extract_test_sequences(
        test_scaled, sequence_length=config.SEQUENCE_LENGTH, feature_cols=informative_features
    )
    
    # Predict RUL
    predicted_ruls = model.predict(X_test).flatten()
    predicted_ruls = np.clip(predicted_ruls, a_min=0, a_max=None)  # RUL cannot be negative
    
    test_mae, test_rmse = None, None
    actual_ruls = None
    if os.path.exists(config.RUL_DATA_FILE):
        actual_ruls = preprocess.load_rul_ground_truth(config.RUL_DATA_FILE).values[: len(test_engine_ids)]
        
        # In standard CMAPSS evaluation, actual test RUL is clipped at RUL_CAP as well
        actual_ruls_capped = np.clip(actual_ruls, a_min=0, a_max=config.RUL_CAP)
        
        errors = predicted_ruls - actual_ruls_capped
        test_mae = float(np.mean(np.abs(errors)))
        test_rmse = float(np.sqrt(np.mean(np.square(errors))))
        
        plot_actual_vs_predicted(test_engine_ids, actual_ruls_capped, predicted_ruls)

    # Calculate final train and validation metrics
    val_loss, val_mae, val_rmse = model.evaluate(X_val, y_val, verbose=0)
    train_loss, train_mae, train_rmse = model.evaluate(X_train, y_train, verbose=0)

    # Save Metadata
    metadata = {
        "project_name": "TurboLife AI",
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "2-Layer Deep LSTM Regressor",
        "sequence_length": config.SEQUENCE_LENGTH,
        "rul_cap": config.RUL_CAP,
        "feature_columns": informative_features,
        "input_shape": list(input_shape),
        "hyperparameters": {
            "lstm_units_1": config.LSTM_UNITS_1,
            "lstm_units_2": config.LSTM_UNITS_2,
            "dense_units": config.DENSE_UNITS,
            "dropout_rate": config.DROPOUT_RATE,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "epochs_run": len(history.history["loss"]),
        },
        "metrics": {
            "train_mae": round(float(train_mae), 2),
            "train_rmse": round(float(train_rmse), 2),
            "val_mae": round(float(val_mae), 2),
            "val_rmse": round(float(val_rmse), 2),
            "test_mae": round(test_mae, 2) if test_mae is not None else "N/A",
            "test_rmse": round(test_rmse, 2) if test_rmse is not None else "N/A",
        },
    }
    model_module.save_model_metadata(config.METADATA_FILE, metadata)

    # Display Executive Summary Table
    print("\n" + "=" * 70)
    print("                    TRAINING & EVALUATION SUMMARY                    ")
    print("=" * 70)
    print(f" Train Split MAE:        {metadata['metrics']['train_mae']} cycles")
    print(f" Train Split RMSE:       {metadata['metrics']['train_rmse']} cycles")
    print(f" Validation Split MAE:   {metadata['metrics']['val_mae']} cycles")
    print(f" Validation Split RMSE:  {metadata['metrics']['val_rmse']} cycles")
    if test_mae is not None:
        print(f" Test Fleet MAE:         {metadata['metrics']['test_mae']} cycles")
        print(f" Test Fleet RMSE:        {metadata['metrics']['test_rmse']} cycles")
    print("=" * 70)

    # Sample Predictions Display
    print("\nSample Test Engine Predictions:")
    sample_df = pd.DataFrame({
        "Engine ID": test_engine_ids[:10],
        "Final Cycle": test_cycles[:10],
        "Predicted RUL": np.round(predicted_ruls[:10], 1),
        "Actual RUL": np.round(actual_ruls[:10], 1) if actual_ruls is not None else ["N/A"] * 10,
    })
    print(sample_df.to_string(index=False))
    print(f"\nArtifacts saved to:\n - Model: {config.MODEL_FILE}\n - Scaler: {config.SCALER_FILE}\n - Metadata: {config.METADATA_FILE}\n - Reports: {config.REPORTS_DIR}\n")


if __name__ == "__main__":
    main()
