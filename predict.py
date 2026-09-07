"""
Inference and batch prediction module for TurboLife AI.
Provides functions to evaluate engine degradation, predict Remaining Useful Life (RUL),
categorize operational risk tiers, and generate automated maintenance recommendations.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

import config
import preprocess
import model as model_module

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def classify_risk(rul: float) -> Tuple[str, str]:
    """
    Classifies Remaining Useful Life into standard aviation maintenance risk tiers
    and returns actionable engineering recommendations.
    
    Tiers:
      - Healthy (Green): > 80 cycles
      - Monitor (Yellow): 31 to 80 cycles
      - High Risk (Red): <= 30 cycles
    """
    if rul > config.HEALTHY_THRESHOLD:
        return config.RISK_HEALTHY, config.RECOMMENDATIONS[config.RISK_HEALTHY]
    elif rul > config.MONITOR_THRESHOLD:
        return config.RISK_MONITOR, config.RECOMMENDATIONS[config.RISK_MONITOR]
    else:
        return config.RISK_HIGH, config.RECOMMENDATIONS[config.RISK_HIGH]


def predict_fleet_rul(
    test_df: Optional[pd.DataFrame] = None,
    test_filepath: os.PathLike = config.TEST_DATA_FILE,
    rul_filepath: Optional[os.PathLike] = config.RUL_DATA_FILE,
    model: Optional[Any] = None,
    scaler: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Executes batch inference over turbofan engine sensor telemetry.
    
    Args:
        test_df: Optional DataFrame with test telemetry. If None, loads from test_filepath.
        test_filepath: File path to NASA CMAPSS test text file.
        rul_filepath: Optional ground-truth RUL file for validation comparison.
        model: Optional pre-loaded Keras model.
        scaler: Optional pre-loaded MinMaxScaler.
        metadata: Optional model metadata dictionary.
        
    Returns:
        pd.DataFrame containing:
          - engine_id
          - current_cycle
          - predicted_rul
          - actual_rul (if ground truth available)
          - error (if ground truth available)
          - risk_level (Healthy / Monitor / High Risk)
          - recommendation
    """
    # 1. Load artifacts if not passed
    if model is None:
        if not os.path.exists(config.MODEL_FILE):
            raise FileNotFoundError(
                f"Trained model not found at {config.MODEL_FILE}. Please train the model first using 'python train.py'."
            )
        model = model_module.load_trained_model(config.MODEL_FILE)

    if scaler is None:
        if not os.path.exists(config.SCALER_FILE):
            raise FileNotFoundError(f"Scaler not found at {config.SCALER_FILE}.")
        scaler = preprocess.load_scaler(config.SCALER_FILE)

    if metadata is None:
        if not os.path.exists(config.METADATA_FILE):
            raise FileNotFoundError(f"Model metadata not found at {config.METADATA_FILE}.")
        metadata = model_module.load_model_metadata(config.METADATA_FILE)

    feature_cols = metadata["feature_columns"]
    sequence_length = metadata.get("sequence_length", config.SEQUENCE_LENGTH)

    # 2. Load test telemetry
    if test_df is None:
        if not os.path.exists(test_filepath):
            raise FileNotFoundError(f"Test data file not found at {test_filepath}.")
        test_df = preprocess.load_cmapss_file(test_filepath)

    # 3. Preprocess and extract sliding window sequences
    test_scaled = preprocess.scale_features(test_df, scaler, feature_cols)
    X_test, engine_ids, current_cycles = preprocess.extract_test_sequences(
        test_scaled, sequence_length=sequence_length, feature_cols=feature_cols
    )

    # 4. Predict RUL
    raw_predictions = model.predict(X_test, verbose=0).flatten()
    predicted_ruls = np.clip(raw_predictions, a_min=0, a_max=None)

    # 5. Check Ground Truth availability
    actual_ruls = None
    if rul_filepath and os.path.exists(rul_filepath):
        try:
            truth_series = preprocess.load_rul_ground_truth(rul_filepath)
            actual_ruls = truth_series.values[: len(engine_ids)]
        except Exception as e:
            logging.warning(f"Could not load RUL ground truth: {e}")

    # 6. Build Results DataFrame
    results = []
    for idx, eng_id in enumerate(engine_ids):
        pred_rul = float(np.round(predicted_ruls[idx], 1))
        risk, rec = classify_risk(pred_rul)
        
        row_dict = {
            "engine_id": int(eng_id),
            "current_cycle": int(current_cycles[idx]),
            "predicted_rul": pred_rul,
            "risk_level": risk,
            "recommendation": rec,
        }
        
        if actual_ruls is not None:
            act_rul = float(actual_ruls[idx])
            row_dict["actual_rul"] = act_rul
            row_dict["error"] = round(pred_rul - act_rul, 1)
            
        results.append(row_dict)

    df_results = pd.DataFrame(results)
    return df_results


def main():
    print("=" * 70)
    print(" TurboLife AI – Turbofan Engine Fleet RUL Inference ")
    print("=" * 70)

    try:
        results_df = predict_fleet_rul()
        
        print(f"\nSuccessfully evaluated {len(results_df)} engines in test fleet.")
        
        # Risk distribution breakdown
        risk_counts = results_df["risk_level"].value_counts()
        print("\nFleet Risk Summary:")
        for r_name in [config.RISK_HEALTHY, config.RISK_MONITOR, config.RISK_HIGH]:
            count = risk_counts.get(r_name, 0)
            pct = (count / len(results_df)) * 100
            print(f" • {r_name.ljust(10)}: {count:3d} engines ({pct:5.1f}%)")
            
        print("\nTop 10 Most Critical Engines (Lowest RUL):")
        critical_df = results_df.sort_values(by="predicted_rul").head(10)
        display_cols = ["engine_id", "current_cycle", "predicted_rul", "risk_level"]
        if "actual_rul" in critical_df.columns:
            display_cols.extend(["actual_rul", "error"])
        print(critical_df[display_cols].to_string(index=False))

    except Exception as e:
        print(f"\nError running inference: {e}")
        print("Tip: If you have not trained the model yet, run 'python train.py' first.")


if __name__ == "__main__":
    main()
