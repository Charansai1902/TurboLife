"""
Data preprocessing pipeline for TurboLife AI.
Handles loading NASA C-MAPSS dataset, calculating capped RUL targets,
filtering low-variance sensors, normalizing features, splitting engines,
and constructing 3D temporal sliding window sequences for LSTM training and inference.
"""

import os
import urllib.request
import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def load_cmapss_file(filepath: os.PathLike) -> pd.DataFrame:
    """
    Loads space-separated NASA C-MAPSS data file into a pandas DataFrame with proper column headers.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    # C-MAPSS raw files are whitespace delimited with possible trailing whitespace
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    
    # Drop any trailing NaN columns caused by trailing spaces
    df = df.dropna(axis=1, how="all")
    
    # Assign column names up to number of columns present
    cols = config.ALL_COLUMNS[: df.shape[1]]
    df.columns = cols
    
    # Ensure standard types
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    
    logging.info(f"Loaded {filepath.name if hasattr(filepath, 'name') else filepath}: {df.shape[0]} rows, {df['engine_id'].nunique()} engines.")
    return df


def load_rul_ground_truth(filepath: os.PathLike) -> pd.Series:
    """
    Loads official ground-truth Remaining Useful Life for test engines.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"RUL file not found at: {filepath}")
    
    df_rul = pd.read_csv(filepath, sep=r"\s+", header=None)
    rul_series = df_rul.iloc[:, 0].astype(float)
    logging.info(f"Loaded {len(rul_series)} ground truth RUL values from {filepath}.")
    return rul_series


def add_rul_target(df: pd.DataFrame, rul_cap: int = config.RUL_CAP) -> pd.DataFrame:
    """
    Computes Remaining Useful Life (RUL) for each engine cycle in training data.
    RUL = (maximum operating cycle for that engine) - (current cycle).
    Applies piecewise linear capping at `rul_cap` cycles.
    """
    df_with_rul = df.copy()
    
    # Find max cycle for each engine
    max_cycles = df_with_rul.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    
    df_with_rul = df_with_rul.merge(max_cycles, on="engine_id", how="left")
    df_with_rul["rul"] = df_with_rul["max_cycle"] - df_with_rul["cycle"]
    
    # Drop intermediate max_cycle
    df_with_rul = df_with_rul.drop(columns=["max_cycle"])
    
    # Apply piecewise linear upper bound (RUL clipping)
    if rul_cap is not None and rul_cap > 0:
        df_with_rul["rul"] = df_with_rul["rul"].clip(upper=rul_cap)
        
    return df_with_rul


def identify_informative_features(
    df: pd.DataFrame, variance_threshold: float = config.VARIANCE_THRESHOLD
) -> List[str]:
    """
    Identifies sensors and operational settings that exhibit dynamic variance.
    Filters out static sensor channels (near-zero variance) that carry no degradation signal.
    """
    candidate_cols = config.SETTING_COLUMNS + config.SENSOR_COLUMNS
    available_cols = [c for c in candidate_cols if c in df.columns]
    
    variances = df[available_cols].var()
    informative_cols = variances[variances > variance_threshold].index.tolist()
    
    dropped_cols = [c for c in available_cols if c not in informative_cols]
    logging.info(f"Retained {len(informative_cols)} informative features.")
    if dropped_cols:
        logging.info(f"Dropped {len(dropped_cols)} static/low-variance features: {dropped_cols}")
        
    return informative_cols


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> MinMaxScaler:
    """
    Fits a MinMaxScaler (range 0 to 1) strictly on training engine data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feature_cols])
    return scaler


def scale_features(
    df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: List[str]
) -> pd.DataFrame:
    """
    Transforms feature columns in the dataframe using the pre-fitted scaler.
    """
    scaled_df = df.copy()
    scaled_df[feature_cols] = scaler.transform(df[feature_cols])
    return scaled_df


def train_val_split_by_engine(
    df: pd.DataFrame, val_ratio: float = config.VAL_SPLIT_RATIO, random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train and validation subsets grouped strictly by engine_id.
    Prevents temporal sequence data leakage between training and validation folds.
    """
    unique_engines = df["engine_id"].unique()
    train_engines, val_engines = train_test_split(
        unique_engines, test_size=val_ratio, random_state=random_state
    )
    
    train_df = df[df["engine_id"].isin(train_engines)].copy()
    val_df = df[df["engine_id"].isin(val_engines)].copy()
    
    logging.info(f"Engine Split: {len(train_engines)} Train Engines ({len(train_df)} rows), {len(val_engines)} Val Engines ({len(val_df)} rows)")
    return train_df, val_df


def generate_sequences(
    df: pd.DataFrame,
    sequence_length: int = config.SEQUENCE_LENGTH,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D time-series engine records into 3D sliding-window tensor sequences.
    
    Returns:
        X: 3D tensor of shape (num_samples, sequence_length, num_features)
        y: 1D array of shape (num_samples,) containing RUL at the end of each window
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("setting_")]
        
    X_list, y_list = [], []
    
    for engine_id, group in df.groupby("engine_id"):
        features = group[feature_cols].values
        ruls = group["rul"].values
        n_rows = len(group)
        
        if n_rows < sequence_length:
            # Handle engines with fewer cycles by padding the earliest cycle
            pad_len = sequence_length - n_rows
            padding = np.repeat(features[:1], pad_len, axis=0)
            padded_features = np.vstack([padding, features])
            X_list.append(padded_features)
            y_list.append(ruls[-1])
        else:
            # Sliding window of sequence_length
            for i in range(sequence_length, n_rows + 1):
                X_list.append(features[i - sequence_length : i])
                y_list.append(ruls[i - 1])
                
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def extract_test_sequences(
    test_df: pd.DataFrame,
    sequence_length: int = config.SEQUENCE_LENGTH,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Extracts the latest sequence_length cycles for each engine in the test set.
    
    Returns:
        X_test: 3D tensor of shape (num_engines, sequence_length, num_features)
        engine_ids: List of engine IDs corresponding to each sequence
        current_cycles: List of the final operating cycle observed for each engine
    """
    if feature_cols is None:
        feature_cols = [c for c in test_df.columns if c.startswith("sensor_") or c.startswith("setting_")]
        
    X_list = []
    engine_ids = []
    current_cycles = []
    
    for engine_id, group in test_df.groupby("engine_id"):
        features = group[feature_cols].values
        n_rows = len(group)
        current_cycle = int(group["cycle"].iloc[-1])
        
        if n_rows < sequence_length:
            # Pad early cycles if engine was halted prior to sequence_length
            pad_len = sequence_length - n_rows
            padding = np.repeat(features[:1], pad_len, axis=0)
            seq = np.vstack([padding, features])
        else:
            # Take the latest sequence_length cycles representing current health state
            seq = features[-sequence_length:]
            
        X_list.append(seq)
        engine_ids.append(int(engine_id))
        current_cycles.append(current_cycle)
        
    X_test = np.array(X_list, dtype=np.float32)
    return X_test, engine_ids, current_cycles


def save_scaler(scaler: MinMaxScaler, filepath: os.PathLike = config.SCALER_FILE):
    """Saves fitted MinMaxScaler with joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)
    logging.info(f"Saved MinMaxScaler to {filepath}")


def load_scaler(filepath: os.PathLike = config.SCALER_FILE) -> MinMaxScaler:
    """Loads fitted MinMaxScaler with joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scaler file not found at: {filepath}")
    scaler = joblib.load(filepath)
    return scaler


def generate_synthetic_cmapss_data(data_dir: os.PathLike = config.DATA_DIR):
    """
    Generates realistic synthetic CMAPSS FD001 dataset files for offline/demo operation.
    Simulates exponential degradation across key turbofan thermodynamic parameters.
    """
    os.makedirs(data_dir, exist_ok=True)
    logging.info("Generating realistic synthetic CMAPSS FD001 dataset for standalone execution...")
    
    np.random.seed(config.RANDOM_STATE)
    
    # 1. Generate train_FD001.txt (100 run-to-failure engines)
    train_rows = []
    for engine in range(1, 101):
        max_life = np.random.randint(130, 360)
        for cycle in range(1, max_life + 1):
            deg = np.exp((cycle / max_life) * 2.5) / np.exp(2.5)  # 0 to 1 non-linear wear
            noise = np.random.normal(0, 0.02)
            
            # Operational settings (FD001 is sea-level single operating regime)
            s1 = round(float(np.random.normal(0.000, 0.002)), 4)
            s2 = round(float(np.random.normal(0.000, 0.0003)), 4)
            s3 = 100.0
            
            # Key degradation sensors
            t2 = 518.67
            t24 = round(float(642.0 + 5.0 * deg + np.random.normal(0, 0.4)), 2)
            t30 = round(float(1585.0 + 35.0 * deg + np.random.normal(0, 1.2)), 2)
            t50 = round(float(1400.0 + 28.0 * deg + np.random.normal(0, 1.0)), 2)
            p2 = 14.62
            p15 = 21.61
            p30 = round(float(554.0 - 15.0 * deg + np.random.normal(0, 0.5)), 2)
            nf = round(float(2388.0 + 0.15 * deg + np.random.normal(0, 0.05)), 2)
            nc = round(float(9050.0 + 30.0 * deg + np.random.normal(0, 1.5)), 2)
            epr = 1.30
            ps30 = round(float(47.2 + 1.2 * deg + np.random.normal(0, 0.08)), 2)
            phi = round(float(521.5 - 12.0 * deg + np.random.normal(0, 0.4)), 2)
            nrf = round(float(2388.0 + 0.12 * deg + np.random.normal(0, 0.04)), 2)
            nrc = round(float(8130.0 + 20.0 * deg + np.random.normal(0, 1.2)), 2)
            bpr = round(float(8.40 + 0.35 * deg + np.random.normal(0, 0.02)), 4)
            farb = 0.03
            htbleed = int(392 + 5 * deg + np.random.normal(0, 1))
            nf_dmd = 2388
            pcnfr_dmd = 100.0
            w31 = round(float(38.8 - 0.7 * deg + np.random.normal(0, 0.05)), 2)
            w32 = round(float(23.3 - 0.5 * deg + np.random.normal(0, 0.04)), 2)
            
            row = [engine, cycle, s1, s2, s3, t2, t24, t30, t50, p2, p15, p30, nf, nc, epr,
                   ps30, phi, nrf, nrc, bpr, farb, htbleed, nf_dmd, pcnfr_dmd, w31, w32]
            train_rows.append(row)
            
    df_train = pd.DataFrame(train_rows)
    df_train.to_csv(config.TRAIN_DATA_FILE, sep=" ", header=False, index=False)
    
    # 2. Generate test_FD001.txt & RUL_FD001.txt (100 engines stopped before failure)
    test_rows = []
    rul_values = []
    for engine in range(1, 101):
        total_life = np.random.randint(140, 360)
        remaining_rul = np.random.randint(10, 135)
        cutoff_cycle = max(31, total_life - remaining_rul)
        rul_values.append(total_life - cutoff_cycle)
        
        for cycle in range(1, cutoff_cycle + 1):
            deg = np.exp((cycle / total_life) * 2.5) / np.exp(2.5)
            s1 = round(float(np.random.normal(0.000, 0.002)), 4)
            s2 = round(float(np.random.normal(0.000, 0.0003)), 4)
            s3 = 100.0
            t2 = 518.67
            t24 = round(float(642.0 + 5.0 * deg + np.random.normal(0, 0.4)), 2)
            t30 = round(float(1585.0 + 35.0 * deg + np.random.normal(0, 1.2)), 2)
            t50 = round(float(1400.0 + 28.0 * deg + np.random.normal(0, 1.0)), 2)
            p2 = 14.62
            p15 = 21.61
            p30 = round(float(554.0 - 15.0 * deg + np.random.normal(0, 0.5)), 2)
            nf = round(float(2388.0 + 0.15 * deg + np.random.normal(0, 0.05)), 2)
            nc = round(float(9050.0 + 30.0 * deg + np.random.normal(0, 1.5)), 2)
            epr = 1.30
            ps30 = round(float(47.2 + 1.2 * deg + np.random.normal(0, 0.08)), 2)
            phi = round(float(521.5 - 12.0 * deg + np.random.normal(0, 0.4)), 2)
            nrf = round(float(2388.0 + 0.12 * deg + np.random.normal(0, 0.04)), 2)
            nrc = round(float(8130.0 + 20.0 * deg + np.random.normal(0, 1.2)), 2)
            bpr = round(float(8.40 + 0.35 * deg + np.random.normal(0, 0.02)), 4)
            farb = 0.03
            htbleed = int(392 + 5 * deg + np.random.normal(0, 1))
            nf_dmd = 2388
            pcnfr_dmd = 100.0
            w31 = round(float(38.8 - 0.7 * deg + np.random.normal(0, 0.05)), 2)
            w32 = round(float(23.3 - 0.5 * deg + np.random.normal(0, 0.04)), 2)
            
            row = [engine, cycle, s1, s2, s3, t2, t24, t30, t50, p2, p15, p30, nf, nc, epr,
                   ps30, phi, nrf, nrc, bpr, farb, htbleed, nf_dmd, pcnfr_dmd, w31, w32]
            test_rows.append(row)
            
    df_test = pd.DataFrame(test_rows)
    df_test.to_csv(config.TEST_DATA_FILE, sep=" ", header=False, index=False)
    
    df_rul = pd.DataFrame(rul_values)
    df_rul.to_csv(config.RUL_DATA_FILE, sep=" ", header=False, index=False)
    logging.info("Generated synthetic training, test, and RUL files successfully.")


def ensure_dataset_available(data_dir: os.PathLike = config.DATA_DIR):
    """
    Checks if train, test, and RUL files exist in data/.
    If missing, attempts downloading from verified repository mirrors;
    if network is unreachable, generates high-fidelity synthetic CMAPSS datasets.
    """
    os.makedirs(data_dir, exist_ok=True)
    required_files = [config.TRAIN_DATA_FILE, config.TEST_DATA_FILE, config.RUL_DATA_FILE]
    
    if all(os.path.exists(f) for f in required_files):
        logging.info("All C-MAPSS dataset files verified in data/.")
        return

    logging.info("Dataset files not fully present. Attempting download from NASA mirror...")
    urls_and_targets = [
        ("https://raw.githubusercontent.com/shadgriffin/NASA_Jet_Engine/master/Train_FD001.txt", config.TRAIN_DATA_FILE),
        ("https://raw.githubusercontent.com/shadgriffin/NASA_Jet_Engine/master/Test_FD001.txt", config.TEST_DATA_FILE),
        ("https://raw.githubusercontent.com/shadgriffin/NASA_Jet_Engine/master/RUL_FD001.txt", config.RUL_DATA_FILE),
    ]
    
    download_success = True
    for url, target_path in urls_and_targets:
        if not os.path.exists(target_path):
            try:
                urllib.request.urlretrieve(url, target_path)
                logging.info(f"Downloaded {target_path.name} from {url}")
            except Exception as e:
                logging.warning(f"Download failed for {url}: {e}")
                download_success = False
                break
                
    if not download_success or not all(os.path.exists(f) for f in required_files):
        logging.info("Network download unavailable. Falling back to synthetic CMAPSS simulation...")
        generate_synthetic_cmapss_data(data_dir)
