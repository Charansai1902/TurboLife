"""
Configuration module for TurboLife AI - Turbofan Engine RUL Prediction.
Defines paths, column schemas, hyperparameters, sensor metadata, and risk rules.
"""

import os
import sys
from pathlib import Path

# ==========================================
# Paths Configuration & Sandboxing Isolation
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
KERAS_DIR = BASE_DIR / ".keras"
MPL_DIR = BASE_DIR / ".matplotlib"
STREAMLIT_DIR = BASE_DIR / ".streamlit"

# Configure Keras, TensorFlow, Matplotlib, and Streamlit local directories
os.environ["KERAS_HOME"] = str(KERAS_DIR)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["STREAMLIT_CONFIG_DIR"] = str(STREAMLIT_DIR)
os.environ["STREAMLIT_CREDENTIALS_FILE"] = str(STREAMLIT_DIR / "credentials.toml")
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

# Ensure directories exist
for folder in [DATA_DIR, MODELS_DIR, REPORTS_DIR, ASSETS_DIR, KERAS_DIR, MPL_DIR, STREAMLIT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_FILE = DATA_DIR / "train_FD001.txt"
TEST_DATA_FILE = DATA_DIR / "test_FD001.txt"
RUL_DATA_FILE = DATA_DIR / "RUL_FD001.txt"

MODEL_FILE = MODELS_DIR / "rul_lstm.keras"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

TRAINING_HISTORY_PLOT = REPORTS_DIR / "training_history.png"
ACTUAL_VS_PREDICTED_PLOT = REPORTS_DIR / "actual_vs_predicted.png"

# Remote Dataset URLs for automatic downloading
CMAPSS_DOWNLOAD_URLS = [
    "https://raw.githubusercontent.com/shadgriffin/NASA_Jet_Engine/master/Train_FD001.txt",
    "https://raw.githubusercontent.com/shadgriffin/NASA_Jet_Engine/master/Test_FD001.txt",
    "https://raw.githubusercontent.com/shadgriffin/NASA_Jet_Engine/master/RUL_FD001.txt",
]

# ==========================================
# Data Schema Definition
# ==========================================
INDEX_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Engineering Sensor Descriptions & Units for CMAPSS Turbofan
SENSOR_METADATA = {
    "setting_1": {"name": "Altitude / Mach Operating Setting 1", "unit": "Mach"},
    "setting_2": {"name": "Throttle Resolver Angle Setting 2", "unit": "deg"},
    "setting_3": {"name": "Sea Level Condition Setting 3", "unit": "TRA"},
    "sensor_1": {"name": "Fan Inlet Temperature (T2)", "unit": "°R"},
    "sensor_2": {"name": "LPC Outlet Temperature (T24)", "unit": "°R"},
    "sensor_3": {"name": "HPC Outlet Temperature (T30)", "unit": "°R"},
    "sensor_4": {"name": "LPT Outlet Temperature (T50)", "unit": "°R"},
    "sensor_5": {"name": "Fan Inlet Pressure (P2)", "unit": "psia"},
    "sensor_6": {"name": "Bypass Duct Pressure (P15)", "unit": "psia"},
    "sensor_7": {"name": "HPC Outlet Pressure (P30)", "unit": "psia"},
    "sensor_8": {"name": "Physical Fan Speed (Nf)", "unit": "rpm"},
    "sensor_9": {"name": "Physical Core Speed (Nc)", "unit": "rpm"},
    "sensor_10": {"name": "Engine Pressure Ratio (epr)", "unit": "-"},
    "sensor_11": {"name": "HPC Outlet Static Pressure (Ps30)", "unit": "psia"},
    "sensor_12": {"name": "Ratio of Fuel Flow to Ps30 (phi)", "unit": "pps/psia"},
    "sensor_13": {"name": "Corrected Fan Speed (NRf)", "unit": "rpm"},
    "sensor_14": {"name": "Corrected Core Speed (NRc)", "unit": "rpm"},
    "sensor_15": {"name": "Bypass Ratio (BPR)", "unit": "-"},
    "sensor_16": {"name": "Burner Fuel-Air Ratio (farB)", "unit": "-"},
    "sensor_17": {"name": "Bleed Enthalpy (htBleed)", "unit": "-"},
    "sensor_18": {"name": "Demanded Fan Speed (Nf_dmd)", "unit": "rpm"},
    "sensor_19": {"name": "Demanded Corrected Fan Speed (PCNfR_dmd)", "unit": "rpm"},
    "sensor_20": {"name": "HPT Coolant Bleed (W31)", "unit": "lbm/s"},
    "sensor_21": {"name": "LPT Coolant Bleed (W32)", "unit": "lbm/s"},
}

# ==========================================
# Machine Learning Hyperparameters
# ==========================================
SEQUENCE_LENGTH = 30        # Number of historical cycles in each sliding window
RUL_CAP = 125               # Piecewise linear RUL upper bound (cycles)
VARIANCE_THRESHOLD = 1e-4   # Drops uninformative / constant sensor channels

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
DENSE_UNITS = 32
VAL_SPLIT_RATIO = 0.2       # 20% validation split grouped strictly by engine_id
RANDOM_STATE = 42

# ==========================================
# Operational Risk Tiers & Maintenance Guidance
# ==========================================
HEALTHY_THRESHOLD = 80       # > 80 cycles remaining
MONITOR_THRESHOLD = 30       # 31 - 80 cycles remaining, <= 30 is High Risk

RISK_HEALTHY = "Healthy"
RISK_MONITOR = "Monitor"
RISK_HIGH = "High Risk"

RISK_COLORS = {
    RISK_HEALTHY: "#10b981",  # Emerald Green
    RISK_MONITOR: "#f59e0b",  # Amber Yellow
    RISK_HIGH: "#ef4444",     # Crimson Red
}

RECOMMENDATIONS = {
    RISK_HEALTHY: "Continue normal operation. Keep monitoring sensor trends.",
    RISK_MONITOR: "Plan an inspection during an upcoming maintenance window.",
    RISK_HIGH: "Schedule maintenance urgently before further operation.",
}
