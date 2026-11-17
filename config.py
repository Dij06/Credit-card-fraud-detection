"""
Configuration module for FinSecure Credit Card Fraud Detection.
Lightweight, reproducible settings optimized for student laptops.
"""

from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
METRICS_DIR = RESULTS_DIR / "metrics"

# Ensure runtime directories exist
for directory in [DATASET_DIR, MODELS_DIR, RESULTS_DIR, GRAPHS_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
RAW_DATASET_FILENAME = "creditcard.csv"
RAW_DATASET_PATH = DATASET_DIR / RAW_DATASET_FILENAME
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

# Features and Target
TARGET_COLUMN = "Class"
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
V_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = [TIME_COLUMN] + V_FEATURES + [AMOUNT_COLUMN]

# Reproducibility
RANDOM_SEED = 42

# Data Splitting Ratios (Stratified: 70% Train, 15% Validation, 15% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Preprocessing Artifacts
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METADATA_PATH = MODELS_DIR / "preprocessing_meta.joblib"

# Model Artifact Paths
LOGREG_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
ANN_MODEL_PATH = MODELS_DIR / "ann_fraud_detector.keras"
AUTOENCODER_MODEL_PATH = MODELS_DIR / "autoencoder_anomaly.keras"
AE_THRESHOLD_PATH = MODELS_DIR / "autoencoder_threshold.joblib"

# Currency Settings
DEFAULT_CURRENCY = "EUR"
EUR_TO_INR_RATE = 90.00  # Configurable demonstration rate for India

# Results Artifacts
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
HOURLY_ACTIVITY_CSV = RESULTS_DIR / "metrics" / "hourly_activity.csv"
TEST_SAMPLE_POOL_CSV = METRICS_DIR / "test_sample_pool.csv"
SAMPLE_BATCH_TEMPLATE_CSV = METRICS_DIR / "sample_batch_template.csv"
METRICS_SUMMARY_PATH = METRICS_DIR / "model_comparison_metrics.json"

# Fast Laptop Execution Flags
DEV_SAMPLE_SIZE = 50000

# Model Hyperparameters (Compact architectures)
LOGREG_PARAMS = {
    "C": 0.01,
    "penalty": "l2",
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "max_iter": 500
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

ANN_PARAMS = {
    "dense_layers": [64, 32, 16],
    "dropout_rate": 0.25,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 15,
    "patience": 4
}

AUTOENCODER_PARAMS = {
    "encoder_layers": [24, 16, 8],
    "decoder_layers": [16, 24],
    "latent_dim": 8,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 15,
    "patience": 4
}
