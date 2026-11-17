"""
Data Processing Module for Credit Card Fraud Detection.
Consolidates data loading, schema validation, zero-leakage preprocessing,
stratified train/val/test splitting, SMOTE balancing (train-only), and EDA utilities.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import sys
import logging
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# 1. DATA INGESTION & VALIDATION
# =====================================================================

def check_dataset_exists(filepath: Optional[Path] = None) -> bool:
    """Checks whether creditcard.csv exists at the configured path."""
    target_path = Path(filepath) if filepath else config.RAW_DATASET_PATH
    return target_path.is_file() and target_path.stat().st_size > 0


def load_raw_dataset(filepath: Optional[Path] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads raw credit card transactions dataset from disk.
    Supports nrows limit for development or fast verification.
    """
    target_path = Path(filepath) if filepath else config.RAW_DATASET_PATH

    if not target_path.exists():
        error_msg = (
            f"Dataset not found at: {target_path}\n"
            f"Please download 'creditcard.csv' from Kaggle:\n"
            f"{config.KAGGLE_DATASET_URL}\n"
            f"and place it into: {config.DATASET_DIR}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Loading dataset from: %s (nrows=%s)", target_path, nrows)
    df = pd.read_csv(target_path, nrows=nrows)
    logger.info("Successfully loaded dataset with %d rows and %d columns.", len(df), df.shape[1])
    return df


def validate_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates dataset schema against expected creditcard.csv characteristics:
    - 30 input features (Time, V1-V28, Amount) + Class target
    - Binary Class (0: Legitimate, 1: Fraudulent)
    - Zero null values
    """
    expected_cols = [config.TIME_COLUMN] + config.V_FEATURES + [config.AMOUNT_COLUMN, config.TARGET_COLUMN]
    missing_cols = [c for c in expected_cols if c not in df.columns]

    class_dist = {}
    fraud_pct = 0.0
    if config.TARGET_COLUMN in df.columns:
        counts = df[config.TARGET_COLUMN].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in counts.items()}
        fraud_pct = float((df[config.TARGET_COLUMN] == 1).mean() * 100)

    report = {
        "is_valid": len(missing_cols) == 0,
        "missing_columns": missing_cols,
        "total_rows": len(df),
        "total_columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": class_dist,
        "fraud_percentage": fraud_pct,
    }
    return report


def get_dataset_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes distribution statistics for Time, Amount, and target."""
    stats = {
        "total_transactions": len(df),
        "legitimate_count": int((df[config.TARGET_COLUMN] == 0).sum()) if config.TARGET_COLUMN in df.columns else len(df),
        "fraud_count": int((df[config.TARGET_COLUMN] == 1).sum()) if config.TARGET_COLUMN in df.columns else 0,
        "amount_stats": {
            "mean": float(df[config.AMOUNT_COLUMN].mean()),
            "std": float(df[config.AMOUNT_COLUMN].std()),
            "min": float(df[config.AMOUNT_COLUMN].min()),
            "median": float(df[config.AMOUNT_COLUMN].median()),
            "max": float(df[config.AMOUNT_COLUMN].max())
        }
    }
    return stats


# =====================================================================
# 2. ZERO-LEAKAGE PREPROCESSING & SPLITTING
# =====================================================================

def split_data_stratified(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_state: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Performs 3-way stratified partition (70% Train, 15% Validation, 15% Test)
    preserving class ratio without any data leakage.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_state
    )

    val_relative_ratio = val_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_relative_ratio),
        stratify=y_temp,
        random_state=random_state
    )

    logger.info("Stratified Split Completed: Train=%d, Val=%d, Test=%d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_and_transform_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_save_path: Optional[Path] = config.SCALER_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Scales 'Time' and 'Amount' using RobustScaler.
    CRITICAL: Fits ONLY on X_train. Transforms X_train, X_val, and X_test to guarantee zero leakage.
    """
    cols_to_scale = [config.TIME_COLUMN, config.AMOUNT_COLUMN]
    scaler = RobustScaler()
    scaler.fit(X_train[cols_to_scale])

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    if scaler_save_path:
        Path(scaler_save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        logger.info("Fitted RobustScaler saved to: %s", scaler_save_path)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_smote_training_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.10,
    random_state: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE oversampling EXCLUSIVELY to the training split.
    Validation and Test sets remain strictly un-resampled to preserve real-world prevalence.
    """
    logger.info("Applying SMOTE on Training partition (Before: %d legit, %d fraud)...",
                int((y_train == 0).sum()), int((y_train == 1).sum()))
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info("SMOTE Completed (After: %d legit, %d fraud).",
                int((y_res == 0).sum()), int((y_res == 1).sum()))
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)


def extract_legitimate_transactions(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Extracts only legitimate transactions (Class 0) from training split for Autoencoder training."""
    legit_indices = y_train[y_train == 0].index
    return X_train.loc[legit_indices].copy()


def run_full_preprocessing_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """Executes the complete leakage-safe data transformation pipeline."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(df)
    X_train_s, X_val_s, X_test_s, scaler = fit_and_transform_features(X_train, X_val, X_test)
    X_train_smote, y_train_smote = apply_smote_training_only(X_train_s, y_train)
    X_train_legit_only = extract_legitimate_transactions(X_train_s, y_train)

    meta = {
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
        "smote_train_shape": X_train_smote.shape,
        "legit_train_shape": X_train_legit_only.shape,
        "feature_names": list(X_train.columns)
    }
    joblib.dump(meta, config.METADATA_PATH)

    return {
        "X_train_scaled": X_train_s,
        "y_train": y_train,
        "X_train_smote": X_train_smote,
        "y_train_smote": y_train_smote,
        "X_train_legit": X_train_legit_only,
        "X_val": X_val_s,
        "y_val": y_val,
        "X_test": X_test_s,
        "y_test": y_test,
        "scaler": scaler,
        "metadata": meta
    }


def transform_inference_record(
    record: Union[Dict[str, float], pd.DataFrame, pd.Series],
    scaler: Optional[RobustScaler] = None
) -> np.ndarray:
    """
    Transforms a single or batch inference record to match model input shape:
    Scales Time and Amount using the fitted RobustScaler, leaves V1-V28 intact.
    """
    if scaler is None:
        if config.SCALER_PATH.exists():
            scaler = joblib.load(config.SCALER_PATH)
        else:
            raise FileNotFoundError(f"Fitted scaler not found at {config.SCALER_PATH}. Run training first.")

    cols = [config.TIME_COLUMN] + config.V_FEATURES + [config.AMOUNT_COLUMN]

    if isinstance(record, dict):
        df_row = pd.DataFrame([record])
    elif isinstance(record, pd.Series):
        df_row = pd.DataFrame([record.to_dict()])
    elif isinstance(record, pd.DataFrame):
        df_row = record.copy()
    else:
        raise TypeError(f"Unsupported record type: {type(record)}")

    for c in cols:
        if c not in df_row.columns:
            df_row[c] = 0.0

    df_row = df_row[cols].copy()
    cols_to_scale = [config.TIME_COLUMN, config.AMOUNT_COLUMN]
    df_row[cols_to_scale] = scaler.transform(df_row[cols_to_scale])
    return df_row


# =====================================================================
# 3. EDA VISUALIZATION GENERATOR
# =====================================================================

def generate_eda_visualizations(
    df: pd.DataFrame,
    output_dir: Path = config.GRAPHS_DIR,
    metrics_dir: Path = config.METRICS_DIR
):
    """Generates all clean exploratory analysis figures and statistical summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    audit = validate_dataset_structure(df)
    summary = get_dataset_summary_stats(df)

    # Save summary JSON
    summary_path = metrics_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"audit": audit, "summary": summary}, f, indent=4)

    # Styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Class Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    labels = ["Legitimate", "Fraudulent"]
    colors = ["#1E3A8A", "#DC2626"]

    bars = ax1.bar(labels, counts.values, color=colors, edgecolor="#0F172A", alpha=0.9, width=0.45)
    for b in bars:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, yval + (yval * 0.02), f"{yval:,}", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Transaction Count by Class", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count")

    ax2.bar(labels, counts.values, color=colors, edgecolor="#0F172A", alpha=0.9, width=0.45)
    ax2.set_yscale("log")
    ax2.set_title("Logarithmic Class Frequency", fontsize=12, fontweight="bold")
    for i, c in enumerate(counts.values):
        ax2.text(i, c * 1.25, f"{c:,}\n({(c/len(df))*100:.3f}%)", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_class_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Transaction Amount Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    legit_amt = df[df[config.TARGET_COLUMN] == 0][config.AMOUNT_COLUMN]
    fraud_amt = df[df[config.TARGET_COLUMN] == 1][config.AMOUNT_COLUMN]

    ax1.hist(legit_amt, bins=50, color="#1E3A8A", alpha=0.75, log=True, label=f"Legit (Mean: ${legit_amt.mean():.2f})")
    ax1.hist(fraud_amt, bins=50, color="#DC2626", alpha=0.85, log=True, label=f"Fraud (Mean: ${fraud_amt.mean():.2f})")
    ax1.set_title("Transaction Amount Distribution (Log Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Amount ($)")
    ax1.legend()

    sns.boxplot(x=config.TARGET_COLUMN, y=config.AMOUNT_COLUMN, data=df[df[config.AMOUNT_COLUMN] < 800],
                palette=["#1E3A8A", "#DC2626"], ax=ax2, fliersize=2)
    ax2.set_xticklabels(["Legitimate", "Fraudulent"])
    ax2.set_title("Amount Outlier Comparison (< $800)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_amount_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Top Correlations with Fraud
    corr = df.corr()[config.TARGET_COLUMN].drop(config.TARGET_COLUMN).sort_values()
    top_corr = pd.concat([corr.head(5), corr.tail(5)])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bar_cols = ["#DC2626" if val < 0 else "#2563EB" for val in top_corr.values]
    ax.barh(top_corr.index, top_corr.values, color=bar_cols, edgecolor="#0F172A", alpha=0.85)
    ax.axvline(0, color="#64748B", linestyle="--")
    ax.set_title("Top 10 Feature Correlations with Fraud Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_feature_correlations.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Time Distribution
    fig, ax = plt.subplots(figsize=(11, 4))
    hours = (df[config.TIME_COLUMN] / 3600) % 24
    sns.kdeplot(hours[df[config.TARGET_COLUMN] == 0], color="#1E3A8A", label="Legitimate", fill=True, alpha=0.25, ax=ax)
    sns.kdeplot(hours[df[config.TARGET_COLUMN] == 1], color="#DC2626", label="Fraudulent", fill=True, alpha=0.35, ax=ax)
    ax.set_title("Transaction Density Across 24-Hour Day Cycle", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day (0 to 24)")
    ax.set_ylabel("Transaction Density")
    ax.set_xlim(0, 24)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eda_time_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("All EDA visualizations successfully generated in %s", output_dir)
