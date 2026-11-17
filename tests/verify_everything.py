"""
Comprehensive End-to-End Test Suite for FinSecure.
Verifies syntax, dataset loading, scaling, model inference,
batch predictions, and dashboard dependencies.
"""
import sys
import os
import ast
import glob
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["STREAMLIT_LOG_LEVEL"] = "error"
import pandas as pd

# Ensure project root in pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import config
from src.data_processing import (
    load_raw_dataset,
    validate_dataset_structure,
    split_data_stratified,
    apply_smote_training_only
)
from src.prediction import FraudPredictionEngine

results = []

def record(test_name, status, message=""):
    results.append({"test": test_name, "status": status, "message": message})
    badge = "[PASS]" if status else "[FAIL]"
    print(f"{badge} {test_name}: {message}")

print("=" * 70)
print("FINSECURE: FULL SYSTEM END-TO-END VERIFICATION")
print("=" * 70)

# 1. AST SYNTAX CHECK ACROSS ALL PYTHON FILES
print("\n--- Phase 1: Python AST Syntax Validation ---")
py_files = sorted(glob.glob(str(project_root / "**" / "*.py"), recursive=True))
ast_errors = 0
for f in py_files:
    rel_path = os.path.relpath(f, project_root)
    try:
        with open(f, "r", encoding="utf-8") as fp:
            ast.parse(fp.read(), filename=f)
        record(f"Syntax: {rel_path}", True, "Valid Python AST")
    except Exception as e:
        record(f"Syntax: {rel_path}", False, str(e))
        ast_errors += 1

# 2. REQUIRED FILES & ASSETS
print("\n--- Phase 2: Static Files & Artifact Verification ---")
artifacts = [
    ("Dataset (creditcard.csv)", config.RAW_DATASET_PATH),
    ("Scaler (scaler.joblib)", config.SCALER_PATH),
    ("Logistic Regression Model", config.LOGREG_MODEL_PATH),
    ("Random Forest Model", config.RF_MODEL_PATH),
    ("ANN Model", config.ANN_MODEL_PATH),
    ("Autoencoder Model", config.AUTOENCODER_MODEL_PATH),
    ("Autoencoder Threshold", config.AE_THRESHOLD_PATH),
    ("Model Comparison Metrics JSON", config.METRICS_SUMMARY_PATH),
    ("Hourly Activity CSV", config.HOURLY_ACTIVITY_CSV),
    ("Test Sample Pool CSV", config.TEST_SAMPLE_POOL_CSV),
    ("Sample Batch Template CSV", config.SAMPLE_BATCH_TEMPLATE_CSV),
    ("EDA Amount Distribution Graph", config.GRAPHS_DIR / "eda_amount_distribution.png"),
    ("EDA Class Distribution Graph", config.GRAPHS_DIR / "eda_class_distribution.png"),
    ("EDA Feature Correlations Graph", config.GRAPHS_DIR / "eda_feature_correlations.png"),
    ("EDA Time Distribution Graph", config.GRAPHS_DIR / "eda_time_distribution.png"),
    ("Confusion Matrices Graph", config.GRAPHS_DIR / "confusion_matrices_comparison.png"),
    ("ROC-PR Curves Graph", config.GRAPHS_DIR / "roc_pr_curves_comparison.png"),
]

for name, path in artifacts:
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        record(f"Asset: {name}", True, f"Found ({p.stat().st_size:,} bytes)")
    else:
        record(f"Asset: {name}", False, "Missing or 0 bytes")

# 3. DATA PROCESSING PIPELINE
print("\n--- Phase 3: Data Processing Pipeline ---")
try:
    df_sample = load_raw_dataset(nrows=20000)
    record("Data Loading", True, f"Loaded {len(df_sample)} rows x {len(df_sample.columns)} cols")
except Exception as e:
    record("Data Loading", False, str(e))

try:
    validation_info = validate_dataset_structure(df_sample)
    record("Dataset Structure Validation", validation_info["is_valid"], f"Valid={validation_info['is_valid']}, Missing values={validation_info['missing_values']}")
except Exception as e:
    record("Dataset Structure Validation", False, str(e))

try:
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(df_sample, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    record("Stratified Split", True, f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
except Exception as e:
    record("Stratified Split", False, str(e))

try:
    X_train_res, y_train_res = apply_smote_training_only(X_train, y_train)
    record("SMOTE Oversampling", True, f"Post-SMOTE train rows={len(X_train_res)} (Legit={sum(y_train_res==0)}, Fraud={sum(y_train_res==1)})")
except Exception as e:
    record("SMOTE Oversampling", False, str(e))

# 4. PREDICTION ENGINE & INFERENCE
print("\n--- Phase 4: Prediction Engine & Single Inference ---")
try:
    engine = FraudPredictionEngine()
    available_models = engine.get_available_models()
    record("Prediction Engine Initialized", len(available_models) > 0, f"Loaded models: {available_models}")
except Exception as e:
    record("Prediction Engine Initialized", False, str(e))
    available_models = []

pool_df = pd.read_csv(config.TEST_SAMPLE_POOL_CSV)
legit_record = pool_df[pool_df["Class"] == 0].iloc[0].to_dict()
fraud_record = pool_df[pool_df["Class"] == 1].iloc[0].to_dict()

for model_name in available_models:
    try:
        res_legit = engine.predict_single_transaction(legit_record, model_name=model_name)
        res_fraud = engine.predict_single_transaction(fraud_record, model_name=model_name)
        
        score_val = res_fraud["raw_score"]
        record(
            f"Single Inference: {model_name}",
            True,
            f"Legit verdict='{res_legit['verdict']}', Fraud verdict='{res_fraud['verdict']}' (Score={score_val:.4f})"
        )
    except Exception as e:
        record(f"Single Inference: {model_name}", False, str(e))

# Test consensus comparison across all available models
try:
    consensus_res = engine.compare_all_models(fraud_record)
    record("Multi-Model Consensus", True, f"Votes: {consensus_res['fraud_votes']}/{consensus_res['total_models']}, Verdict: {consensus_res['consensus_verdict']}")
except Exception as e:
    record("Multi-Model Consensus", False, str(e))

# 5. BATCH PREDICTION PIPELINE
print("\n--- Phase 5: Batch CSV Auditing ---")
batch_df = pd.read_csv(config.SAMPLE_BATCH_TEMPLATE_CSV)
for model_name in available_models:
    try:
        batch_out = engine.predict_batch(batch_df, model_name=model_name)
        df_out = batch_out["results_dataframe"]
        has_cols = all(col in df_out.columns for col in ["Prediction", "Risk_Level", "Is_Fraud"])
        record(
            f"Batch Audit: {model_name}",
            has_cols and len(df_out) == len(batch_df),
            f"Audited {batch_out['total_transactions']} txns, Flagged Fraud={batch_out['fraud_count']} ({batch_out['fraud_rate_percentage']:.1f}%)"
        )
    except Exception as e:
        record(f"Batch Audit: {model_name}", False, str(e))

# 6. DASHBOARD APP INITIALIZATION & CONSTANTS
print("\n--- Phase 6: Dashboard Integration Sanity ---")
try:
    from app.dashboard import (
        get_cached_hourly_activity,
        get_cached_test_sample_pool,
        get_cached_batch_template
    )
    hourly_df = get_cached_hourly_activity()
    pool_data = get_cached_test_sample_pool()
    batch_tpl = get_cached_batch_template()
    
    dash_ok = (
        len(hourly_df) > 0 and
        len(pool_data) > 0 and
        len(batch_tpl) > 0
    )
    record("Dashboard Cache Functions", dash_ok, f"Hourly rows={len(hourly_df)}, Pool rows={len(pool_data)}, Batch Template rows={len(batch_tpl)}")
except Exception as e:
    record("Dashboard Cache Functions", False, str(e))

# SUMMARY
print("\n" + "=" * 70)
total_tests = len(results)
passed_tests = sum(1 for r in results if r["status"])
failed_tests = total_tests - passed_tests
print(f"VERIFICATION SUMMARY: {passed_tests}/{total_tests} PASSED ({failed_tests} failed)")
print("=" * 70)

if failed_tests > 0:
    print("\nFailed checks:")
    for r in results:
        if not r["status"]:
            print(f"  - {r['test']}: {r['message']}")
    sys.exit(1)
else:
    print("\nALL SYSTEM SUBSYSTEMS AND VERIFICATION CHECKS PASSED WITH ZERO ERRORS.")
    sys.exit(0)
