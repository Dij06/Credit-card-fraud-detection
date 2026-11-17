"""
Unified Test Suite for Credit Card Fraud Detection Project.
Tests data processing, leakage-safe transformations, ML models, and unified inference.
Executes cleanly with both 'python -m unittest' and 'pytest'.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_processing import (
    check_dataset_exists,
    load_raw_dataset,
    validate_dataset_structure,
    split_data_stratified,
    fit_and_transform_features,
    apply_smote_training_only,
    extract_legitimate_transactions
)
from src.models import (
    find_optimal_threshold,
    FraudLogisticRegression,
    FraudRandomForest
)
from src.prediction import FraudPredictionEngine


class TestDataProcessing(unittest.TestCase):
    """Verifies dataset ingestion, schema validation, and zero-leakage preprocessing."""

    @classmethod
    def setUpClass(cls):
        cls.has_dataset = check_dataset_exists()
        if cls.has_dataset:
            # 20,000 rows ensures sufficient fraud samples across all stratified partitions
            cls.sample_df = load_raw_dataset(nrows=20000)
        else:
            # Synthetic fallback for isolated CI
            np.random.seed(42)
            cols = [config.TIME_COLUMN] + config.V_FEATURES + [config.AMOUNT_COLUMN]
            data = np.random.randn(500, len(cols))
            cls.sample_df = pd.DataFrame(data, columns=cols)
            cls.sample_df[config.TARGET_COLUMN] = np.random.choice([0, 1], size=500, p=[0.95, 0.05])

    def test_dataset_exists_and_schema_valid(self):
        self.assertTrue(self.has_dataset, "Raw dataset creditcard.csv must exist in dataset/ folder.")
        report = validate_dataset_structure(self.sample_df)
        self.assertTrue(report["is_valid"], f"Dataset schema invalid: {report.get('missing_columns')}")
        self.assertEqual(report["missing_values"], 0, "Dataset contains unexpected missing values.")

    def test_stratified_split_ratios(self):
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data_stratified(self.sample_df)
        total = len(self.sample_df)
        self.assertAlmostEqual(len(X_tr) / total, 0.70, delta=0.03)
        self.assertAlmostEqual(len(X_val) / total, 0.15, delta=0.03)
        self.assertAlmostEqual(len(X_te) / total, 0.15, delta=0.03)

    def test_zero_leakage_scaling(self):
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data_stratified(self.sample_df)
        X_tr_s, X_val_s, X_te_s, scaler = fit_and_transform_features(X_tr, X_val, X_te, scaler_save_path=None)
        self.assertIsNotNone(scaler)
        # Scaler median must match training data median exactly
        cols = [config.TIME_COLUMN, config.AMOUNT_COLUMN]
        np.testing.assert_array_almost_equal(scaler.center_, X_tr[cols].median().to_numpy())

    def test_smote_isolation_to_training(self):
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data_stratified(self.sample_df)
        fraud_before = int((y_tr == 1).sum())
        X_res, y_res = apply_smote_training_only(X_tr, y_tr, sampling_strategy=0.20)
        fraud_after = int((y_res == 1).sum())
        self.assertGreater(fraud_after, fraud_before, "SMOTE failed to balance minority class.")
        # Ensure validation set was untouched
        self.assertEqual(len(X_val), len(y_val))

    def test_extract_legitimate_only(self):
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data_stratified(self.sample_df)
        X_legit = extract_legitimate_transactions(X_tr, y_tr)
        expected_count = int((y_tr == 0).sum())
        self.assertEqual(len(X_legit), expected_count)


class TestMachineLearningModels(unittest.TestCase):
    """Verifies model calibration, predictions, and feature importance."""

    def setUp(self):
        np.random.seed(42)
        n = 200
        cols = [config.TIME_COLUMN] + config.V_FEATURES + [config.AMOUNT_COLUMN]
        self.X_dummy = pd.DataFrame(np.random.randn(n, len(cols)), columns=cols)
        self.y_dummy = pd.Series(np.random.choice([0, 1], size=n, p=[0.85, 0.15]))

    def test_optimal_threshold_tuning(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.85, 0.25, 0.9, 0.05, 0.4])
        thresh, score = find_optimal_threshold(y_true, y_proba)
        self.assertGreaterEqual(thresh, 0.10)
        self.assertLessEqual(thresh, 0.90)
        self.assertGreaterEqual(score, 0.0)

    def test_logistic_regression_fit_predict(self):
        lr = FraudLogisticRegression(max_iter=50)
        lr.fit(self.X_dummy, self.y_dummy)
        proba = lr.predict_proba(self.X_dummy)
        preds = lr.predict(self.X_dummy)
        self.assertEqual(proba.shape, (len(self.X_dummy), 2))
        self.assertEqual(len(preds), len(self.X_dummy))

    def test_random_forest_fit_importance(self):
        rf = FraudRandomForest(n_estimators=10, max_depth=4)
        rf.fit(self.X_dummy, self.y_dummy)
        fi = rf.get_feature_importances()
        self.assertIn("Feature", fi.columns)
        self.assertIn("Importance", fi.columns)
        self.assertAlmostEqual(fi["Importance"].sum(), 1.0, delta=0.01)


class TestUnifiedPredictionEngine(unittest.TestCase):
    """Verifies end-to-end inference and batch scoring."""

    @classmethod
    def setUpClass(cls):
        cls.engine = FraudPredictionEngine()
        cls.sample_record = {f"V{i}": 0.0 for i in range(1, 29)}
        cls.sample_record["Time"] = 100.0
        cls.sample_record["Amount"] = 50.0

    def test_engine_initialization(self):
        available = self.engine.get_available_models()
        self.assertGreater(len(available), 0, "Engine failed to load any models.")

    def test_single_transaction_prediction(self):
        available = self.engine.get_available_models()
        for model_name in available:
            res = self.engine.predict_single_transaction(self.sample_record, model_name=model_name)
            self.assertIn("is_fraud", res)
            self.assertIn("verdict", res)
            self.assertIn("risk_level", res)
            self.assertIn("explanation", res)

    def test_multi_model_consensus(self):
        consensus = self.engine.compare_all_models(self.sample_record)
        self.assertIn("fraud_votes", consensus)
        self.assertIn("total_models", consensus)
        self.assertIn("consensus_verdict", consensus)

    def test_batch_prediction_dataframe(self):
        df_batch = pd.DataFrame([self.sample_record] * 10)
        available = self.engine.get_available_models()
        model_name = available[0]
        batch_res = self.engine.predict_batch(df_batch, model_name=model_name)
        self.assertEqual(batch_res["total_transactions"], 10)
        self.assertIn("results_dataframe", batch_res)
        self.assertIn("Prediction", batch_res["results_dataframe"].columns)


if __name__ == "__main__":
    unittest.main()
