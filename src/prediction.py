"""
Unified Prediction Engine for Credit Card Fraud Detection.
Provides high-performance inference for single transactions and bulk CSV uploads.
Supports all 4 models, threshold overrides, consensus voting, and structured reports.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import logging
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_processing import transform_inference_record
from src.models import (
    FraudLogisticRegression,
    FraudRandomForest,
    FraudANNClassifier,
    FraudAutoencoderDetector
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FraudPredictionEngine:
    """
    Unified predictor supporting:
    - Logistic Regression (Linear Supervised ML)
    - Random Forest (Ensemble Supervised ML)
    - Artificial Neural Network (Supervised Deep Learning)
    - Autoencoder (Unsupervised Anomaly Detection)
    - Consensus Voting (All Models)
    """

    SUPPORTED_MODELS = [
        "Random Forest",
        "Artificial Neural Network",
        "Logistic Regression",
        "Autoencoder"
    ]

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = None
        self.metadata = None
        self._load_components()

    def _load_components(self):
        """Loads fitted scaler and all serialized model weights from disk."""
        if config.SCALER_PATH.exists():
            try:
                self.scaler = joblib.load(config.SCALER_PATH)
                logger.info("Loaded preprocessor scaler.")
            except Exception as e:
                logger.warning("Error loading scaler: %s", e)

        if config.METADATA_PATH.exists():
            try:
                self.metadata = joblib.load(config.METADATA_PATH)
            except Exception:
                pass

        # 1. Logistic Regression
        if config.LOGREG_MODEL_PATH.exists():
            try:
                self.models["Logistic Regression"] = FraudLogisticRegression.load()
            except Exception as e:
                logger.warning("Could not load Logistic Regression: %s", e)

        # 2. Random Forest
        if config.RF_MODEL_PATH.exists():
            try:
                self.models["Random Forest"] = FraudRandomForest.load()
            except Exception as e:
                logger.warning("Could not load Random Forest: %s", e)

        # 3. Artificial Neural Network
        if config.ANN_MODEL_PATH.exists():
            try:
                self.models["Artificial Neural Network"] = FraudANNClassifier.load()
            except Exception as e:
                logger.warning("Could not load ANN: %s", e)

        # 4. Autoencoder
        if config.AUTOENCODER_MODEL_PATH.exists():
            try:
                self.models["Autoencoder"] = FraudAutoencoderDetector.load()
            except Exception as e:
                logger.warning("Could not load Autoencoder: %s", e)

    def is_model_available(self, model_name: str) -> bool:
        return model_name in self.models

    def get_available_models(self) -> List[str]:
        return list(self.models.keys())

    def predict_single_transaction(
        self,
        features_dict: Dict[str, float],
        model_name: str = "Random Forest",
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes prediction on an individual financial transaction.
        Outputs clear business verdicts, confidence/anomaly scores, and thresholds.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not loaded. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        X_scaled = transform_inference_record(features_dict, self.scaler)

        is_fraud = False
        probability = None
        anomaly_score = None
        active_threshold = 0.50

        if model_name in ["Logistic Regression", "Random Forest"]:
            active_threshold = threshold if threshold is not None else model.optimal_threshold
            proba_arr = model.predict_proba(X_scaled)
            probability = float(proba_arr[0, 1])
            is_fraud = bool(probability >= active_threshold)
            score_type = "Fraud Probability"
            raw_score = probability

        elif model_name == "Artificial Neural Network":
            active_threshold = threshold if threshold is not None else model.optimal_threshold
            proba_arr = model.predict_proba(X_scaled)
            probability = float(proba_arr[0])
            is_fraud = bool(probability >= active_threshold)
            score_type = "Fraud Probability"
            raw_score = probability

        elif model_name == "Autoencoder":
            active_threshold = threshold if threshold is not None else model.anomaly_threshold
            err_arr = model.compute_reconstruction_error(X_scaled)
            anomaly_score = float(err_arr[0])
            is_fraud = bool(anomaly_score >= active_threshold)
            score_type = "Reconstruction Anomaly Score (MSE)"
            raw_score = anomaly_score

        verdict = "Potential fraudulent transaction detected" if is_fraud else "Transaction appears legitimate"
        risk_level = "HIGH RISK" if is_fraud else "LOW RISK"

        if is_fraud:
            if model_name == "Autoencoder":
                explanation = (
                    f"Transaction reconstruction error ({anomaly_score:.4f} MSE) exceeds normal threshold ({active_threshold:.4f}). "
                    f"The pattern significantly deviates from the normal baseline learned from legitimate transactions."
                )
            else:
                explanation = (
                    f"Model estimated fraud likelihood of {probability*100:.1f}%, which meets or exceeds the decision threshold of {active_threshold*100:.1f}%."
                )
        else:
            if model_name == "Autoencoder":
                explanation = (
                    f"Transaction reconstruction error ({anomaly_score:.4f} MSE) is well within normal tolerance ({active_threshold:.4f}). "
                    f"Transaction characteristics conform to learned legitimate behavior."
                )
            else:
                explanation = (
                    f"Model estimated fraud likelihood of {probability*100:.1f}%, which is below the alert threshold of {active_threshold*100:.1f}%."
                )

        return {
            "model_used": model_name,
            "is_fraud": is_fraud,
            "verdict": verdict,
            "risk_level": risk_level,
            "score_type": score_type,
            "raw_score": raw_score,
            "fraud_probability": probability,
            "anomaly_score": anomaly_score,
            "threshold_used": active_threshold,
            "explanation": explanation,
            "features_analyzed": len(features_dict)
        }

    def compare_all_models(
        self,
        features_dict: Dict[str, float],
        threshold_overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Runs single transaction through all available models and assesses consensus."""
        thresholds = threshold_overrides or {}
        individual_results = {}
        fraud_votes = 0

        for name in self.SUPPORTED_MODELS:
            if name in self.models:
                t = thresholds.get(name, None)
                res = self.predict_single_transaction(features_dict, model_name=name, threshold=t)
                individual_results[name] = res
                if res["is_fraud"]:
                    fraud_votes += 1

        total_available = len(individual_results)
        consensus_ratio = fraud_votes / total_available if total_available > 0 else 0.0

        if fraud_votes >= 3:
            consensus_verdict = "HIGH FRAUD RISK - Multi-Model Consensus"
            overall_status = "CRITICAL ALERT"
        elif fraud_votes in [1, 2]:
            consensus_verdict = "AMBIGUOUS / SUSPICIOUS - Divergent Model Predictions"
            overall_status = "MANUAL REVIEW RECOMMENDED"
        else:
            consensus_verdict = "TRANSACTION APPEARS LEGITIMATE - Unanimous Agreement"
            overall_status = "APPROVED"

        return {
            "fraud_votes": fraud_votes,
            "total_models": total_available,
            "consensus_ratio": consensus_ratio,
            "consensus_verdict": consensus_verdict,
            "overall_status": overall_status,
            "individual_results": individual_results
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        model_name: str = "Random Forest",
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes high-throughput batch prediction on a transaction DataFrame.
        Returns aggregate KPIs, flagged counts, and enriched DataFrame with audit scores.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not loaded.")

        model = self.models[model_name]
        cols_needed = [config.TIME_COLUMN] + config.V_FEATURES + [config.AMOUNT_COLUMN]
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            raise ValueError(f"Batch dataset missing required columns: {missing[:5]} (Total missing: {len(missing)})")

        df_in = df[cols_needed].copy()
        X_scaled = transform_inference_record(df_in, self.scaler)

        active_threshold = threshold
        if model_name in ["Logistic Regression", "Random Forest", "Artificial Neural Network"]:
            if active_threshold is None:
                active_threshold = getattr(model, "optimal_threshold", 0.50)
            if model_name == "Artificial Neural Network":
                probas = model.predict_proba(X_scaled)
            else:
                probas = model.predict_proba(X_scaled)[:, 1]

            preds = (probas >= active_threshold).astype(int)
            scores = probas
            score_col_name = "Fraud_Probability"

        elif model_name == "Autoencoder":
            if active_threshold is None:
                active_threshold = getattr(model, "anomaly_threshold", 1.0)
            errors = model.compute_reconstruction_error(X_scaled)
            preds = (errors >= active_threshold).astype(int)
            scores = errors
            score_col_name = "Reconstruction_MSE"

        df_out = df.copy()
        df_out["Prediction"] = np.where(preds == 1, "Potentially Fraudulent", "Legitimate")
        df_out["Is_Fraud"] = preds
        df_out["Risk_Level"] = np.where(preds == 1, "High Risk", "Low Risk")
        df_out[score_col_name] = np.asarray(scores, dtype=float)

        total = len(df_out)
        fraud_cnt = int(preds.sum())
        legit_cnt = total - fraud_cnt
        fraud_rate = float((fraud_cnt / total) * 100) if total > 0 else 0.0

        return {
            "model_used": model_name,
            "total_transactions": total,
            "legitimate_count": legit_cnt,
            "fraud_count": fraud_cnt,
            "fraud_rate_percentage": fraud_rate,
            "active_threshold": float(active_threshold),
            "results_dataframe": df_out
        }
