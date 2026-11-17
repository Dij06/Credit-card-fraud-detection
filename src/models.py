"""
Machine Learning & Deep Learning Models Module.
Consolidates the four required fraud detection architectures:
1. Logistic Regression (Traditional Linear ML with Class Weighting)
2. Random Forest (Traditional Ensemble ML with Feature Importance)
3. Artificial Neural Network / ANN (Supervised Deep Learning Classifier)
4. Autoencoder (Unsupervised Deep Learning Anomaly Detector)
"""

from pathlib import Path
from typing import Any, Tuple, Optional
import sys
import logging
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# THRESHOLD CALIBRATION UTILITY
# =====================================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_metric: str = "f1"
) -> Tuple[float, float]:
    """
    Finds optimal classification threshold on validation data to maximize F1-score.
    Overcomes default 0.50 threshold limitation under extreme class imbalance.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]

    if best_idx < len(thresholds):
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = 0.50

    best_threshold = float(np.clip(best_threshold, 0.10, 0.90))
    logger.info("Optimal threshold calibration: %.4f (Val F1: %.4f)", best_threshold, best_f1)
    return best_threshold, float(best_f1)


# =====================================================================
# 1. LOGISTIC REGRESSION (SUPERVISED ML)
# =====================================================================

class FraudLogisticRegression:
    """Logistic Regression model wrapper with threshold tuning for fraud classification."""

    def __init__(self, **kwargs):
        params = config.LOGREG_PARAMS.copy()
        params.update(kwargs)
        self.model = LogisticRegression(**params)
        self.optimal_threshold: float = 0.50
        self.val_f1: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.info("Training Logistic Regression on %d samples...", len(X_train))
        self.model.fit(X_train, y_train)
        return self

    def tune_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        val_proba = self.predict_proba(X_val)[:, 1]
        self.optimal_threshold, self.val_f1 = find_optimal_threshold(y_val.to_numpy(), val_proba)
        return self.optimal_threshold

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        t = threshold if threshold is not None else self.optimal_threshold
        proba = self.predict_proba(X)[:, 1]
        return (proba >= t).astype(int)

    def save(self, filepath: Optional[Path] = config.LOGREG_MODEL_PATH):
        target_path = Path(filepath) if filepath else config.LOGREG_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "optimal_threshold": self.optimal_threshold, "val_f1": self.val_f1}, target_path)
        logger.info("Saved Logistic Regression to: %s", target_path)

    @classmethod
    def load(cls, filepath: Optional[Path] = config.LOGREG_MODEL_PATH):
        target_path = Path(filepath) if filepath else config.LOGREG_MODEL_PATH
        data = joblib.load(target_path)
        instance = cls()
        instance.model = data["model"]
        if not hasattr(instance.model, "multi_class"):
            instance.model.multi_class = "ovr"
        instance.optimal_threshold = data.get("optimal_threshold", 0.50)
        instance.val_f1 = data.get("val_f1", 0.0)
        return instance


# =====================================================================
# 2. RANDOM FOREST (SUPERVISED ML ENSEMBLE)
# =====================================================================

class FraudRandomForest:
    """Random Forest Classifier wrapper with threshold calibration and feature importance."""

    def __init__(self, **kwargs):
        params = config.RF_PARAMS.copy()
        params.update(kwargs)
        self.model = RandomForestClassifier(**params)
        self.optimal_threshold: float = 0.50
        self.val_f1: float = 0.0
        self.feature_names: list = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        logger.info("Training Random Forest on %d samples...", len(X_train))
        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else []
        self.model.fit(X_train, y_train)
        return self

    def tune_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        val_proba = self.predict_proba(X_val)[:, 1]
        self.optimal_threshold, self.val_f1 = find_optimal_threshold(y_val.to_numpy(), val_proba)
        return self.optimal_threshold

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        t = threshold if threshold is not None else self.optimal_threshold
        proba = self.predict_proba(X)[:, 1]
        return (proba >= t).astype(int)

    def get_feature_importances(self) -> pd.DataFrame:
        cols = self.feature_names if self.feature_names else config.FEATURE_COLUMNS
        importances = self.model.feature_importances_
        return pd.DataFrame({"Feature": cols, "Importance": importances}).sort_values("Importance", ascending=False)

    def save(self, filepath: Optional[Path] = config.RF_MODEL_PATH):
        target_path = Path(filepath) if filepath else config.RF_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "optimal_threshold": self.optimal_threshold,
            "val_f1": self.val_f1,
            "feature_names": self.feature_names
        }, target_path, compress=3)
        logger.info("Saved Random Forest to: %s", target_path)

    @classmethod
    def load(cls, filepath: Optional[Path] = config.RF_MODEL_PATH):
        target_path = Path(filepath) if filepath else config.RF_MODEL_PATH
        data = joblib.load(target_path)
        instance = cls()
        instance.model = data["model"]
        instance.optimal_threshold = data.get("optimal_threshold", 0.50)
        instance.val_f1 = data.get("val_f1", 0.0)
        instance.feature_names = data.get("feature_names", [])
        return instance


# =====================================================================
# 3. ARTIFICIAL NEURAL NETWORK (SUPERVISED DEEP LEARNING)
# =====================================================================

def build_ann_classifier(
    input_dim: int = 30,
    dense_layers: list = config.ANN_PARAMS["dense_layers"],
    dropout_rate: float = config.ANN_PARAMS["dropout_rate"],
    learning_rate: float = config.ANN_PARAMS["learning_rate"]
):
    """
    Constructs a Feed-Forward Neural Network:
    Input (30) -> Dense(64, ReLU) -> Dropout(0.25) -> Dense(32, ReLU) -> Dropout(0.25) ->
    Dense(16, ReLU) -> Output Dense(1, Sigmoid).
    """
    from keras import layers, models, optimizers, losses, metrics

    model = models.Sequential(name="ANN_Fraud_Classifier")
    model.add(layers.Input(shape=(input_dim,)))

    for i, units in enumerate(dense_layers):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i+1}"))
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

    model.add(layers.Dense(1, activation="sigmoid", name="fraud_output"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
            metrics.AUC(name="auc", curve="PR")
        ]
    )
    return model


class FraudANNClassifier:
    """Wrapper for Keras ANN fraud detection model."""

    def __init__(self, **kwargs):
        self.input_dim = kwargs.get("input_dim", 30)
        self.dense_layers = kwargs.get("dense_layers", config.ANN_PARAMS["dense_layers"])
        self.dropout_rate = kwargs.get("dropout_rate", config.ANN_PARAMS["dropout_rate"])
        self.learning_rate = kwargs.get("learning_rate", config.ANN_PARAMS["learning_rate"])
        self.batch_size = kwargs.get("batch_size", config.ANN_PARAMS["batch_size"])
        self.epochs = kwargs.get("epochs", config.ANN_PARAMS["epochs"])
        self.patience = kwargs.get("patience", config.ANN_PARAMS["patience"])

        self.model = None
        self.history = None
        self.optimal_threshold: float = 0.50
        self.val_f1: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        from keras import callbacks

        input_dim = X_train.shape[1]
        self.model = build_ann_classifier(
            input_dim=input_dim,
            dense_layers=self.dense_layers,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate
        )

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        total = len(y_train)
        class_weights = {0: (total / (2.0 * n_neg)), 1: (total / (2.0 * n_pos))}

        early_stopping = callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )

        X_tr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
        y_tr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
        X_v = X_val.to_numpy() if hasattr(X_val, "to_numpy") else X_val
        y_v = y_val.to_numpy() if hasattr(y_val, "to_numpy") else y_val

        self.history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_v, y_v),
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weights,
            callbacks=[early_stopping],
            verbose=1
        )

        self.tune_threshold(X_val, y_val)
        return self

    def tune_threshold(self, X_val: Any, y_val: Any) -> float:
        val_proba = self.predict_proba(X_val)
        y_true = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.array(y_val)
        self.optimal_threshold, self.val_f1 = find_optimal_threshold(y_true, val_proba)
        return self.optimal_threshold

    def predict_proba(self, X: Any) -> np.ndarray:
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        preds = self.model.predict(X_arr, batch_size=512, verbose=0)
        return preds.ravel()

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        t = threshold if threshold is not None else self.optimal_threshold
        proba = self.predict_proba(X)
        return (proba >= t).astype(int)

    def save(self, filepath: Optional[Path] = config.ANN_MODEL_PATH):
        target_path = Path(filepath) if filepath else config.ANN_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(target_path))
        meta_path = target_path.with_suffix(".meta.joblib")
        joblib.dump({"optimal_threshold": self.optimal_threshold, "val_f1": self.val_f1}, meta_path)
        logger.info("Saved ANN model to: %s", target_path)

    @classmethod
    def load(cls, filepath: Optional[Path] = config.ANN_MODEL_PATH):
        try:
            from tensorflow.keras import models
        except Exception:
            from keras import models
        target_path = Path(filepath) if filepath else config.ANN_MODEL_PATH
        instance = cls()
        instance.model = models.load_model(str(target_path))
        meta_path = target_path.with_suffix(".meta.joblib")
        if meta_path.exists():
            meta = joblib.load(meta_path)
            instance.optimal_threshold = meta.get("optimal_threshold", 0.50)
            instance.val_f1 = meta.get("val_f1", 0.0)
        return instance


# =====================================================================
# 4. AUTOENCODER (UNSUPERVISED DEEP LEARNING ANOMALY DETECTION)
# =====================================================================

def build_autoencoder_architecture(
    input_dim: int = 30,
    encoder_layers: list = config.AUTOENCODER_PARAMS["encoder_layers"],
    decoder_layers: list = config.AUTOENCODER_PARAMS["decoder_layers"],
    latent_dim: int = config.AUTOENCODER_PARAMS["latent_dim"],
    learning_rate: float = config.AUTOENCODER_PARAMS["learning_rate"]
):
    """
    Constructs an asymmetric Autoencoder for anomaly detection:
    Encoder: Input(30) -> Dense(24, ReLU) -> Dense(16, ReLU) -> Latent(8, ReLU)
    Decoder: Latent(8) -> Dense(16, ReLU) -> Dense(24, ReLU) -> Output Dense(30, Linear)
    """
    from keras import layers, models, optimizers, losses

    input_tensor = layers.Input(shape=(input_dim,), name="transaction_input")
    x = input_tensor
    for i, units in enumerate(encoder_layers[:-1]):
        x = layers.Dense(units, activation="relu", name=f"enc_dense_{i+1}")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent_bottleneck")(x)

    x = latent
    for i, units in enumerate(decoder_layers):
        x = layers.Dense(units, activation="relu", name=f"dec_dense_{i+1}")(x)
    reconstructed = layers.Dense(input_dim, activation="linear", name="reconstruction_output")(x)

    autoencoder = models.Model(inputs=input_tensor, outputs=reconstructed, name="Fraud_Autoencoder")
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=losses.MeanSquaredError())
    return autoencoder


class FraudAutoencoderDetector:
    """Wrapper for Autoencoder Unsupervised Anomaly Detector."""

    def __init__(self, **kwargs):
        self.input_dim = kwargs.get("input_dim", 30)
        self.encoder_layers = kwargs.get("encoder_layers", config.AUTOENCODER_PARAMS["encoder_layers"])
        self.decoder_layers = kwargs.get("decoder_layers", config.AUTOENCODER_PARAMS["decoder_layers"])
        self.latent_dim = kwargs.get("latent_dim", config.AUTOENCODER_PARAMS["latent_dim"])
        self.learning_rate = kwargs.get("learning_rate", config.AUTOENCODER_PARAMS["learning_rate"])
        self.batch_size = kwargs.get("batch_size", config.AUTOENCODER_PARAMS["batch_size"])
        self.epochs = kwargs.get("epochs", config.AUTOENCODER_PARAMS["epochs"])
        self.patience = kwargs.get("patience", config.AUTOENCODER_PARAMS["patience"])

        self.model = None
        self.anomaly_threshold: float = 1.0
        self.val_f1: float = 0.0

    def fit(self, X_train_legit: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.Series):
        from keras import callbacks

        input_dim = X_train_legit.shape[1]
        self.model = build_autoencoder_architecture(
            input_dim=input_dim,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            latent_dim=self.latent_dim,
            learning_rate=self.learning_rate
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )

        X_tr = X_train_legit.to_numpy() if hasattr(X_train_legit, "to_numpy") else X_train_legit
        X_val_legit = X_val[y_val == 0]
        X_v = X_val_legit.to_numpy() if hasattr(X_val_legit, "to_numpy") else X_val_legit

        self.model.fit(
            X_tr, X_tr,
            validation_data=(X_v, X_v),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            shuffle=True,
            verbose=1
        )

        self.determine_threshold(X_val, y_val)
        return self

    def compute_reconstruction_error(self, X: Any) -> np.ndarray:
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        reconstructed = self.model.predict(X_arr, batch_size=512, verbose=0)
        mse = np.mean(np.square(X_arr - reconstructed), axis=1)
        return mse

    def determine_threshold(self, X_val: Any, y_val: Any) -> float:
        errors = self.compute_reconstruction_error(X_val)
        y_true = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.array(y_val)

        precisions, recalls, thresholds = precision_recall_curve(y_true, errors)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)

        if best_idx < len(thresholds):
            self.anomaly_threshold = float(thresholds[best_idx])
            self.val_f1 = float(f1_scores[best_idx])
        else:
            self.anomaly_threshold = float(np.percentile(errors[y_true == 0], 95))
            self.val_f1 = 0.50

        logger.info("Autoencoder anomaly threshold calibrated to MSE: %.4f (Val F1: %.4f)",
                    self.anomaly_threshold, self.val_f1)
        return self.anomaly_threshold

    def predict(self, X: Any, threshold: Optional[float] = None) -> np.ndarray:
        t = threshold if threshold is not None else self.anomaly_threshold
        errors = self.compute_reconstruction_error(X)
        return (errors >= t).astype(int)

    def save(self, filepath: Optional[Path] = config.AUTOENCODER_MODEL_PATH):
        target_path = Path(filepath) if filepath else config.AUTOENCODER_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(target_path))
        thresh_path = config.AE_THRESHOLD_PATH
        joblib.dump({"threshold": self.anomaly_threshold, "val_f1": self.val_f1}, thresh_path)
        logger.info("Saved Autoencoder to %s (Threshold: %s)", target_path, thresh_path)

    @classmethod
    def load(cls, filepath: Optional[Path] = config.AUTOENCODER_MODEL_PATH):
        try:
            from tensorflow.keras import models
        except Exception:
            from keras import models
        target_path = Path(filepath) if filepath else config.AUTOENCODER_MODEL_PATH
        instance = cls()
        instance.model = models.load_model(str(target_path))
        thresh_path = config.AE_THRESHOLD_PATH
        if thresh_path.exists():
            data = joblib.load(thresh_path)
            instance.anomaly_threshold = data.get("threshold", 1.0)
            instance.val_f1 = data.get("val_f1", 0.0)
        return instance
