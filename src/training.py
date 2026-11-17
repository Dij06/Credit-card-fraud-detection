"""
Training and Evaluation Orchestrator Module.
Manages end-to-end model training, threshold calibration, and test evaluation.
Outputs publication-quality performance graphs and results/model_comparison.csv.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

import config
from src.data_processing import (
    load_raw_dataset,
    run_full_preprocessing_pipeline,
    generate_eda_visualizations
)
from src.models import (
    FraudLogisticRegression,
    FraudRandomForest,
    FraudANNClassifier,
    FraudAutoencoderDetector
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_single_model(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Computes comprehensive performance metrics for a model on test partition."""
    y_true = y_test.to_numpy()

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
    elif hasattr(model, "compute_reconstruction_error"):
        raw_err = model.compute_reconstruction_error(X_test)
        max_err = np.percentile(raw_err, 99.9) + 1e-10
        y_proba = np.clip(raw_err / max_err, 0.0, 1.0)
    else:
        raise AttributeError(f"Model {model_name} missing probability/scoring method.")

    if threshold is not None:
        y_pred = model.predict(X_test, threshold=threshold)
    elif hasattr(model, "optimal_threshold"):
        y_pred = model.predict(X_test, threshold=model.optimal_threshold)
    elif hasattr(model, "anomaly_threshold"):
        y_pred = model.predict(X_test, threshold=model.anomaly_threshold)
    else:
        y_pred = (y_proba >= 0.50).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.50

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = float(precision)

    used_thresh = threshold if threshold is not None else getattr(
        model, "optimal_threshold", getattr(model, "anomaly_threshold", 0.50)
    )

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "optimal_threshold": float(used_thresh),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba
    }
    return metrics


def evaluate_and_plot_all_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = config.GRAPHS_DIR,
    metrics_dir: Path = config.METRICS_DIR
) -> pd.DataFrame:
    """Evaluates all models, saves comparison table and publication graphs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_list = []
    eval_cache = {}

    for name, model in models_dict.items():
        logger.info("Evaluating: %s...", name)
        res = evaluate_single_model(name, model, X_test, y_test)
        eval_cache[name] = res
        results_list.append({
            "Model": name,
            "Accuracy": round(res["accuracy"], 4),
            "Recall": round(res["recall"], 4),
            "Precision": round(res["precision"], 4),
            "F1-Score": round(res["f1_score"], 4),
            "ROC-AUC": round(res["roc_auc"], 4),
            "PR-AUC": round(res["pr_auc"], 4),
            "Optimal Threshold": round(res["optimal_threshold"], 4)
        })

    df_comp = pd.DataFrame(results_list)

    # Save to both standard results/model_comparison.csv and results/metrics
    comp_csv = config.RESULTS_DIR / "model_comparison.csv"
    df_comp.to_csv(comp_csv, index=False)
    df_comp.to_csv(metrics_dir / "model_comparison_table.csv", index=False)

    serializable_cache = {}
    for k, v in eval_cache.items():
        serializable_cache[k] = {
            "model_name": v["model_name"],
            "accuracy": v["accuracy"],
            "precision": v["precision"],
            "recall": v["recall"],
            "f1_score": v["f1_score"],
            "roc_auc": v["roc_auc"],
            "pr_auc": v["pr_auc"],
            "optimal_threshold": v["optimal_threshold"],
            "confusion_matrix": v["confusion_matrix"]
        }
    with open(metrics_dir / "model_comparison_metrics.json", "w") as f:
        json.dump(serializable_cache, f, indent=4)

    # 1. Confusion Matrices Comparison
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(eval_cache.items()):
        cm = np.array([
            [res["confusion_matrix"]["tn"], res["confusion_matrix"]["fp"]],
            [res["confusion_matrix"]["fn"], res["confusion_matrix"]["tp"]]
        ])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        axes[idx].set_title(f"{name}\n(Recall: {res['recall']*100:.1f}%, Prec: {res['precision']*100:.1f}%)",
                            fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2. ROC and PR Curves Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {"Logistic Regression": "#2563EB", "Random Forest": "#059669",
              "Artificial Neural Network": "#7C3AED", "Autoencoder": "#D97706"}

    for name, res in eval_cache.items():
        col = colors.get(name, "#2563EB")
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", color=col, lw=2)

        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_proba"])
        ax2.plot(rec, prec, label=f"{name} (PR-AUC={res['pr_auc']:.3f})", color=col, lw=2)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")

    ax2.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_dir / "roc_pr_curves_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Model Performance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 4.5))
    metrics_to_plot = ["Precision", "Recall", "F1-Score"]
    df_plot = df_comp.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette=["#2563EB", "#059669", "#D97706"], ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison (Test Split)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "models_performance_barchart.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Feature Importance for Random Forest
    if "Random Forest" in models_dict and hasattr(models_dict["Random Forest"], "get_feature_importances"):
        rf_fi = models_dict["Random Forest"].get_feature_importances().head(10)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(rf_fi["Feature"][::-1], rf_fi["Importance"][::-1], color="#059669", edgecolor="#0F172A", alpha=0.85)
        ax.set_title("Random Forest Top 10 Features (Gini Importance)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(output_dir / "rf_feature_importance.png", dpi=200, bbox_inches="tight")
        plt.close()

    logger.info("Evaluation complete! Results saved to %s", comp_csv)
    return df_comp


def run_training_pipeline(models_to_train: str = "all", dev_mode: bool = False):
    """
    Master pipeline:
    1. Loads data and generates EDA visualizations
    2. Runs zero-leakage preprocessing
    3. Trains selected models: Logistic Regression, Random Forest, ANN, Autoencoder
    4. Evaluates all models on held-out test data and saves benchmarks
    """
    logger.info("=== FRAUD DETECTION TRAINING PIPELINE ===")
    nrows = config.DEV_SAMPLE_SIZE if dev_mode else None
    raw_df = load_raw_dataset(nrows=nrows)

    logger.info("Generating EDA charts...")
    generate_eda_visualizations(raw_df)

    logger.info("Executing preprocessing pipeline...")
    data = run_full_preprocessing_pipeline(raw_df)

    trained_models = {}

    # ML Models
    if models_to_train in ["all", "ml"]:
        lr = FraudLogisticRegression()
        lr.fit(data["X_train_smote"], data["y_train_smote"])
        lr.tune_threshold(data["X_val"], data["y_val"])
        lr.save()
        trained_models["Logistic Regression"] = lr

        rf = FraudRandomForest()
        rf.fit(data["X_train_smote"], data["y_train_smote"])
        rf.tune_threshold(data["X_val"], data["y_val"])
        rf.save()
        trained_models["Random Forest"] = rf

    # Supervised ANN
    if models_to_train in ["all", "dl", "ann"]:
        ann = FraudANNClassifier(epochs=5 if dev_mode else config.ANN_PARAMS["epochs"])
        ann.fit(data["X_train_smote"], data["y_train_smote"], data["X_val"], data["y_val"])
        ann.save()
        trained_models["Artificial Neural Network"] = ann

    # Unsupervised Autoencoder
    if models_to_train in ["all", "dl", "autoencoder"]:
        ae = FraudAutoencoderDetector(epochs=5 if dev_mode else config.AUTOENCODER_PARAMS["epochs"])
        ae.fit(data["X_train_legit"], data["X_val"], data["y_val"])
        ae.save()
        trained_models["Autoencoder"] = ae

    # Evaluate on held-out test split
    if trained_models:
        evaluate_and_plot_all_models(trained_models, data["X_test"], data["y_test"])
    logger.info("Master Training Pipeline successfully completed.")
