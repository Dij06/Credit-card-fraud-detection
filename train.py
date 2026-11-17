"""
Master Execution Script for Credit Card Fraud Detection.
Invokes the consolidated training and evaluation pipeline.

Usage:
    python train.py                 # Trains and evaluates all 4 models
    python train.py --models ml     # Trains only Logistic Regression and Random Forest
    python train.py --models dl     # Trains only ANN and Autoencoder
    python train.py --dev           # Fast development mode
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import run_training_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Training Pipeline")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        choices=["all", "ml", "dl", "ann", "autoencoder"],
        help="Models to train: all, ml, dl, ann, autoencoder (default: all)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Fast verification mode with smaller sample size"
    )
    args = parser.parse_args()

    run_training_pipeline(models_to_train=args.models, dev_mode=args.dev)
