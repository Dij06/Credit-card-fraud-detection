# Integrated Credit Card Fraud Detection Platform (FinSecure)
### Using Machine Learning and Deep Learning Techniques

An institutional-grade credit card fraud surveillance and anomaly detection system built with Python, Scikit-learn, TensorFlow/Keras, and Streamlit. Satisfies academic and practical requirements for undergraduate Computer Science & Engineering Machine Learning and Deep Learning PBL submissions.

---

## 1. Project Overview & Architecture

Payment fraud accounts for billions in global financial losses annually. This platform combines classical supervised Machine Learning algorithms and Deep Learning neural architectures to monitor transactions, capture anomalous spending behavior, and protect cardholders without generating excessive false declines.

```
                          [ Incoming Transaction ]
                                     │
                                     ▼
                     [ RobustScaler Transformation ]
                     (Zero-Leakage; Fits Train Only)
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   [ Logistic Regression ]    [ Random Forest ]     [ Deep Neural Network ]
   (Linear Class-Weighted)    (100 Decision Trees)  (Dense + Dropout Layers)
           │                         │                         │
     Fraud Probability         Fraud Probability         Fraud Probability
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     │
                         [ Deep Autoencoder (AE) ]
                         (Unsupervised Reconstruction)
                         Trained on Legitimately Clean
                                     │
                             Reconstruction MSE
                                     │
                                     ▼
                      [ Calibrated Decision Engine ]
                     (Threshold-Tuned on Validation)
                                     │
                                     ▼
                [ Clear Business Verdict: Legit or Fraud ]
```

---

## 2. The 5-Step Fraud Detection Workflow

1. **Step 1: Transaction Input**  
   The application receives transaction details: elapsed seconds (`Time`), charge amount (`Amount`), and 28 privacy-preserving PCA features (`V1` to `V28`).
2. **Step 2: Data Processing**  
   The transaction is scaled using a pre-fitted `RobustScaler` (median & interquartile range) to handle financial outliers without data leakage.
3. **Step 3: Fraud Analysis**  
   The active detection model computes a prediction score: supervised classifiers estimate fraud likelihood ($0.0$ to $1.0$), while the Autoencoder computes a reconstruction error (MSE).
4. **Step 4: Decision**  
   The score is evaluated against an optimal threshold calibrated on validation data (minimizing false alarms on clean transactions).
5. **Step 5: Result**  
   The transaction is categorized with a clear status:
   - **Transaction appears legitimate** (Cleared for settlement)
   - **Potential fraudulent transaction detected** (Flagged for block/manual review)

---

## 3. Simplified Repository Structure

```text
credit-card-fraud-detection/
│
├── dataset/
│   ├── .gitkeep
│   └── creditcard.csv                 # Kaggle benchmark (284,807 transactions)
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py             # Ingestion, schema audit, zero-leakage transforms, SMOTE, EDA
│   ├── models.py                      # Logistic Regression, Random Forest, ANN, Autoencoder
│   ├── training.py                    # Pipeline orchestration, validation thresholding, test evaluation
│   └── prediction.py                  # Single & batch inference engine, consensus scoring
│
├── app/
│   ├── __init__.py
│   └── dashboard.py                   # Self-contained modern fintech Streamlit application
│
├── models/
│   ├── scaler.joblib                  # Fitted RobustScaler
│   ├── preprocessing_meta.joblib      # Preprocessing metadata
│   ├── logistic_regression.joblib     # Calibrated Logistic Regression
│   ├── random_forest.joblib           # Calibrated Random Forest
│   ├── ann_fraud_detector.keras       # Trained ANN weights
│   ├── ann_fraud_detector.meta.joblib # ANN metadata & optimal threshold
│   ├── autoencoder_anomaly.keras      # Trained Autoencoder weights
│   └── autoencoder_threshold.joblib   # Calibrated reconstruction threshold
│
├── results/
│   ├── graphs/
│   │   ├── eda_class_distribution.png
│   │   ├── eda_amount_distribution.png
│   │   ├── eda_feature_correlations.png
│   │   ├── eda_time_distribution.png
│   │   ├── ann_training_history.png
│   │   ├── autoencoder_reconstruction_error.png
│   │   ├── confusion_matrices_comparison.png
│   │   ├── roc_pr_curves_comparison.png
│   │   ├── models_performance_barchart.png
│   │   └── rf_feature_importance.png
│   ├── metrics/
│   │   ├── dataset_summary.json
│   │   ├── hourly_activity.csv
│   │   ├── model_comparison_metrics.json
│   │   ├── model_comparison_table.csv
│   │   ├── sample_batch_template.csv
│   │   └── test_sample_pool.csv
│   └── model_comparison.csv           # Master benchmark table
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_experimental_analysis.ipynb
│
├── tests/
│   └── test_project.py                # Consolidated unit test suite (12 tests)
│
├── config.py                          # Master configuration & hyperparameters
├── train.py                           # CLI training pipeline entrypoint
├── requirements.txt                   # Verified minimal dependencies
├── .gitignore                         # Git exclusion rules
├── LICENSE                            # MIT License
└── README.md
```

---

## 4. Empirical Benchmark on Held-Out Test Set

Evaluated on **42,722 unseen transactions** containing **74 genuine fraud cases** (0.173% prevalence):

| Metric | Random Forest (ML) | Deep Learning ANN | Logistic Regression (ML) | Autoencoder (Unsupervised) |
| :--- | :---: | :---: | :---: | :---: |
| **Model Type** | Ensemble ML | Supervised DL | Linear ML | Anomaly Detection DL |
| **Operating Threshold** | **0.8456** | **0.9000** | **0.9000** | **3.0607 (MSE)** |
| **Accuracy** | **99.94%** | 99.86% | 99.49% | 99.77% |
| **Recall (Sensitivity)** | 72.97% (54/74) | **83.78% (62/74)** | **85.14% (63/74)** | 70.27% (52/74) |
| **Precision** | **91.53%** | 57.41% | 23.42% | 40.31% |
| **F1-Score** | **0.8120** | 0.6813 | 0.3673 | 0.5123 |
| **ROC-AUC** | **0.9756** | 0.9748 | 0.9641 | 0.9144 |
| **PR-AUC** | **0.7953** | 0.7745 | 0.7821 | 0.4982 |
| **True Positives (TP)** | 54 | **62** | 63 | 52 |
| **False Positives (FP)** | **5** | 46 | 206 | 77 |
| **False Negatives (FN)**| 20 | **12** | 11 | 22 |
| **True Negatives (TN)** | **42,643** | 42,602 | 42,442 | 42,571 |

### Operational Findings:
- **Random Forest** is the recommended production model: highest **Precision (91.53%)** and highest **F1-Score (0.8120)**, generating only **5 false alarms** across 42,648 clean transactions.
- **Artificial Neural Network (ANN)** provides maximum fraud capture (**83.78% Recall**), intercepting 62 out of 74 attacks.
- **Autoencoder** demonstrates strong unsupervised capability, capturing **70.27% of fraud cases** purely by measuring reconstruction deviation from legitimate behavior.

---

## 5. Quickstart & Execution Guide

### A. Environment Setup
```bash
# Clone the repository and navigate into the project directory
git clone https://github.com/Dij06/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection

# Install required dependencies
pip install -r requirements.txt
```

### B. Dataset Setup (Optional for Inference, Required for Model Training)
The repository includes all pre-trained model weights, test sample pools, and aggregated metrics, allowing the application to run immediately out-of-the-box upon cloning without downloading the large historical dataset.

If you wish to re-train the models from scratch using `python train.py`:
1. Download the European Credit Card Fraud Detection dataset (`creditcard.csv`) from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place `creditcard.csv` directly into the `dataset/` directory:
   ```text
   Credit-card-fraud-detection/
   └── dataset/
       └── creditcard.csv
   ```

### C. Run Automated Unit Tests
```bash
python -m unittest tests/test_project.py -v
```

### D. Launch the Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

### E. Re-train Models (Optional)
```bash
python train.py
```

---

## 6. Dashboard Sections

1. **Dashboard (Dataset Overview)**: Macro-level monitoring across 284,807 European cardholder transactions, key KPIs, class distribution, 48-hour volume activity, and confirmed fraud occurrence over time.
2. **Detect Fraud**: Individual transaction screening supporting sample loading (Legitimate, Fraudulent, Random), custom amount and elapsed time inputs with synthetic scenario alerts, and multi-model consensus scoring.
3. **Batch Detection**: High-throughput automated screening for CSV files with schema validation, model selection, detailed predictions, and downloadable scored CSV reports.
4. **EDA & Model Performance**: Two comprehensive tabs presenting 6 core exploratory distributions alongside empirical benchmarks and confusion matrices for all 4 ML and DL architectures.
5. **About**: Project overview, academic PBL specifications, system workflow, and technology stack.
