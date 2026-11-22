# Credit Card Fraud Detection

A Python machine-learning project that classifies credit-card transactions as legitimate or potentially fraudulent using transaction attributes and historical transaction-activity features. The current reproducible baseline uses a **Random Forest classifier** and a chronological train/test split.

## Project objective

Identify potentially fraudulent transactions while evaluating performance with fraud-focused metrics. Because fraudulent transactions are relatively rare, accuracy alone is not a sufficient measure; precision, recall, F1-score, average precision (PR-AUC) and the confusion matrix should also be reviewed.

## Dataset files

| File | Records in supplied data | Current use |
|---|---:|---|
| `Transaction.csv` | 100,000 | Used by the current training script |
| `Customer.csv` | 10,127 | Included as an additional dataset; not currently integrated into the model |
| `Call_history.csv` | 101,174 | Included as an additional dataset; not currently integrated into the model |

The model currently uses `Transaction.csv` only. The customer and call-history datasets are not joined because the supplied implementation does not establish a verified transaction-to-customer mapping. They should not be described as model inputs until a valid join and feature pipeline are implemented.

## Methodology

1. Load transaction records and clean the amount and transaction timestamp.
2. Create time-based features such as hour, weekday and night-time indicator.
3. Create log-transformed transaction amount and trailing 1-hour/24-hour transaction-count and amount features. Rolling features use prior transactions only.
4. Encode categorical transaction attributes using the training period only; unseen test-period categories are safely ignored.
5. Split records chronologically: earlier records for training and later records for testing.
6. Train a Random Forest classifier with `class_weight="balanced"`.
7. Evaluate using a confusion matrix, precision, recall, F1-score, average precision and ROC-AUC.
8. Save the trained model, feature order, test predictions and metrics.

**Leakage prevention:** `Fraud` is the target label and is explicitly excluded from the model's input features. The chronological split is created before fitting categorical encodings, so future test-only categories do not influence the training feature schema. Do not train or evaluate a model with the target column included in `X`.

## Repository structure

```text
credit-card-fraud-detection/
├── Transaction.csv
├── Customer.csv
├── Call_history.csv
├── notebook.ipynb
├── train_model.py
├── requirements.txt
├── README.md
└── outputs/                 # created when training runs
    ├── fraud_model.joblib
    ├── fraud_predictions.csv
    └── metrics.json
```

The currently published GitHub repository contains older generated artifacts (`fraud_model.joblib`, `final_features.parquet` and `fraud_predictions.csv`). These were produced by the earlier notebook, whose model inputs included the target label. **Delete those three old files from GitHub** when replacing the project files; do not use them for performance claims or inference. This cleaned project package intentionally omits them so they cannot be mistaken for valid outputs.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Debalina27/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the model

Keep `Transaction.csv` in the repository root, then run:

```bash
python train_model.py
```

The script creates an `outputs/` directory containing:

- `fraud_model.joblib` — trained model and its expected feature-column order.
- `fraud_predictions.csv` — test-set actual labels, predicted labels and fraud probabilities.
- `metrics.json` — test-set evaluation metrics.

## Important limitations

- This is an experimental academic baseline, not a production payment-fraud prevention system.
- The supplied customer and call-history files are not currently used by the model.
- Results must be regenerated after the leakage fix. Do not quote metrics from the previously committed model or prediction file as valid performance results.
- A chronological holdout is used, but further validation, threshold tuning, drift monitoring, and testing on genuinely unseen data are needed before real-world use.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest
- Joblib

## License

Add a license file if you intend to permit reuse of this project. Until then, no license is specified.
