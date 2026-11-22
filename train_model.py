"""Train a leakage-safe credit-card fraud detection baseline.

Uses Transaction.csv only. Customer.csv and Call_history.csv are included in
this repository but are not joined into this model because no verified key
mapping is implemented in the supplied notebook.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Transaction.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TARGET = "Fraud"
DROP_COLUMNS = ["Transaction ID", "DateTime", "Shipping Address", TARGET]


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Date", "Time", "Amount", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Parse transaction time and clean the amount.
    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip(),
        format="%d-%b-%y %H",
        errors="coerce",
    )
    df["Amount"] = pd.to_numeric(
        df["Amount"].astype(str).str.replace("£", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["DateTime", "Amount", TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])].copy()
    df = df.sort_values("DateTime").reset_index(drop=True)

    df["Hour"] = df["DateTime"].dt.hour
    df["WeekdayIndex"] = df["DateTime"].dt.weekday
    df["IsNight"] = df["Hour"].between(0, 6).astype(int)
    df["LogAmount"] = np.log1p(df["Amount"].clip(lower=0))

    # Shift(1) ensures each transaction only uses earlier transaction history.
    time_indexed = df.set_index("DateTime")
    df["TxnsLast1H"] = time_indexed["Amount"].rolling("1h", closed="left").count().to_numpy()
    df["TxnsLast24H"] = time_indexed["Amount"].rolling("24h", closed="left").count().to_numpy()
    df["AmtLast1H"] = time_indexed["Amount"].rolling("1h", closed="left").sum().to_numpy()
    df["AmtLast24H"] = time_indexed["Amount"].rolling("24h", closed="left").sum().to_numpy()

    return df


def main() -> None:
    df = load_and_prepare(DATA_PATH)

    # Chronological holdout: oldest 80% for training, newest 20% for testing.
    split_position = int(len(df) * 0.8)
    train = df.iloc[:split_position].copy()
    test = df.iloc[split_position:].copy()
    if train.empty or test.empty:
        raise ValueError("Not enough valid rows for a chronological train/test split.")

    # Separate the chronological holdout BEFORE fitting categorical encoding.
    # This prevents categories appearing only in the future test period from
    # influencing the training feature schema.
    feature_source = df.drop(columns=DROP_COLUMNS + ["Date", "Time"], errors="ignore")
    X_raw = feature_source.drop(columns=[TARGET], errors="ignore")
    categorical = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    X_train_raw = X_raw.iloc[:split_position].copy()
    X_test_raw = X_raw.iloc[split_position:].copy()

    X_train = pd.get_dummies(
        X_train_raw, columns=categorical, drop_first=True, dtype=int
    )
    X_test = pd.get_dummies(
        X_test_raw, columns=categorical, drop_first=True, dtype=int
    )
    # Ignore unseen test-only categories safely; preserve the training schema.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Explicitly exclude target and identifiers from model inputs.
    feature_columns = X_train.columns.tolist()
    y_train = train[TARGET]
    y_test = test[TARGET]

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "rows_used": int(len(df)),
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "fraud_rate_overall": float(df[TARGET].mean()),
        "fraud_rate_test": float(y_test.mean()),
        "feature_count": int(len(feature_columns)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": report,
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
    }

    # Save model together with the exact feature order used at training time.
    joblib.dump({"model": model, "feature_columns": feature_columns}, OUTPUT_DIR / "fraud_model.joblib")
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    results = test[[c for c in ["Transaction ID", "DateTime", "Amount", TARGET] if c in test.columns]].copy()
    results["Predicted_Fraud"] = predictions
    results["Fraud_Probability"] = probabilities
    results.to_csv(OUTPUT_DIR / "fraud_predictions.csv", index=False)

    print("Training completed.")
    print(f"Rows: {len(df):,} | Train: {len(train):,} | Test: {len(test):,}")
    print(f"Features: {len(feature_columns)}")
    print("Confusion matrix (labels 0, 1):")
    print(np.array(metrics["confusion_matrix_labels_0_1"]))
    print(classification_report(y_test, predictions, digits=4, zero_division=0))
    print(f"Average precision: {metrics['average_precision']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
