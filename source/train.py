import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from snowflake.snowpark import Session
from snowflake.ml.experiment import ExperimentTracking

DATABASE = os.environ.get("DATABASE", "DEMO_FRAUD_DETECTION_DB")
SCHEMA = os.environ.get("SCHEMA", "ML_PIPELINE")
MODEL_STAGE = os.environ.get("MODEL_STAGE", f"@{DATABASE}.{SCHEMA}.ML_MODELS")
MODEL_FILE = "fraud_model.pkl"
METRICS_FILE = "metrics.pkl"
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "FRAUD_DETECTION_EXPERIMENT")
VERSION_NAME = os.environ.get("VERSION_NAME", None)
REGISTER_MODEL = os.environ.get("REGISTER_MODEL", "false").lower() == "true"

FEATURE_COLS = [
    "TRANSACTION_AMOUNT",
    "TRANSACTION_HOUR",
    "TRANSACTION_DAY_OF_WEEK",
    "IS_WEEKEND",
    "CARD_PRESENT",
    "IS_INTERNATIONAL",
    "ACCOUNT_AGE_DAYS",
    "CREDIT_SCORE",
    "AVG_MONTHLY_TRANSACTIONS",
    "AVG_TRANSACTION_AMOUNT",
    "NUM_CARDS_ON_FILE",
    "IS_PREMIUM_MEMBER",
    "NUM_PREVIOUS_FRAUD_REPORTS",
    "AVG_TXN_AMOUNT",
    "STD_TXN_AMOUNT",
    "MAX_TXN_AMOUNT",
    "MIN_TXN_AMOUNT",
    "TOTAL_TXN_AMOUNT",
    "TXN_COUNT",
    "AVG_TXN_HOUR",
    "STD_TXN_HOUR",
    "INTL_TXN_COUNT",
    "INTL_TXN_RATIO",
    "CARD_PRESENT_RATIO",
    "AMOUNT_TO_AVG_RATIO",
    "AMOUNT_TO_MAX_RATIO",
    "IS_HIGH_VALUE_TXN",
    "IS_ODD_HOUR",
    "RISK_SCORE",
]
LABEL_COL = "IS_FRAUD"


def calculate_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "fraud_detection_rate": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }


def main():
    print("Starting fraud detection model training...")

    session = Session.builder.getOrCreate()
    print("Connected to Snowflake")

    exp = ExperimentTracking(
        session=session, database_name=DATABASE, schema_name=SCHEMA
    )
    exp.set_experiment(EXPERIMENT_NAME)
    print(f"Experiment: {EXPERIMENT_NAME}")

    print(f"\nLoading training data from {DATABASE}.{SCHEMA}.FRAUD_TRAINING_DATA...")
    train_pdf = session.table(f"{DATABASE}.{SCHEMA}.FRAUD_TRAINING_DATA").to_pandas()
    print(f"Loaded {len(train_pdf):,} rows")

    available_features = [c for c in FEATURE_COLS if c in train_pdf.columns]
    print(f"Using {len(available_features)} features")

    X = train_pdf[available_features].fillna(0)
    y = train_pdf[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(
        f"Fraud ratio - Train: {y_train.mean() * 100:.2f}% | Test: {y_test.mean() * 100:.2f}%"
    )

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    print("Training complete!")

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    metrics = calculate_metrics(y_test.values, y_pred, y_proba)
    print("\n=== Fraud Detection Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    import time as _time

    run_name = f"baseline_{int(_time.time())}"

    baseline_params = {
        "n_estimators": "200",
        "max_depth": "5",
        "learning_rate": "0.05",
        "scale_pos_weight": str(scale_pos_weight),
        "subsample": "0.8",
        "colsample_bytree": "0.8",
    }
    with exp.start_run(run_name):
        exp.log_params({**baseline_params, "run_type": "baseline"})
        exp.log_metrics(
            {
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_auc_roc": metrics["auc_roc"],
                "test_auc_pr": metrics["auc_pr"],
            }
        )
    print(f"\nLogged run '{run_name}' to experiment {EXPERIMENT_NAME}")

    os.makedirs("/tmp/model", exist_ok=True)
    model_path = f"/tmp/model/{MODEL_FILE}"
    metrics_path = f"/tmp/model/{METRICS_FILE}"

    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": available_features}, f)
    print(f"\nModel saved to {model_path}")

    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")

    session.sql(f"CREATE STAGE IF NOT EXISTS {DATABASE}.{SCHEMA}.ML_MODELS").collect()
    session.file.put(model_path, MODEL_STAGE, auto_compress=False, overwrite=True)
    session.file.put(metrics_path, MODEL_STAGE, auto_compress=False, overwrite=True)
    print(f"\nUploaded artifacts to {MODEL_STAGE}")

    test_df = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )
    test_spdf = session.create_dataframe(test_df)
    test_spdf.write.mode("overwrite").save_as_table(
        f"{DATABASE}.{SCHEMA}.FRAUD_TEST_DATA"
    )
    print(f"Saved test data to {DATABASE}.{SCHEMA}.FRAUD_TEST_DATA")

    if REGISTER_MODEL and VERSION_NAME:
        from snowflake.ml.registry import Registry
        from snowflake.ml.model import type_hints

        print(f"\nRegistering model to registry as {VERSION_NAME}...")
        reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
        sample_df = session.create_dataframe(X_test.head(100))

        reg.log_model(
            model=model,
            model_name="FRAUD_DETECTION_MODEL",
            version_name=VERSION_NAME,
            sample_input_data=sample_df,
            metrics={
                "auc": metrics["auc_roc"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
            },
            comment=f"Challenger model - AUC: {metrics['auc_roc']:.4f}",
            target_platforms=[type_hints.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
        )
        print(f"Registered: FRAUD_DETECTION_MODEL:{VERSION_NAME}")

    print("\n=== Training Job Complete ===")
    print(f"Model artifact: {MODEL_STAGE}/{MODEL_FILE}")
    print(f"Metrics artifact: {MODEL_STAGE}/{METRICS_FILE}")
    print(f"AUC: {metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
