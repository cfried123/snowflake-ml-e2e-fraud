"""XGBoost fraud-model training job (runs as an ML Job on SPCS).

Pipeline: chronological train/validation/test split -> fit on train ->
pick APPROVE/REVIEW/DECLINE thresholds on validation -> report metrics on the
most recent (test) slice -> persist model, thresholds, metrics, feature
importance and run-keyed test predictions.
"""

import json
import os
import pickle
import time

import pandas as pd
import xgboost as xgb
from snowflake.ml.experiment import ExperimentTracking
from snowflake.snowpark import Session

from utils import (
    BASELINE_SCORE_COL,
    FEATURE_COLS,
    ID_COL,
    LABEL_COL,
    TIMESTAMP_COL,
    calculate_fraud_metrics,
    compute_feature_importance,
    select_decision_thresholds,
    time_based_split,
)

DATABASE = os.environ.get("DATABASE", "DEMO_FRAUD_DETECTION_DB")
SCHEMA = os.environ.get("SCHEMA", "ML_PIPELINE")
MODEL_STAGE = os.environ.get("MODEL_STAGE", f"@{DATABASE}.{SCHEMA}.ML_MODELS")
MODEL_NAME = os.environ.get("MODEL_NAME", "FRAUD_DETECTION_MODEL")
MODEL_FILE = "fraud_model.pkl"
METRICS_FILE = "metrics.pkl"
IMPORTANCE_FILE = "feature_importance.json"
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "FRAUD_DETECTION_EXPERIMENT")
VERSION_NAME = os.environ.get("VERSION_NAME", None)
REGISTER_MODEL = os.environ.get("REGISTER_MODEL", "false").lower() == "true"
RUN_ID = os.environ.get("RUN_ID") or VERSION_NAME or f"run_{int(time.time())}"
CHAMPION_VERSION = os.environ.get("CHAMPION_VERSION") or None
TEST_FRACTION = float(os.environ.get("TEST_FRACTION", "0.2"))
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.2"))
REVIEW_TARGET_RECALL = float(os.environ.get("REVIEW_TARGET_RECALL", "0.9"))
TEST_DATA_TABLE = f"{DATABASE}.{SCHEMA}.FRAUD_TEST_DATA"

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "aucpr",
}

# Metrics persisted to the registry / used by the champion-challenger gate.
REGISTRY_METRIC_KEYS = [
    "auc_roc",
    "auc_pr",
    "recall_at_1pct_fpr",
    "recall_at_5pct_fpr",
    "precision",
    "recall",
    "f1",
]


def calculate_metrics(y_true, y_proba, threshold):
    return calculate_fraud_metrics(y_true, y_proba, threshold=threshold)


def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def load_training_frame(session: Session) -> pd.DataFrame:
    table = f"{DATABASE}.{SCHEMA}.FRAUD_TRAINING_DATA"
    print(f"\nLoading training data from {table}...")
    pdf = session.table(table).to_pandas()
    print(f"Loaded {len(pdf):,} rows")

    required = FEATURE_COLS + [LABEL_COL, TIMESTAMP_COL, ID_COL]
    missing = [c for c in required if c not in pdf.columns]
    if missing:
        raise ValueError(
            f"FRAUD_TRAINING_DATA is missing required columns {missing}. "
            "Re-run notebooks/02_feature_engineering.ipynb to rebuild it."
        )
    pdf[TIMESTAMP_COL] = pd.to_datetime(pdf[TIMESTAMP_COL])
    return pdf


def evaluate_champion(session, X_test, y_test):
    """Score the current champion on this run's holdout for a like-for-like gate.

    Returns ``None`` when there is no champion or it cannot score the current
    feature set (e.g. it was trained on an older feature contract).
    """
    if not CHAMPION_VERSION:
        return None
    try:
        from snowflake.ml.registry import Registry

        reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
        champion = reg.get_model(MODEL_NAME).version(CHAMPION_VERSION).load(force=True)
        champion_features = list(champion.get_booster().feature_names or [])
        if champion_features != list(X_test.columns):
            print(
                f"Champion {CHAMPION_VERSION} uses a different feature set; "
                "skipping same-holdout comparison."
            )
            return None
        proba = champion.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test.values, proba, threshold=0.5)
        print_metrics(f"Champion {CHAMPION_VERSION} on this holdout", metrics)
        return metrics
    except Exception as e:  # noqa: BLE001 - comparison is best-effort
        print(f"Could not evaluate champion {CHAMPION_VERSION}: {type(e).__name__}")
        return None


def save_test_predictions(session, test_df, y_proba, thresholds):
    """Append this run's holdout (keyed by RUN_ID) to FRAUD_TEST_DATA."""
    out = test_df[[ID_COL, TIMESTAMP_COL] + FEATURE_COLS + [LABEL_COL]].copy()
    out.insert(0, "RUN_ID", RUN_ID)
    out.insert(1, "MODEL_VERSION", VERSION_NAME or "")
    out["FRAUD_SCORE"] = y_proba.astype(float)
    out["PREDICTED_FRAUD"] = (y_proba >= thresholds["decline_threshold"]).astype(int)
    out = out.reset_index(drop=True)

    existing = session.sql(
        f"SHOW TABLES LIKE 'FRAUD_TEST_DATA' IN SCHEMA {DATABASE}.{SCHEMA}"
    ).collect()
    if existing:
        cols = [c.upper() for c in session.table(TEST_DATA_TABLE).columns]
        if "RUN_ID" not in cols:
            legacy = f"{TEST_DATA_TABLE}_LEGACY_{int(time.time())}"
            print(f"Renaming un-versioned {TEST_DATA_TABLE} to {legacy}")
            session.sql(f"ALTER TABLE {TEST_DATA_TABLE} RENAME TO {legacy}").collect()

    session.create_dataframe(out).write.mode("append").save_as_table(
        TEST_DATA_TABLE, column_order="name"
    )
    print(f"Appended {len(out):,} test rows to {TEST_DATA_TABLE} (RUN_ID={RUN_ID})")


def main():
    print("Starting fraud detection model training...")

    session = Session.builder.getOrCreate()
    print("Connected to Snowflake")

    exp = ExperimentTracking(
        session=session, database_name=DATABASE, schema_name=SCHEMA
    )
    exp.set_experiment(EXPERIMENT_NAME)
    print(f"Experiment: {EXPERIMENT_NAME} | Run: {RUN_ID}")

    pdf = load_training_frame(session)
    print(
        f"Using {len(FEATURE_COLS)} features (baseline {BASELINE_SCORE_COL} excluded)"
    )

    train_df, val_df, test_df = time_based_split(pdf, TEST_FRACTION, VAL_FRACTION)
    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(
            f"{name}: {len(part):,} rows | fraud {part[LABEL_COL].mean() * 100:.2f}% | "
            f"{part[TIMESTAMP_COL].min()} -> {part[TIMESTAMP_COL].max()}"
        )

    X_train, y_train = train_df[FEATURE_COLS], train_df[LABEL_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[LABEL_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[LABEL_COL]

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    print("Training complete!")

    val_proba = model.predict_proba(X_val)[:, 1]
    thresholds = select_decision_thresholds(
        y_val.values, val_proba, review_target_recall=REVIEW_TARGET_RECALL
    )
    print(
        f"\nDecision bands (from validation): APPROVE < {thresholds['review_threshold']:.4f} "
        f"<= REVIEW < {thresholds['decline_threshold']:.4f} <= DECLINE"
    )
    val_metrics = calculate_metrics(
        y_val.values, val_proba, thresholds["decline_threshold"]
    )
    print_metrics("Validation Metrics", val_metrics)

    test_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(
        y_test.values, test_proba, thresholds["decline_threshold"]
    )
    metrics.update(
        {
            "review_threshold": thresholds["review_threshold"],
            "decline_threshold": thresholds["decline_threshold"],
            "review_target_recall": thresholds["review_target_recall"],
            "run_id": RUN_ID,
        }
    )
    print_metrics("Test Metrics (most recent slice)", metrics)

    baseline = calculate_fraud_metrics(
        y_test.values, test_df[BASELINE_SCORE_COL].astype(float).values, threshold=3.0
    )
    print(
        f"\nBaseline {BASELINE_SCORE_COL}: AUC-PR={baseline['auc_pr']:.4f} "
        f"AUC-ROC={baseline['auc_roc']:.4f} "
        f"Recall@1%FPR={baseline['recall_at_1pct_fpr']:.4f}"
    )
    metrics["baseline_risk_score_auc_pr"] = baseline["auc_pr"]
    metrics["baseline_risk_score_auc_roc"] = baseline["auc_roc"]

    champion_metrics = evaluate_champion(session, X_test, y_test)
    if champion_metrics:
        metrics["champion_version"] = CHAMPION_VERSION
        for k in ("auc_pr", "recall_at_1pct_fpr", "auc_roc"):
            metrics[f"champion_{k}"] = champion_metrics[k]

    importance = compute_feature_importance(model, X_test)
    print("\n=== Top features (mean |SHAP| on test) ===")
    print(importance.head(15).to_string(index=False))

    os.makedirs("/tmp/model", exist_ok=True)
    model_path = f"/tmp/model/{MODEL_FILE}"
    metrics_path = f"/tmp/model/{METRICS_FILE}"
    importance_path = f"/tmp/model/{IMPORTANCE_FILE}"

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": list(FEATURE_COLS),
                "decision_thresholds": thresholds,
                "run_id": RUN_ID,
            },
            f,
        )
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    with open(importance_path, "w") as f:
        json.dump(importance.to_dict(orient="records"), f, indent=2)

    run_params = {
        **{k: str(v) for k, v in XGB_PARAMS.items()},
        "scale_pos_weight": str(scale_pos_weight),
        "run_type": "baseline",
        "run_id": RUN_ID,
        "split": f"time_based(test={TEST_FRACTION},val={VAL_FRACTION})",
        "n_features": str(len(FEATURE_COLS)),
    }
    logged_metrics = {
        f"test_{k}": float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    logged_metrics.update({f"val_{k}": val_metrics[k] for k in REGISTRY_METRIC_KEYS})
    with exp.start_run(RUN_ID):
        exp.log_params(run_params)
        exp.log_metrics(logged_metrics)
        for path in (importance_path, metrics_path):
            try:
                exp.log_artifact(path, artifact_path="evaluation")
            except Exception as e:  # noqa: BLE001 - artifacts are also on the stage
                print(
                    f"log_artifact({os.path.basename(path)}) skipped: {type(e).__name__}"
                )
    print(f"\nLogged run '{RUN_ID}' to experiment {EXPERIMENT_NAME}")

    session.sql(f"CREATE STAGE IF NOT EXISTS {DATABASE}.{SCHEMA}.ML_MODELS").collect()
    for stage_dir in (MODEL_STAGE, f"{MODEL_STAGE}/runs/{RUN_ID}"):
        for path in (model_path, metrics_path, importance_path):
            session.file.put(path, stage_dir, auto_compress=False, overwrite=True)
    print(f"Uploaded artifacts to {MODEL_STAGE} and {MODEL_STAGE}/runs/{RUN_ID}")

    save_test_predictions(session, test_df, test_proba, thresholds)

    if REGISTER_MODEL and VERSION_NAME:
        from snowflake.ml.model import type_hints
        from snowflake.ml.registry import Registry

        print(f"\nRegistering model to registry as {VERSION_NAME}...")
        reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
        sample_df = session.create_dataframe(X_test.head(100).reset_index(drop=True))
        registry_metrics = {k: metrics[k] for k in REGISTRY_METRIC_KEYS}
        registry_metrics.update(
            {
                "auc": metrics["auc_roc"],
                "review_threshold": thresholds["review_threshold"],
                "decline_threshold": thresholds["decline_threshold"],
                "run_id": RUN_ID,
            }
        )
        reg.log_model(
            model=model,
            model_name=MODEL_NAME,
            version_name=VERSION_NAME,
            sample_input_data=sample_df,
            metrics=registry_metrics,
            comment=(
                f"Challenger - AUC-PR: {metrics['auc_pr']:.4f}, "
                f"Recall@1%FPR: {metrics['recall_at_1pct_fpr']:.4f}"
            ),
            target_platforms=[type_hints.TargetPlatform.SNOWPARK_CONTAINER_SERVICES],
        )
        print(f"Registered: {MODEL_NAME}:{VERSION_NAME}")

    print("\n=== Training Job Complete ===")
    print(f"Model artifact: {MODEL_STAGE}/runs/{RUN_ID}/{MODEL_FILE}")
    print(f"Metrics artifact: {MODEL_STAGE}/runs/{RUN_ID}/{METRICS_FILE}")
    print(
        f"AUC-PR: {metrics['auc_pr']:.4f} | Recall@1%FPR: {metrics['recall_at_1pct_fpr']:.4f} "
        f"| AUC-ROC: {metrics['auc_roc']:.4f}"
    )


if __name__ == "__main__":
    main()
