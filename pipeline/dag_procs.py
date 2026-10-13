"""Stored-procedure handlers for the champion/challenger retraining DAG.

Registered from ``pipeline/ml_dag.ipynb`` with
``session.sproc.register_from_file`` so the logic lives in version control
(and is unit-tested) instead of inline SQL strings.

Promotion policy (see README "Promotion criteria"): a challenger is promoted
when AUC-PR **or** recall at 1% FPR improves by more than ``MIN_IMPROVEMENT``
(relative), and neither metric regresses by more than ``MAX_REGRESSION``.
AUC-ROC is logged for reference only.
"""

import math
import os
import pickle
import time
from datetime import datetime
from typing import Dict, Optional

MODEL_NAME = "FRAUD_DETECTION_MODEL"
MIN_IMPROVEMENT = 0.01
MAX_REGRESSION = 0.01
GATE_METRICS = ("auc_pr", "recall_at_1pct_fpr")

PSI_DRIFT_THRESHOLD = 0.2
DRIFT_RETRAIN_COOLDOWN_HOURS = 12

RUN_LOG_COLUMNS = {
    "RUN_ID": "VARCHAR",
    "RUN_TIMESTAMP": "TIMESTAMP_NTZ",
    "STEP_NAME": "VARCHAR",
    "STATUS": "VARCHAR",
    "CHALLENGER_VERSION": "VARCHAR",
    "CHAMPION_VERSION": "VARCHAR",
    "CHALLENGER_AUC": "FLOAT",
    "CHAMPION_AUC": "FLOAT",
    "CHALLENGER_AUC_PR": "FLOAT",
    "CHAMPION_AUC_PR": "FLOAT",
    "CHALLENGER_RECALL_AT_1PCT_FPR": "FLOAT",
    "CHAMPION_RECALL_AT_1PCT_FPR": "FLOAT",
    "CHALLENGER_DECLINE_THRESHOLD": "FLOAT",
    "CHALLENGER_REVIEW_THRESHOLD": "FLOAT",
    "COMPARISON_BASIS": "VARCHAR",
    "PROMOTED": "BOOLEAN",
    "MESSAGE": "VARCHAR",
}

CHAMPION_TRACKER_COLUMNS = {
    "MODEL_NAME": "VARCHAR",
    "CHAMPION_VERSION": "VARCHAR",
    "CHAMPION_AUC": "FLOAT",
    "CHAMPION_AUC_PR": "FLOAT",
    "CHAMPION_RECALL_AT_1PCT_FPR": "FLOAT",
    "CHAMPION_REVIEW_THRESHOLD": "FLOAT",
    "CHAMPION_DECLINE_THRESHOLD": "FLOAT",
    "UPDATED_AT": "TIMESTAMP_NTZ",
}

DRIFT_LOG_COLUMNS = {
    "CHECK_ID": "VARCHAR",
    "CHECK_TIMESTAMP": "TIMESTAMP_NTZ",
    "MONITOR_NAME": "VARCHAR",
    "FEATURE_NAME": "VARCHAR",
    "METRIC_NAME": "VARCHAR",
    "METRIC_VALUE": "FLOAT",
    "THRESHOLD": "FLOAT",
    "IS_DRIFTED": "BOOLEAN",
}


# ---------------------------------------------------------------------------
# Pure decision logic (unit-tested)
# ---------------------------------------------------------------------------


def relative_change(new: Optional[float], base: Optional[float]) -> float:
    """Relative change of ``new`` vs ``base``; +inf when there is no usable base."""
    if new is None:
        return -math.inf
    if base is None or base <= 0:
        return math.inf if new > 0 else 0.0
    return (new - base) / base


def decide_promotion(
    challenger: Dict[str, Optional[float]],
    champion: Dict[str, Optional[float]],
    min_improvement: float = MIN_IMPROVEMENT,
    max_regression: float = MAX_REGRESSION,
) -> Dict:
    """Champion/challenger gate on AUC-PR and recall@1%FPR."""
    changes = {
        m: relative_change(challenger.get(m), champion.get(m)) for m in GATE_METRICS
    }
    improved = [m for m, c in changes.items() if c > min_improvement]
    regressed = [m for m, c in changes.items() if c < -max_regression]
    promote = bool(improved) and not regressed

    def _fmt(c):
        return "n/a" if math.isinf(c) else f"{c * 100:+.2f}%"

    detail = ", ".join(
        f"{m}={challenger.get(m) or 0:.4f} vs {champion.get(m) or 0:.4f} ({_fmt(c)})"
        for m, c in changes.items()
    )
    if promote:
        reason = f"improved {improved}"
    elif regressed:
        reason = f"regressed {regressed}"
    else:
        reason = f"no metric improved by >{min_improvement * 100:g}%"
    return {
        "promote": promote,
        "changes": changes,
        "message": f"{'PROMOTE' if promote else 'KEEP_CHAMPION'}: {reason}; {detail}",
    }


def drifted_features(
    psi_by_feature: Dict[str, float], threshold: float = PSI_DRIFT_THRESHOLD
) -> Dict[str, float]:
    return {f: v for f, v in psi_by_feature.items() if v is not None and v > threshold}


# ---------------------------------------------------------------------------
# Snowflake helpers
# ---------------------------------------------------------------------------


def _fqn(database: str, schema: str, name: str) -> str:
    return f"{database}.{schema}.{name}"


def ensure_tables(session, database: str, schema: str) -> None:
    """Create tracking tables and add any columns introduced by newer versions."""
    for table, columns in (
        ("PIPELINE_RUN_LOG", RUN_LOG_COLUMNS),
        ("MODEL_CHAMPION_TRACKER", CHAMPION_TRACKER_COLUMNS),
        ("MODEL_DRIFT_LOG", DRIFT_LOG_COLUMNS),
    ):
        fqn = _fqn(database, schema, table)
        cols_sql = ", ".join(f"{c} {t}" for c, t in columns.items())
        session.sql(f"CREATE TABLE IF NOT EXISTS {fqn} ({cols_sql})").collect()
        for c, t in columns.items():
            session.sql(f"ALTER TABLE {fqn} ADD COLUMN IF NOT EXISTS {c} {t}").collect()


def _schema_for(columns: Dict[str, str]):
    from snowflake.snowpark.types import (
        BooleanType,
        DoubleType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    types = {
        "VARCHAR": StringType(),
        "FLOAT": DoubleType(),
        "BOOLEAN": BooleanType(),
        "TIMESTAMP_NTZ": TimestampType(),
    }
    return StructType([StructField(c, types[t]) for c, t in columns.items()])


def append_rows(session, table_fqn: str, columns: Dict[str, str], rows) -> None:
    """Append rows via a typed DataFrame (no string-built SQL values)."""
    rows = [[r.get(c) for c in columns] for r in rows]
    if not rows:
        return
    session.create_dataframe(rows, schema=_schema_for(columns)).write.mode(
        "append"
    ).save_as_table(table_fqn, column_order="name")


def log_step(session, database: str, schema: str, **values) -> None:
    values.setdefault("RUN_ID", datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    values.setdefault("RUN_TIMESTAMP", datetime.utcnow())
    append_rows(
        session, _fqn(database, schema, "PIPELINE_RUN_LOG"), RUN_LOG_COLUMNS, [values]
    )


def get_champion(session, database: str, schema: str) -> Optional[Dict]:
    from snowflake.snowpark import functions as F

    rows = (
        session.table(_fqn(database, schema, "MODEL_CHAMPION_TRACKER"))
        .filter(F.col("MODEL_NAME") == F.lit(MODEL_NAME))
        .collect()
    )
    return rows[0].as_dict() if rows else None


def _float(value) -> Optional[float]:
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Stored procedure handlers
# ---------------------------------------------------------------------------


def train_challenger(session, database: str, schema: str, compute_pool: str) -> str:
    """Run ``source/train.py`` as an ML Job and log challenger metrics."""
    from snowflake.ml.jobs import submit_directory

    ensure_tables(session, database, schema)
    model_stage = f"@{database}.{schema}.ML_MODELS"
    source_stage = f"{model_stage}/source"
    version = f"CHALLENGER_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    champion = get_champion(session, database, schema) or {}

    os.makedirs("/tmp/source", exist_ok=True)
    session.file.get(source_stage, "/tmp/source/")

    job = submit_directory(
        session=session,
        dir_path="/tmp/source",
        entrypoint="train.py",
        compute_pool=compute_pool,
        stage_name=f"{database}.{schema}.ML_JOB_STAGE",
        env_vars={
            "DATABASE": database,
            "SCHEMA": schema,
            "MODEL_STAGE": model_stage,
            "MODEL_NAME": MODEL_NAME,
            "EXPERIMENT_NAME": "FRAUD_DETECTION_EXPERIMENT",
            "VERSION_NAME": version,
            "RUN_ID": version,
            "REGISTER_MODEL": "true",
            "CHAMPION_VERSION": champion.get("CHAMPION_VERSION") or "",
        },
    )

    for _ in range(360):
        if job.status in ("DONE", "FAILED"):
            break
        time.sleep(10)

    if job.status != "DONE":
        log_step(
            session,
            database,
            schema,
            RUN_ID=version,
            STEP_NAME="TRAIN",
            STATUS="FAILED",
            CHALLENGER_VERSION=version,
            MESSAGE=f"ML Job status {job.status}",
        )
        return f"FAILED: {version}"

    session.file.get(f"{model_stage}/runs/{version}/metrics.pkl", "/tmp/metrics/")
    with open("/tmp/metrics/metrics.pkl", "rb") as f:
        metrics = pickle.load(f)

    has_same_holdout = metrics.get("champion_auc_pr") is not None
    log_step(
        session,
        database,
        schema,
        RUN_ID=version,
        STEP_NAME="TRAIN",
        STATUS="SUCCESS",
        CHALLENGER_VERSION=version,
        CHAMPION_VERSION=champion.get("CHAMPION_VERSION"),
        CHALLENGER_AUC=_float(metrics.get("auc_roc")),
        CHAMPION_AUC=_float(metrics.get("champion_auc_roc")),
        CHALLENGER_AUC_PR=_float(metrics.get("auc_pr")),
        CHAMPION_AUC_PR=_float(metrics.get("champion_auc_pr")),
        CHALLENGER_RECALL_AT_1PCT_FPR=_float(metrics.get("recall_at_1pct_fpr")),
        CHAMPION_RECALL_AT_1PCT_FPR=_float(metrics.get("champion_recall_at_1pct_fpr")),
        CHALLENGER_DECLINE_THRESHOLD=_float(metrics.get("decline_threshold")),
        CHALLENGER_REVIEW_THRESHOLD=_float(metrics.get("review_threshold")),
        COMPARISON_BASIS="same_holdout" if has_same_holdout else None,
        MESSAGE=(
            f"AUC-PR={metrics.get('auc_pr', 0):.4f}, "
            f"Recall@1%FPR={metrics.get('recall_at_1pct_fpr', 0):.4f}, "
            f"AUC-ROC={metrics.get('auc_roc', 0):.4f}"
        ),
    )
    return f"SUCCESS: {version} (AUC-PR={metrics.get('auc_pr', 0):.4f})"


def _latest_step(session, database: str, schema: str, step: str, status=None):
    from snowflake.snowpark import functions as F

    df = session.table(_fqn(database, schema, "PIPELINE_RUN_LOG")).filter(
        F.col("STEP_NAME") == F.lit(step)
    )
    if status:
        df = df.filter(F.col("STATUS") == F.lit(status))
    rows = df.sort(F.col("RUN_TIMESTAMP").desc()).limit(1).collect()
    return rows[0].as_dict() if rows else None


def validate_challenger(session, database: str, schema: str) -> str:
    ensure_tables(session, database, schema)
    train = _latest_step(session, database, schema, "TRAIN", "SUCCESS")
    if not train:
        return "No challenger"
    champion = get_champion(session, database, schema) or {}

    challenger_metrics = {
        "auc_pr": _float(train.get("CHALLENGER_AUC_PR")),
        "recall_at_1pct_fpr": _float(train.get("CHALLENGER_RECALL_AT_1PCT_FPR")),
    }
    if train.get("COMPARISON_BASIS") == "same_holdout":
        basis = "same_holdout"
        champion_metrics = {
            "auc_pr": _float(train.get("CHAMPION_AUC_PR")),
            "recall_at_1pct_fpr": _float(train.get("CHAMPION_RECALL_AT_1PCT_FPR")),
            "auc_roc": _float(train.get("CHAMPION_AUC")),
        }
    else:
        basis = "champion_tracker"
        champion_metrics = {
            "auc_pr": _float(champion.get("CHAMPION_AUC_PR")),
            "recall_at_1pct_fpr": _float(champion.get("CHAMPION_RECALL_AT_1PCT_FPR")),
            "auc_roc": _float(champion.get("CHAMPION_AUC")),
        }

    decision = decide_promotion(challenger_metrics, champion_metrics)
    log_step(
        session,
        database,
        schema,
        RUN_ID=train.get("RUN_ID"),
        STEP_NAME="VALIDATE",
        STATUS="PROMOTE" if decision["promote"] else "KEEP_CHAMPION",
        CHALLENGER_VERSION=train.get("CHALLENGER_VERSION"),
        CHAMPION_VERSION=champion.get("CHAMPION_VERSION"),
        CHALLENGER_AUC=_float(train.get("CHALLENGER_AUC")),
        CHAMPION_AUC=champion_metrics["auc_roc"],
        CHALLENGER_AUC_PR=challenger_metrics["auc_pr"],
        CHAMPION_AUC_PR=champion_metrics["auc_pr"],
        CHALLENGER_RECALL_AT_1PCT_FPR=challenger_metrics["recall_at_1pct_fpr"],
        CHAMPION_RECALL_AT_1PCT_FPR=champion_metrics["recall_at_1pct_fpr"],
        CHALLENGER_DECLINE_THRESHOLD=_float(train.get("CHALLENGER_DECLINE_THRESHOLD")),
        CHALLENGER_REVIEW_THRESHOLD=_float(train.get("CHALLENGER_REVIEW_THRESHOLD")),
        COMPARISON_BASIS=basis,
        PROMOTED=decision["promote"],
        MESSAGE=decision["message"][:4000],
    )
    return decision["message"]


def promote_if_better(session, database: str, schema: str) -> str:
    from snowflake.snowpark import functions as F

    ensure_tables(session, database, schema)
    last = _latest_step(session, database, schema, "VALIDATE")
    if not last:
        return "No validation"
    if not last.get("PROMOTED"):
        log_step(
            session,
            database,
            schema,
            RUN_ID=last.get("RUN_ID"),
            STEP_NAME="PROMOTE",
            STATUS="SKIPPED",
            CHALLENGER_VERSION=last.get("CHALLENGER_VERSION"),
            PROMOTED=False,
            MESSAGE="Champion retained",
        )
        return "Champion retained"

    tracker = session.table(_fqn(database, schema, "MODEL_CHAMPION_TRACKER"))
    new_values = {
        "CHAMPION_VERSION": F.lit(last["CHALLENGER_VERSION"]),
        "CHAMPION_AUC": F.lit(_float(last.get("CHALLENGER_AUC"))),
        "CHAMPION_AUC_PR": F.lit(_float(last.get("CHALLENGER_AUC_PR"))),
        "CHAMPION_RECALL_AT_1PCT_FPR": F.lit(
            _float(last.get("CHALLENGER_RECALL_AT_1PCT_FPR"))
        ),
        "CHAMPION_REVIEW_THRESHOLD": F.lit(
            _float(last.get("CHALLENGER_REVIEW_THRESHOLD"))
        ),
        "CHAMPION_DECLINE_THRESHOLD": F.lit(
            _float(last.get("CHALLENGER_DECLINE_THRESHOLD"))
        ),
        "UPDATED_AT": F.current_timestamp(),
    }
    result = tracker.update(new_values, tracker["MODEL_NAME"] == F.lit(MODEL_NAME))
    if result.rows_updated == 0:
        append_rows(
            session,
            _fqn(database, schema, "MODEL_CHAMPION_TRACKER"),
            CHAMPION_TRACKER_COLUMNS,
            [
                {
                    "MODEL_NAME": MODEL_NAME,
                    "CHAMPION_VERSION": last["CHALLENGER_VERSION"],
                    "CHAMPION_AUC": _float(last.get("CHALLENGER_AUC")),
                    "CHAMPION_AUC_PR": _float(last.get("CHALLENGER_AUC_PR")),
                    "CHAMPION_RECALL_AT_1PCT_FPR": _float(
                        last.get("CHALLENGER_RECALL_AT_1PCT_FPR")
                    ),
                    "CHAMPION_REVIEW_THRESHOLD": _float(
                        last.get("CHALLENGER_REVIEW_THRESHOLD")
                    ),
                    "CHAMPION_DECLINE_THRESHOLD": _float(
                        last.get("CHALLENGER_DECLINE_THRESHOLD")
                    ),
                    "UPDATED_AT": datetime.utcnow(),
                }
            ],
        )
    log_step(
        session,
        database,
        schema,
        RUN_ID=last.get("RUN_ID"),
        STEP_NAME="PROMOTE",
        STATUS="SUCCESS",
        CHALLENGER_VERSION=last["CHALLENGER_VERSION"],
        CHALLENGER_AUC_PR=_float(last.get("CHALLENGER_AUC_PR")),
        CHALLENGER_RECALL_AT_1PCT_FPR=_float(last.get("CHALLENGER_RECALL_AT_1PCT_FPR")),
        PROMOTED=True,
        MESSAGE=f"New champion: {last['CHALLENGER_VERSION']}",
    )
    return f"PROMOTED: {last['CHALLENGER_VERSION']}"


def _metric_value(row: Dict) -> Optional[float]:
    for key, value in row.items():
        if key.upper() == "METRIC_VALUE" and value is not None:
            return float(value)
    return None


def check_model_drift(
    session,
    database: str,
    schema: str,
    monitor_name: str,
    psi_threshold: float,
    lookback_days: int,
) -> str:
    """Query Model Monitor PSI per feature; log it and trigger retraining on drift."""
    from snowflake.snowpark import functions as F

    from utils import FEATURE_COLS

    ensure_tables(session, database, schema)
    check_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    psi = {}
    for feature in FEATURE_COLS:
        try:
            rows = session.sql(
                "SELECT * FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(?, 'POPULATION_STABILITY_INDEX', ?, "
                "'1 DAY', DATEADD('DAY', ?, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, "
                "CURRENT_TIMESTAMP()::TIMESTAMP_NTZ))",
                params=[monitor_name, feature, -int(lookback_days)],
            ).collect()
        except Exception:  # noqa: BLE001 - e.g. no data yet for this feature
            continue
        values = [
            v for v in (_metric_value(r.as_dict()) for r in rows) if v is not None
        ]
        if values:
            psi[feature] = max(values)

    drifted = drifted_features(psi, psi_threshold)
    now = datetime.utcnow()
    append_rows(
        session,
        _fqn(database, schema, "MODEL_DRIFT_LOG"),
        DRIFT_LOG_COLUMNS,
        [
            {
                "CHECK_ID": check_id,
                "CHECK_TIMESTAMP": now,
                "MONITOR_NAME": monitor_name,
                "FEATURE_NAME": feature,
                "METRIC_NAME": "POPULATION_STABILITY_INDEX",
                "METRIC_VALUE": value,
                "THRESHOLD": float(psi_threshold),
                "IS_DRIFTED": feature in drifted,
            }
            for feature, value in psi.items()
        ],
    )

    if not psi:
        log_step(
            session,
            database,
            schema,
            RUN_ID=check_id,
            STEP_NAME="DRIFT_CHECK",
            STATUS="NO_DATA",
            MESSAGE=f"No drift metrics from {monitor_name}",
        )
        return "NO_DATA"
    if not drifted:
        log_step(
            session,
            database,
            schema,
            RUN_ID=check_id,
            STEP_NAME="DRIFT_CHECK",
            STATUS="OK",
            MESSAGE=f"max PSI={max(psi.values()):.3f} over {len(psi)} features",
        )
        return "OK"

    summary = ", ".join(
        f"{f}={v:.3f}" for f, v in sorted(drifted.items(), key=lambda x: -x[1])
    )
    recent_train = (
        session.table(_fqn(database, schema, "PIPELINE_RUN_LOG"))
        .filter(
            (F.col("STEP_NAME") == F.lit("TRAIN"))
            & (
                F.col("RUN_TIMESTAMP")
                >= F.dateadd("hour", F.lit(-DRIFT_RETRAIN_COOLDOWN_HOURS), F.sysdate())
            )
        )
        .count()
    )
    if recent_train:
        status, action = "DRIFT_COOLDOWN", "retraining already ran within cooldown"
    else:
        session.sql(
            f"EXECUTE TASK {_fqn(database, schema, 'TASK_TRAIN_CHALLENGER')}"
        ).collect()
        status, action = "DRIFT_RETRAIN_TRIGGERED", "triggered TASK_TRAIN_CHALLENGER"
    log_step(
        session,
        database,
        schema,
        RUN_ID=check_id,
        STEP_NAME="DRIFT_CHECK",
        STATUS=status,
        MESSAGE=f"PSI>{psi_threshold}: {summary}; {action}"[:4000],
    )
    return f"{status}: {summary}"
