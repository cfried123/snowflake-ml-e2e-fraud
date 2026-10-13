"""Shared real-time scoring helpers for ``predict.py`` and ``load_test.py``.

Feature names, order and derived-feature logic come from ``source/utils.py``
(the same code used to build training data), so the vector sent to the model
always matches the signature it was trained and registered with.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from snowflake.snowpark import Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    AGG_FV_NAME,
    AGG_FV_VERSION,
    DATABASE,
    DECLINE_THRESHOLD_OVERRIDE,
    DEFAULT_DECLINE_THRESHOLD,
    DEFAULT_REVIEW_THRESHOLD,
    FEATURE_SERVING_WH,
    FEATURE_STORE_SCHEMA,
    MODEL_NAME,
    PROFILE_FV_NAME,
    PROFILE_FV_VERSION,
    REVIEW_THRESHOLD_OVERRIDE,
    SCHEMA,
    USE_OFFLINE_FEATURES,
)
from source.utils import (  # noqa: E402,F401
    FEATURE_COLS,
    PROFILE_FEATURE_COLS,
    TIMESTAMP_COL,
    agg_col,
    build_serving_feature_row,
    get_decision,
    get_online_agg_columns,
    most_recent_time_at_hour,
    to_model_input_vector,
)

SERVICE_PREFIX = "FRAUD_DETECTION_SERVICE_"

_TABLE_SUFFIX = "" if USE_OFFLINE_FEATURES else "$ONLINE"
ONLINE_TABLES = {
    PROFILE_FV_NAME: (
        f'{DATABASE}.{FEATURE_STORE_SCHEMA}."{PROFILE_FV_NAME}${PROFILE_FV_VERSION}{_TABLE_SUFFIX}"',
        PROFILE_FEATURE_COLS,
    ),
    AGG_FV_NAME: (
        f'{DATABASE}.{FEATURE_STORE_SCHEMA}."{AGG_FV_NAME}${AGG_FV_VERSION}{_TABLE_SUFFIX}"',
        get_online_agg_columns(),
    ),
}


def get_session(connection_name: str, warehouse: str = FEATURE_SERVING_WH) -> Session:
    session = Session.builder.config("connection_name", connection_name).getOrCreate()
    session.sql("USE ROLE ACCOUNTADMIN").collect()
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"USE WAREHOUSE {warehouse}").collect()
    return session


def _fetch_one_table(
    session: Session, name: str, table_fqn: str, columns: List[str], customer_id: str
) -> Tuple[str, Dict, float]:
    cols = ", ".join(f'"{c}"' for c in columns)
    t0 = time.time()
    # Security: customer_id comes from CLI input; bind it instead of
    # interpolating it into SQL (OWASP A03 Injection).
    rows = session.sql(
        f"SELECT {cols} FROM {table_fqn} WHERE CUSTOMER_ID = ?", params=[customer_id]
    ).collect()
    elapsed_ms = (time.time() - t0) * 1000
    return name, rows[0].as_dict() if rows else {}, elapsed_ms


def fetch_online_features(
    session: Session, customer_id: str
) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """Fetch profile and aggregate features in parallel from online tables."""
    features: Dict[str, Dict] = {}
    latencies: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=len(ONLINE_TABLES)) as pool:
        futures = [
            pool.submit(_fetch_one_table, session, name, tbl, cols, customer_id)
            for name, (tbl, cols) in ONLINE_TABLES.items()
        ]
        for fut in as_completed(futures):
            name, row, elapsed_ms = fut.result()
            features[name] = row
            latencies[name] = elapsed_ms
    return features, latencies


def build_feature_vector(
    features: Dict[str, Dict],
    amount: float,
    hour: int,
    card_present: int,
    is_intl: int,
    now: Optional[datetime] = None,
) -> List:
    """Build the model input row (``FEATURE_COLS`` order) for one transaction."""
    now = now or datetime.now()
    ts = most_recent_time_at_hour(now, hour)
    transaction = {
        TIMESTAMP_COL: ts,
        "TRANSACTION_AMOUNT": float(amount),
        "TRANSACTION_HOUR": int(hour),
        "TRANSACTION_DAY_OF_WEEK": ts.weekday(),
        "IS_WEEKEND": int(ts.weekday() >= 5),
        "CARD_PRESENT": int(card_present),
        "IS_INTERNATIONAL": int(is_intl),
    }
    row = build_serving_feature_row(
        transaction,
        features.get(PROFILE_FV_NAME, {}),
        features.get(AGG_FV_NAME, {}),
    )
    gap_col = agg_col("MINUTES_SINCE_LAST_TXN")
    if row.get(gap_col) is not None and row[gap_col] < 0:
        # The simulated hour can precede the customer's latest stored txn.
        row[gap_col] = 0.0
    return to_model_input_vector(row)


def resolve_service(session: Session) -> Optional[str]:
    for row in session.sql(f"SHOW SERVICES IN SCHEMA {DATABASE}.{SCHEMA}").collect():
        rd = row.as_dict()
        if (
            rd.get("managing_object_domain") == "Model"
            and MODEL_NAME in (rd.get("managing_object_name") or "")
            and str(rd.get("is_job")).lower() == "false"
            and rd.get("status") not in ("DONE", "FAILED")
        ):
            if rd.get("status") == "SUSPENDED":
                print("  Service is suspended, resuming...")
                session.sql(
                    f"ALTER SERVICE {DATABASE}.{SCHEMA}.{rd['name']} RESUME"
                ).collect()
                time.sleep(5)
            return rd["name"]
    return None


def get_endpoint_url(session: Session, service_name: str) -> Optional[str]:
    rows = session.sql(
        f"SHOW ENDPOINTS IN SERVICE {DATABASE}.{SCHEMA}.{service_name}"
    ).collect()
    for row in rows:
        rd = row.as_dict()
        if rd.get("is_public") and rd.get("ingress_url"):
            return f"https://{rd['ingress_url']}"
    return None


def load_decision_thresholds(
    session: Session, service_name: Optional[str]
) -> Tuple[float, float, str]:
    """Return (review, decline, source) for the deployed model version."""
    if REVIEW_THRESHOLD_OVERRIDE and DECLINE_THRESHOLD_OVERRIDE:
        return (
            float(REVIEW_THRESHOLD_OVERRIDE),
            float(DECLINE_THRESHOLD_OVERRIDE),
            "env",
        )
    if service_name and service_name.upper().startswith(SERVICE_PREFIX):
        version = service_name[len(SERVICE_PREFIX) :]
        try:
            from snowflake.ml.registry import Registry

            reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
            metrics = reg.get_model(MODEL_NAME).version(version).show_metrics()
            review, decline = (
                metrics.get("review_threshold"),
                metrics.get("decline_threshold"),
            )
            if review is not None and decline is not None:
                return float(review), float(decline), f"registry:{version}"
        except Exception as e:  # noqa: BLE001 - fall back to defaults
            print(f"  Could not read thresholds for {version}: {type(e).__name__}")
    return DEFAULT_REVIEW_THRESHOLD, DEFAULT_DECLINE_THRESHOLD, "default"


def call_model(
    endpoint_url: str, pat_token: str, feature_vector: List, row_id: int = 0
) -> Dict:
    try:
        resp = requests.post(
            f"{endpoint_url}/predict-proba",
            json={"data": [[row_id] + feature_vector]},
            headers={
                "Authorization": f'Snowflake Token="{pat_token}"',
                "Content-Type": "application/json",
            },
            timeout=30,
        )
    except requests.RequestException as e:
        return {
            "prob": 0.0,
            "success": False,
            "status_code": None,
            "error": type(e).__name__,
        }
    prob, success = 0.0, False
    if resp.status_code == 200:
        data = resp.json().get("data", [[]])[0]
        if len(data) >= 2:
            pred = data[1]
            prob = (
                float(pred.get("output_feature_1", 0))
                if isinstance(pred, dict)
                else float(pred)
            )
            success = True
    return {
        "prob": prob,
        "success": success,
        "status_code": resp.status_code,
        "error": None if success else resp.text[:500],
    }
