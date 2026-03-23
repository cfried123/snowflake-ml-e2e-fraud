"""Shared utilities for fraud detection simulator."""

from datetime import datetime
from typing import Dict, List

import requests
import snowflake.connector
from snowflake.snowpark import Session
from snowflake.ml.feature_store import FeatureStore

from config import DATABASE, SCHEMA, FEATURE_STORE_SCHEMA, WAREHOUSE, SERVICE_NAME


def get_session(connection_name: str) -> Session:
    conn = snowflake.connector.connect(connection_name=connection_name)
    session = Session.builder.configs({"connection": conn}).create()
    session.sql("USE ROLE ACCOUNTADMIN").collect()
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"USE WAREHOUSE {WAREHOUSE}").collect()
    return session


def get_feature_store(session: Session) -> FeatureStore:
    return FeatureStore(session, DATABASE, FEATURE_STORE_SCHEMA, WAREHOUSE)


def get_endpoint_url(session: Session) -> str:
    endpoints = session.sql(
        f"SHOW ENDPOINTS IN SERVICE {DATABASE}.{SCHEMA}.{SERVICE_NAME}"
    ).collect()
    for row in endpoints:
        row_dict = row.asDict()
        if row_dict.get("is_public") and row_dict.get("ingress_url"):
            return f"https://{row_dict['ingress_url']}"
    raise RuntimeError(f"No public endpoint found for service {SERVICE_NAME}")


def build_feature_vector(
    features: Dict, amount: float, hour: int, card_present: int, is_intl: int
) -> List:
    avg_txn = features.get("AVG_TXN_AMOUNT") or 100
    max_txn = features.get("MAX_TXN_AMOUNT") or 500
    return [
        int(features.get("ACCOUNT_AGE_DAYS") or 365),
        int(features.get("CREDIT_SCORE") or 700),
        int(features.get("AVG_MONTHLY_TRANSACTIONS") or 20),
        float(features.get("AVG_TRANSACTION_AMOUNT") or 100),
        int(features.get("NUM_CARDS_ON_FILE") or 1),
        int(features.get("IS_PREMIUM_MEMBER") or 0),
        int(features.get("NUM_PREVIOUS_FRAUD_REPORTS") or 0),
        float(avg_txn),
        float(features.get("STD_TXN_AMOUNT") or 50),
        float(max_txn),
        float(features.get("MIN_TXN_AMOUNT") or 10),
        float(features.get("TOTAL_TXN_AMOUNT") or 2000),
        int(features.get("TXN_COUNT") or 20),
        float(features.get("AVG_TXN_HOUR") or 12),
        float(features.get("STD_TXN_HOUR") or 4),
        int(features.get("INTL_TXN_COUNT") or 0),
        float(features.get("INTL_TXN_RATIO") or 0),
        float(features.get("CARD_PRESENT_RATIO") or 0.7),
        float(amount),
        int(hour),
        int(datetime.now().weekday()),
        int(datetime.now().weekday() >= 5),
        int(card_present),
        int(is_intl),
        float(amount / (avg_txn + 1)),
        float(amount / (max_txn + 1)),
        int(amount > avg_txn * 3),
        int(hour in [0, 1, 2, 3, 4, 5]),
        float(is_intl * 2 + (1 - card_present) * 1.5),
    ]


def call_model(
    endpoint_url: str, pat_token: str, txn_id: int, feature_vector: List
) -> Dict:
    resp = requests.post(
        f"{endpoint_url}/predict-proba",
        json={"data": [[txn_id] + feature_vector]},
        headers={
            "Authorization": f'Snowflake Token="{pat_token}"',
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    prob = 0.0
    if resp.status_code == 200:
        data = resp.json().get("data", [[]])[0]
        if len(data) >= 2:
            pred = data[1]
            prob = (
                float(pred.get("output_feature_1", 0))
                if isinstance(pred, dict)
                else float(pred)
            )
    return {
        "prob": prob,
        "status_code": resp.status_code,
        "success": resp.status_code == 200,
    }


def get_decision(prob: float) -> str:
    if prob >= 0.7:
        return "DECLINE"
    elif prob >= 0.3:
        return "REVIEW"
    return "APPROVE"
