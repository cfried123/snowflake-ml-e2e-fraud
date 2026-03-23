#!/usr/bin/env python3
"""Single transaction prediction with latency breakdown.

Uses raw SQL against online feature tables for low-latency feature lookups.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import requests
from snowflake.snowpark import Session

from config import (
    DATABASE,
    FEATURE_SERVING_WH,
    FEATURE_STORE_SCHEMA,
    MODEL_NAME,
    SCHEMA,
)

ONLINE_TABLES = {
    "CUSTOMER_PROFILE_FEATURES": (
        f'{DATABASE}.{FEATURE_STORE_SCHEMA}."CUSTOMER_PROFILE_FEATURES$V1$ONLINE"',
        [
            "ACCOUNT_AGE_DAYS",
            "CREDIT_SCORE",
            "AVG_MONTHLY_TRANSACTIONS",
            "AVG_TRANSACTION_AMOUNT",
            "NUM_CARDS_ON_FILE",
            "IS_PREMIUM_MEMBER",
            "NUM_PREVIOUS_FRAUD_REPORTS",
        ],
    ),
    "CUSTOMER_AGG_FEATURES": (
        f'{DATABASE}.{FEATURE_STORE_SCHEMA}."CUSTOMER_AGG_FEATURES$V1$ONLINE"',
        [
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
        ],
    ),
}


def _fetch_one_table(
    session: Session,
    name: str,
    table_fqn: str,
    columns: list[str],
    customer_id: str,
) -> tuple[str, dict, float]:
    """Fetch features from one online table (designed to run in a thread)."""
    cols = ", ".join(f'"{c}"' for c in columns)
    t0 = time.time()
    rows = session.sql(
        f"SELECT {cols} FROM {table_fqn} WHERE CUSTOMER_ID = '{customer_id}'"
    ).collect()
    elapsed_ms = (time.time() - t0) * 1000
    row_dict = rows[0].as_dict() if rows else {}
    return name, row_dict, elapsed_ms


def get_features_from_store(
    session: Session, customer_id: str
) -> tuple[dict, dict[str, float]]:
    """Fetch features from online tables in parallel via raw SQL."""
    features = {}
    latencies = {}

    with ThreadPoolExecutor(max_workers=len(ONLINE_TABLES)) as pool:
        futures = {
            pool.submit(_fetch_one_table, session, name, tbl, cols, customer_id): name
            for name, (tbl, cols) in ONLINE_TABLES.items()
        }
        for fut in as_completed(futures):
            name, row_dict, elapsed_ms = fut.result()
            latencies[name] = elapsed_ms
            features.update(row_dict)

    return features, latencies


def build_feature_vector(
    f: dict, amount: float, hour: int, card_present: int, is_intl: int
) -> list:
    """Build the 27-feature vector for model inference."""
    avg_txn = f.get("AVG_TXN_AMOUNT") or 100
    max_txn = f.get("MAX_TXN_AMOUNT") or 500
    return [
        int(f.get("ACCOUNT_AGE_DAYS") or 365),
        int(f.get("CREDIT_SCORE") or 700),
        int(f.get("AVG_MONTHLY_TRANSACTIONS") or 20),
        float(f.get("AVG_TRANSACTION_AMOUNT") or 100),
        int(f.get("NUM_CARDS_ON_FILE") or 1),
        int(f.get("IS_PREMIUM_MEMBER") or 0),
        int(f.get("NUM_PREVIOUS_FRAUD_REPORTS") or 0),
        float(avg_txn),
        float(f.get("STD_TXN_AMOUNT") or 50),
        float(max_txn),
        float(f.get("MIN_TXN_AMOUNT") or 10),
        float(f.get("TOTAL_TXN_AMOUNT") or 2000),
        int(f.get("TXN_COUNT") or 20),
        float(f.get("AVG_TXN_HOUR") or 12),
        float(f.get("STD_TXN_HOUR") or 4),
        int(f.get("INTL_TXN_COUNT") or 0),
        float(f.get("INTL_TXN_RATIO") or 0),
        float(f.get("CARD_PRESENT_RATIO") or 0.7),
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


def main():
    parser = argparse.ArgumentParser(
        description="Predict fraud for a single transaction"
    )
    parser.add_argument(
        "--customer", "-c", type=str, help="Customer ID (default: random)"
    )
    parser.add_argument(
        "--connection", type=str, default="default", help="Snowflake connection"
    )
    args = parser.parse_args()

    pat_token = os.environ.get("SNOWFLAKE_PAT")
    if not pat_token:
        print("ERROR: Set SNOWFLAKE_PAT environment variable")
        return

    print("Connecting to Snowflake...")
    conn_start = time.time()
    session = Session.builder.config("connection_name", args.connection).getOrCreate()
    session.sql("USE ROLE ACCOUNTADMIN").collect()
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"USE WAREHOUSE {FEATURE_SERVING_WH}").collect()
    conn_ms = (time.time() - conn_start) * 1000
    print(f"  Connection: {conn_ms:.0f}ms")

    if args.customer:
        customer_id = args.customer
    else:
        rows = session.sql(
            f"SELECT DISTINCT CUSTOMER_ID FROM {DATABASE}.{SCHEMA}.TRANSACTIONS LIMIT 100"
        ).collect()
        customers = [row[0] for row in rows]
        customer_id = str(np.random.choice(customers))

    is_suspicious = np.random.random() < 0.3
    if is_suspicious:
        amount = float(np.random.uniform(1500, 5000))
        hour = int(np.random.choice([0, 1, 2, 3, 4, 23]))
        is_intl = int(np.random.random() < 0.7)
        card_present = int(np.random.random() < 0.3)
    else:
        amount = float(np.random.lognormal(4, 1.2))
        hour = int(np.random.randint(8, 22))
        is_intl = int(np.random.random() < 0.08)
        card_present = int(np.random.random() > 0.2)

    print(f"\n{'=' * 60}")
    print("TRANSACTION")
    print(f"  Customer:      {customer_id}")
    print(f"  Amount:        ${amount:,.2f}")
    print(f"  Hour:          {hour}:00")
    print(f"  International: {'Yes' if is_intl else 'No'}")
    print(f"  Card Present:  {'Yes' if card_present else 'No'}")

    print(f"\n{'=' * 60}")
    print("FEATURE LOOKUP (Raw SQL → Online Feature Tables)")
    fs_start = time.time()
    f, fv_latencies = get_features_from_store(session, customer_id)
    fs_ms = (time.time() - fs_start) * 1000
    print(f"  Total Latency: {fs_ms:.0f}ms")
    for fv_name, fv_ms in fv_latencies.items():
        print(f"    {fv_name}: {fv_ms:.0f}ms")

    if not f:
        print(f"  ERROR: No features found for customer {customer_id}")
        session.close()
        return

    print("\n  Profile Features:")
    print(f"    Account Age:     {f.get('ACCOUNT_AGE_DAYS', 'N/A')} days")
    print(f"    Credit Score:    {f.get('CREDIT_SCORE', 'N/A')}")
    print(f"    Premium Member:  {'Yes' if f.get('IS_PREMIUM_MEMBER') else 'No'}")
    print(f"    Fraud Reports:   {f.get('NUM_PREVIOUS_FRAUD_REPORTS', 0)}")

    print("\n  Aggregate Features:")
    avg_txn = f.get("AVG_TXN_AMOUNT") or 100
    max_txn = f.get("MAX_TXN_AMOUNT") or 500
    print(f"    Avg Amount:      ${avg_txn:,.2f}")
    print(f"    Max Amount:      ${max_txn:,.2f}")
    print(f"    Txn Count:       {f.get('TXN_COUNT', 'N/A')}")
    print(f"    Intl Ratio:      {(f.get('INTL_TXN_RATIO') or 0) * 100:.1f}%")
    print(f"    Card Present:    {(f.get('CARD_PRESENT_RATIO') or 0) * 100:.1f}%")

    feature_vector = build_feature_vector(f, amount, hour, card_present, is_intl)

    print(f"\n{'=' * 60}")
    print("RESOLVING MODEL SERVICE")
    svc_rows = session.sql(f"SHOW SERVICES IN SCHEMA {DATABASE}.{SCHEMA}").collect()
    service_name = None
    for row in svc_rows:
        rd = row.as_dict()
        if (
            rd.get("managing_object_domain") == "Model"
            and MODEL_NAME in (rd.get("managing_object_name") or "")
            and str(rd.get("is_job")).lower() == "false"
            and rd.get("status") not in ("DONE", "FAILED")
        ):
            service_name = rd["name"]
            break

    if not service_name:
        print("ERROR: No inference service found for model")
        session.close()
        return

    print(f"  Service: {service_name}")

    if rd.get("status") == "SUSPENDED":
        print("  Service is suspended, resuming...")
        session.sql(
            f"ALTER SERVICE {DATABASE}.{SCHEMA}.{service_name} RESUME"
        ).collect()
        print("  Waiting for service to start...")
        time.sleep(5)

    endpoint_rows = session.sql(
        f"SHOW ENDPOINTS IN SERVICE {DATABASE}.{SCHEMA}.{service_name}"
    ).collect()
    endpoint_url = None
    for row in endpoint_rows:
        row_dict = row.as_dict()
        if row_dict.get("is_public") and row_dict.get("ingress_url"):
            endpoint_url = f"https://{row_dict['ingress_url']}"
            break

    if not endpoint_url:
        print("ERROR: No public endpoint found")
        session.close()
        return

    print(f"\n{'=' * 60}")
    print("MODEL INFERENCE")
    inf_start = time.time()
    resp = requests.post(
        f"{endpoint_url}/predict-proba",
        json={"data": [[0] + feature_vector]},
        headers={
            "Authorization": f'Snowflake Token="{pat_token}"',
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    inf_ms = (time.time() - inf_start) * 1000
    print(f"  Latency: {inf_ms:.0f}ms")

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
    else:
        print(f"  ERROR: {resp.status_code} - {resp.text}")

    decision = "DECLINE" if prob >= 0.7 else "REVIEW" if prob >= 0.3 else "APPROVE"
    decision_color = {
        "APPROVE": "\033[92m",
        "REVIEW": "\033[93m",
        "DECLINE": "\033[91m",
    }

    print(f"\n{'=' * 60}")
    print("RESULT")
    print(f"  Fraud Probability: {prob:.1%}")
    print(f"  Decision:          {decision_color[decision]}{decision}\033[0m")

    print("\n  Risk Factors:")
    amount_ratio = amount / (avg_txn + 1)
    if amount_ratio > 3:
        print(f"    [!] Amount is {amount_ratio:.1f}x average")
    if hour in [0, 1, 2, 3, 4, 5]:
        print(f"    [!] Unusual hour ({hour}:00)")
    if is_intl and (f.get("INTL_TXN_RATIO") or 0) < 0.1:
        print("    [!] International (customer rarely does intl)")
    if not card_present and (f.get("CARD_PRESENT_RATIO") or 0) > 0.8:
        print("    [!] Card not present (customer usually uses card)")
    if (
        amount_ratio <= 3
        and hour not in [0, 1, 2, 3, 4, 5]
        and not is_intl
        and card_present
    ):
        print("    None detected")

    print(f"\n{'=' * 60}")
    print("LATENCY SUMMARY")
    total_ms = fs_ms + inf_ms
    print(f"  Feature Lookup: {fs_ms:>6.0f}ms  ({fs_ms / total_ms * 100:>5.1f}%)")
    for fv_name, fv_ms in fv_latencies.items():
        print(f"    └ {fv_name}: {fv_ms:>6.0f}ms")
    print(f"  Model Inference:{inf_ms:>6.0f}ms  ({inf_ms / total_ms * 100:>5.1f}%)")
    print(f"  Total:          {total_ms:>6.0f}ms")

    session.close()


if __name__ == "__main__":
    main()
