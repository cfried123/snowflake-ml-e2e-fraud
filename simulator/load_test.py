#!/usr/bin/env python3
"""
Load Test - Fraud Detection Pipeline

Runs concurrent transactions through the full pipeline:
  Raw SQL feature lookup → Feature vector build → Model inference

Usage:
    export SNOWFLAKE_PAT=<token>
    python simulator/load_test.py -n 100 -c 10 --connection <connection-name>
"""

import argparse
import os
import sys
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


def _fetch_one_table(session, name, table_fqn, columns, customer_id):
    cols = ", ".join(f'"{c}"' for c in columns)
    rows = session.sql(
        f"SELECT {cols} FROM {table_fqn} WHERE CUSTOMER_ID = '{customer_id}'"
    ).collect()
    return name, rows[0].as_dict() if rows else {}


def get_features(session, customer_id):
    features = {}
    with ThreadPoolExecutor(max_workers=len(ONLINE_TABLES)) as pool:
        futures = {
            pool.submit(_fetch_one_table, session, name, tbl, cols, customer_id): name
            for name, (tbl, cols) in ONLINE_TABLES.items()
        }
        for fut in as_completed(futures):
            _, row_dict = fut.result()
            features.update(row_dict)
    return features


def build_feature_vector(f, amount, hour, card_present, is_intl):
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


def resolve_service(session):
    svc_rows = session.sql(f"SHOW SERVICES IN SCHEMA {DATABASE}.{SCHEMA}").collect()
    for row in svc_rows:
        rd = row.as_dict()
        if (
            rd.get("managing_object_domain") == "Model"
            and MODEL_NAME in (rd.get("managing_object_name") or "")
            and str(rd.get("is_job")).lower() == "false"
            and rd.get("status") not in ("DONE", "FAILED")
        ):
            service_name = rd["name"]
            if rd.get("status") == "SUSPENDED":
                print("  Service is suspended, resuming...")
                session.sql(
                    f"ALTER SERVICE {DATABASE}.{SCHEMA}.{service_name} RESUME"
                ).collect()
                time.sleep(5)
            return service_name
    return None


def get_endpoint_url(session, service_name):
    rows = session.sql(
        f"SHOW ENDPOINTS IN SERVICE {DATABASE}.{SCHEMA}.{service_name}"
    ).collect()
    for row in rows:
        rd = row.as_dict()
        if rd.get("is_public") and rd.get("ingress_url"):
            return f"https://{rd['ingress_url']}"
    return None


def call_model(endpoint_url, pat_token, feature_vector):
    try:
        resp = requests.post(
            f"{endpoint_url}/predict-proba",
            json={"data": [[0] + feature_vector]},
            headers={
                "Authorization": f'Snowflake Token="{pat_token}"',
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [[]])[0]
            if len(data) >= 2:
                pred = data[1]
                prob = (
                    float(pred.get("output_feature_1", 0))
                    if isinstance(pred, dict)
                    else float(pred)
                )
                return prob, True
        return 0.0, False
    except Exception:
        return 0.0, False


def process_one(session, endpoint_url, pat_token, customer_id, txn_id):
    t0 = time.time()

    amount = float(np.random.lognormal(4, 1.5))
    hour = int(np.random.randint(0, 24))
    is_intl = int(np.random.random() < 0.15)
    card_present = int(np.random.random() > 0.3)

    fs_start = time.time()
    f = get_features(session, customer_id)
    fs_ms = (time.time() - fs_start) * 1000

    feature_vector = build_feature_vector(f, amount, hour, card_present, is_intl)

    inf_start = time.time()
    prob, success = call_model(endpoint_url, pat_token, feature_vector)
    inf_ms = (time.time() - inf_start) * 1000

    decision = "DECLINE" if prob >= 0.7 else "REVIEW" if prob >= 0.3 else "APPROVE"
    total_ms = (time.time() - t0) * 1000

    return {
        "txn_id": txn_id,
        "customer_id": customer_id,
        "amount": amount,
        "prob": prob,
        "decision": decision,
        "fs_ms": fs_ms,
        "inf_ms": inf_ms,
        "total_ms": total_ms,
        "success": success,
    }


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Load Test")
    parser.add_argument(
        "-n", type=int, default=100, help="Number of transactions (default: 100)"
    )
    parser.add_argument("-c", type=int, default=5, help="Concurrency (default: 5)")
    parser.add_argument("--connection", type=str, default="default")
    args = parser.parse_args()

    pat_token = os.getenv("SNOWFLAKE_PAT")
    if not pat_token:
        print("ERROR: Set SNOWFLAKE_PAT environment variable")
        sys.exit(1)

    print("Connecting to Snowflake...")
    session = Session.builder.config("connection_name", args.connection).getOrCreate()
    session.sql("USE ROLE ACCOUNTADMIN").collect()
    session.sql(f"USE DATABASE {DATABASE}").collect()
    session.sql(f"USE WAREHOUSE {FEATURE_SERVING_WH}").collect()

    print("Resolving model service...")
    service_name = resolve_service(session)
    if not service_name:
        print("ERROR: No inference service found for model")
        session.close()
        sys.exit(1)
    print(f"  Service: {service_name}")

    endpoint_url = get_endpoint_url(session, service_name)
    if not endpoint_url:
        print("ERROR: No public endpoint found")
        session.close()
        sys.exit(1)
    print(f"  Endpoint: {endpoint_url}")

    rows = session.sql(
        f"SELECT DISTINCT CUSTOMER_ID FROM {DATABASE}.{SCHEMA}.TRANSACTIONS LIMIT 500"
    ).collect()
    customer_ids = [r[0] for r in rows]

    print(f"\n{'=' * 80}")
    print(f"LOAD TEST: {args.n} transactions, concurrency={args.c}")
    print(f"{'=' * 80}")

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.c) as pool:
        futures = {
            pool.submit(
                process_one,
                session,
                endpoint_url,
                pat_token,
                str(np.random.choice(customer_ids)),
                i,
            ): i
            for i in range(args.n)
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            color = {"APPROVE": "\033[92m", "REVIEW": "\033[93m", "DECLINE": "\033[91m"}
            reset = "\033[0m"
            ok = "OK" if r["success"] else "FAIL"
            print(
                f"[{r['txn_id']:4d}] {r['customer_id']} | "
                f"${r['amount']:>8.2f} | "
                f"prob={r['prob']:.3f} {color[r['decision']]}{r['decision']:8s}{reset} | "
                f"FS={r['fs_ms']:>4.0f}ms Inf={r['inf_ms']:>4.0f}ms "
                f"Total={r['total_ms']:>5.0f}ms | {ok}"
            )

    elapsed = time.time() - start
    session.close()

    ok = [r for r in results if r["success"]]
    if not ok:
        print("\nNo successful transactions.")
        return

    fs_times = [r["fs_ms"] for r in ok]
    inf_times = [r["inf_ms"] for r in ok]
    total_times = [r["total_ms"] for r in ok]
    decisions = [r["decision"] for r in ok]

    print(f"\n{'=' * 80}")
    print(f"RESULTS: {len(ok)}/{args.n} successful | {args.n / elapsed:.1f} TPS")
    print("\n  Latency (ms)       Avg      p50      p95      p99")
    print(f"  {'─' * 52}")
    for label, times in [
        ("Feature Store", fs_times),
        ("Model Inference", inf_times),
        ("Total", total_times),
    ]:
        print(
            f"  {label:<16s}{np.mean(times):>6.0f}   "
            f"{np.percentile(times, 50):>6.0f}   "
            f"{np.percentile(times, 95):>6.0f}   "
            f"{np.percentile(times, 99):>6.0f}"
        )
    print(
        f"\n  Decisions: APPROVE={decisions.count('APPROVE')} "
        f"REVIEW={decisions.count('REVIEW')} "
        f"DECLINE={decisions.count('DECLINE')}"
    )


if __name__ == "__main__":
    main()
