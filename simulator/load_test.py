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

import numpy as np

from config import DATABASE, SCHEMA
from inference import (
    build_feature_vector,
    call_model,
    fetch_online_features,
    get_decision,
    get_endpoint_url,
    get_session,
    load_decision_thresholds,
    resolve_service,
)


def process_one(session, endpoint_url, pat_token, customer_id, txn_id, bands):
    t0 = time.time()

    amount = float(np.random.lognormal(4, 1.5))
    hour = int(np.random.randint(0, 24))
    is_intl = int(np.random.random() < 0.15)
    card_present = int(np.random.random() > 0.3)

    fs_start = time.time()
    f, _ = fetch_online_features(session, customer_id)
    fs_ms = (time.time() - fs_start) * 1000

    feature_vector = build_feature_vector(f, amount, hour, card_present, is_intl)

    inf_start = time.time()
    result = call_model(endpoint_url, pat_token, feature_vector, row_id=txn_id)
    prob, success = result["prob"], result["success"]
    inf_ms = (time.time() - inf_start) * 1000

    decision = get_decision(prob, *bands)
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
    session = get_session(args.connection)

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
    review_th, decline_th, th_source = load_decision_thresholds(session, service_name)
    bands = (review_th, decline_th)
    print(
        f"  Decision bands ({th_source}): REVIEW >= {review_th:.3f}, DECLINE >= {decline_th:.3f}"
    )

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
                bands,
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
