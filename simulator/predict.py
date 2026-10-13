#!/usr/bin/env python3
"""Single transaction prediction with latency breakdown.

Uses raw SQL against online feature tables for low-latency feature lookups.
Feature-vector building, model calls and decision bands are shared with
``load_test.py`` via ``inference.py``.
"""

import argparse
import os
import time

import numpy as np

from config import DATABASE, SCHEMA
from inference import (
    AGG_FV_NAME,
    PROFILE_FV_NAME,
    build_feature_vector,
    call_model,
    fetch_online_features,
    get_decision,
    get_endpoint_url,
    get_session,
    load_decision_thresholds,
    resolve_service,
)
from source.utils import agg_col, is_odd_hour


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
    session = get_session(args.connection)
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
    online, fv_latencies = fetch_online_features(session, customer_id)
    fs_ms = (time.time() - fs_start) * 1000
    print(f"  Total Latency: {fs_ms:.0f}ms")
    for fv_name, fv_ms in fv_latencies.items():
        print(f"    {fv_name}: {fv_ms:.0f}ms")

    profile = online.get(PROFILE_FV_NAME, {})
    agg = online.get(AGG_FV_NAME, {})
    if not profile:
        print(f"  ERROR: No profile features found for customer {customer_id}")
        session.close()
        return

    print("\n  Profile Features:")
    print(f"    Account Age:     {profile.get('ACCOUNT_AGE_DAYS', 'N/A')} days")
    print(f"    Credit Score:    {profile.get('CREDIT_SCORE', 'N/A')}")
    print(f"    Premium Member:  {'Yes' if profile.get('IS_PREMIUM_MEMBER') else 'No'}")
    print(f"    Fraud Reports:   {profile.get('NUM_PREVIOUS_FRAUD_REPORTS', 0)}")

    print("\n  Aggregate Features (current window):")
    txn_count = agg.get(agg_col("TXN_COUNT")) or 0
    avg_txn = agg.get(agg_col("AVG_TXN_AMOUNT"))
    if not txn_count:
        print("    No recent history (HAS_HISTORY=0)")
    else:
        print(f"    Avg Amount:      ${avg_txn:,.2f}")
        print(f"    Max Amount:      ${agg.get(agg_col('MAX_TXN_AMOUNT')):,.2f}")
        print(f"    Txn Count:       {txn_count}")
        print(
            f"    Intl Ratio:      {(agg.get(agg_col('INTL_TXN_RATIO')) or 0) * 100:.1f}%"
        )
        print(
            f"    Card Present:    {(agg.get(agg_col('CARD_PRESENT_RATIO')) or 0) * 100:.1f}%"
        )

    feature_vector = build_feature_vector(online, amount, hour, card_present, is_intl)

    print(f"\n{'=' * 60}")
    print("RESOLVING MODEL SERVICE")
    service_name = resolve_service(session)
    if not service_name:
        print("ERROR: No inference service found for model")
        session.close()
        return
    print(f"  Service: {service_name}")

    endpoint_url = get_endpoint_url(session, service_name)
    if not endpoint_url:
        print("ERROR: No public endpoint found")
        session.close()
        return

    review_th, decline_th, th_source = load_decision_thresholds(session, service_name)
    print(
        f"  Decision bands ({th_source}): REVIEW >= {review_th:.3f}, DECLINE >= {decline_th:.3f}"
    )

    print(f"\n{'=' * 60}")
    print("MODEL INFERENCE")
    inf_start = time.time()
    result = call_model(endpoint_url, pat_token, feature_vector)
    inf_ms = (time.time() - inf_start) * 1000
    print(f"  Latency: {inf_ms:.0f}ms")
    if not result["success"]:
        print(f"  ERROR: {result['status_code']} - {result['error']}")
    prob = result["prob"]

    decision = get_decision(prob, review_th, decline_th)
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
    factors = []
    if avg_txn and amount / (avg_txn + 1) > 3:
        factors.append(f"Amount is {amount / (avg_txn + 1):.1f}x recent average")
    if is_odd_hour(hour):
        factors.append(f"Unusual hour ({hour}:00)")
    if is_intl and (agg.get(agg_col("INTL_TXN_RATIO")) or 0) < 0.1:
        factors.append("International (customer rarely does intl)")
    if not card_present and (agg.get(agg_col("CARD_PRESENT_RATIO")) or 0) > 0.8:
        factors.append("Card not present (customer usually uses card)")
    if not txn_count:
        factors.append("No recent customer history")
    for factor in factors or ["None detected"]:
        print(f"    [!] {factor}" if factors else f"    {factor}")

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
