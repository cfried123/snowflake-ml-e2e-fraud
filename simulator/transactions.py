#!/usr/bin/env python3
"""
Transaction Simulator

Streams synthetic transactions to the TRANSACTIONS table, which feeds
the Dynamic Tables in the Feature Store (refreshing every 1 minute).

Usage:
    python simulator/transactions.py -n 100 --delay 0.5 --connection <connection-name>
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import snowflake.connector

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.config import DATABASE, SCHEMA, TRANSACTIONS_TABLE, WAREHOUSE


def load_reference_data(cursor):
    cursor.execute(
        f"SELECT DISTINCT CUSTOMER_ID FROM {DATABASE}.{SCHEMA}.CUSTOMER_PROFILES LIMIT 500"
    )
    customer_ids = [row[0] for row in cursor.fetchall()]
    cursor.execute(
        f"SELECT DISTINCT MERCHANT_ID, MERCHANT_CATEGORY FROM {DATABASE}.{SCHEMA}.{TRANSACTIONS_TABLE} LIMIT 100"
    )
    merchants = cursor.fetchall()
    merchant_ids = list(set(row[0] for row in merchants))
    merchant_categories = list(set(row[1] for row in merchants))
    return customer_ids, merchant_ids, merchant_categories


def generate_transaction(customer_ids, merchant_ids, merchant_categories, fraud_rate):
    is_suspicious = np.random.random() < fraud_rate
    timestamp = datetime.now()

    if is_suspicious:
        amount = float(np.random.uniform(800, 5000))
        hour = int(np.random.choice([1, 2, 3, 4, 23]))
        is_international = int(np.random.choice([0, 1], p=[0.3, 0.7]))
        card_present = 0
    else:
        amount = float(np.abs(np.random.lognormal(3.5, 1.2)))
        hour = timestamp.hour
        is_international = int(np.random.choice([0, 1], p=[0.92, 0.08]))
        card_present = int(np.random.choice([0, 1], p=[0.25, 0.75]))

    return {
        "TRANSACTION_ID": f"TXN_{int(timestamp.timestamp() * 1000000):018d}",
        "CUSTOMER_ID": str(np.random.choice(customer_ids)),
        "MERCHANT_ID": str(np.random.choice(merchant_ids)),
        "MERCHANT_CATEGORY": str(np.random.choice(merchant_categories)),
        "TRANSACTION_TIMESTAMP": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "TRANSACTION_AMOUNT": round(amount, 2),
        "TRANSACTION_HOUR": hour,
        "TRANSACTION_DAY_OF_WEEK": timestamp.weekday(),
        "IS_WEEKEND": 1 if timestamp.weekday() >= 5 else 0,
        "CARD_PRESENT": card_present,
        "IS_INTERNATIONAL": is_international,
        "IS_FRAUD": 0,
        "_is_suspicious": is_suspicious,
    }


def insert_transaction(cursor, txn):
    sql = f"""
    INSERT INTO {DATABASE}.{SCHEMA}.{TRANSACTIONS_TABLE}
    (TRANSACTION_ID, CUSTOMER_ID, MERCHANT_ID, MERCHANT_CATEGORY, TRANSACTION_TIMESTAMP,
     TRANSACTION_AMOUNT, TRANSACTION_HOUR, TRANSACTION_DAY_OF_WEEK, IS_WEEKEND,
     CARD_PRESENT, IS_INTERNATIONAL, IS_FRAUD)
    VALUES (%(TRANSACTION_ID)s, %(CUSTOMER_ID)s, %(MERCHANT_ID)s, %(MERCHANT_CATEGORY)s,
            %(TRANSACTION_TIMESTAMP)s, %(TRANSACTION_AMOUNT)s, %(TRANSACTION_HOUR)s,
            %(TRANSACTION_DAY_OF_WEEK)s, %(IS_WEEKEND)s, %(CARD_PRESENT)s,
            %(IS_INTERNATIONAL)s, %(IS_FRAUD)s)
    """
    cursor.execute(sql, {k: v for k, v in txn.items() if not k.startswith("_")})


def main():
    parser = argparse.ArgumentParser(description="Transaction Simulator")
    parser.add_argument(
        "-n", "--num", type=int, default=50, help="Number of transactions (default: 50)"
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=0.2,
        help="Delay between txns (default: 0.2)",
    )
    parser.add_argument(
        "-f",
        "--fraud-rate",
        type=float,
        default=0.15,
        help="Fraud rate (default: 0.15)",
    )
    parser.add_argument("--connection", type=str, default="default")
    args = parser.parse_args()

    target_table = f"{DATABASE}.{SCHEMA}.{TRANSACTIONS_TABLE}"
    print("=" * 90)
    print("TRANSACTION SIMULATOR")
    print("=" * 90)
    print(f"  Target: {target_table}")
    print(f"  Count:  {args.num}")
    print(f"  Delay:  {args.delay}s")
    print(f"  Fraud:  {args.fraud_rate * 100:.1f}%")
    print("-" * 90)

    conn = snowflake.connector.connect(connection_name=args.connection)
    cursor = conn.cursor()
    cursor.execute("USE ROLE ACCOUNTADMIN")
    cursor.execute(f"USE DATABASE {DATABASE}")
    cursor.execute(f"USE SCHEMA {SCHEMA}")
    cursor.execute(f"USE WAREHOUSE {WAREHOUSE}")

    print("\nLoading reference data...")
    customer_ids, merchant_ids, merchant_categories = load_reference_data(cursor)
    print(f"  Customers: {len(customer_ids)}")
    print(f"  Merchants: {len(merchant_ids)}")

    print(f"\n{'=' * 90}")
    print("STREAMING TRANSACTIONS")
    print("-" * 90)

    suspicious_count = 0
    for i in range(args.num):
        txn = generate_transaction(
            customer_ids, merchant_ids, merchant_categories, args.fraud_rate
        )
        insert_transaction(cursor, txn)
        conn.commit()

        if txn["_is_suspicious"]:
            suspicious_count += 1

        status = "\033[93mSUSPICIOUS\033[0m" if txn["_is_suspicious"] else "normal"
        print(
            f"[{i + 1:4d}/{args.num}] → {target_table} | "
            f"{txn['TRANSACTION_ID'][-12:]} | {txn['CUSTOMER_ID']} | "
            f"${txn['TRANSACTION_AMOUNT']:>8.2f} | hr={txn['TRANSACTION_HOUR']:02d} | "
            f"intl={txn['IS_INTERNATIONAL']} | card={txn['CARD_PRESENT']} | {status}"
        )

        if args.delay > 0 and i < args.num - 1:
            time.sleep(args.delay)

    print(f"\n{'=' * 90}")
    print("SIMULATION COMPLETE")
    print(f"  Total transactions: {args.num}")
    print(
        f"  Suspicious patterns: {suspicious_count} "
        f"({100 * suspicious_count / args.num:.1f}%)"
    )
    print("  Feature Store Dynamic Tables will refresh within 1 minute.")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
