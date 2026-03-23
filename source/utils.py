import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List
import random


def generate_synthetic_fraud_data(
    n_transactions: int = 100000, fraud_ratio: float = 0.02, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic transaction data for fraud detection.

    Args:
        n_transactions: Number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions
        seed: Random seed for reproducibility

    Returns:
        DataFrame with transaction features and fraud labels
    """
    np.random.seed(seed)
    random.seed(seed)

    n_customers = n_transactions // 50
    n_merchants = n_transactions // 200

    customer_ids = [f"CUST_{i:06d}" for i in range(n_customers)]
    merchant_ids = [f"MERCH_{i:05d}" for i in range(n_merchants)]
    merchant_categories = [
        "grocery",
        "gas_station",
        "restaurant",
        "online_retail",
        "travel",
        "entertainment",
        "electronics",
        "pharmacy",
        "clothing",
        "utilities",
    ]

    base_date = datetime(2025, 1, 1)
    timestamps = [
        base_date
        + timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )
        for _ in range(n_transactions)
    ]

    data = {
        "TRANSACTION_ID": [f"TXN_{i:010d}" for i in range(n_transactions)],
        "CUSTOMER_ID": np.random.choice(customer_ids, n_transactions),
        "MERCHANT_ID": np.random.choice(merchant_ids, n_transactions),
        "MERCHANT_CATEGORY": np.random.choice(merchant_categories, n_transactions),
        "TRANSACTION_TIMESTAMP": timestamps,
        "TRANSACTION_AMOUNT": np.abs(np.random.lognormal(3.5, 1.2, n_transactions)),
        "TRANSACTION_HOUR": [t.hour for t in timestamps],
        "TRANSACTION_DAY_OF_WEEK": [t.weekday() for t in timestamps],
        "IS_WEEKEND": [1 if t.weekday() >= 5 else 0 for t in timestamps],
        "CARD_PRESENT": np.random.choice([0, 1], n_transactions, p=[0.3, 0.7]),
        "IS_INTERNATIONAL": np.random.choice([0, 1], n_transactions, p=[0.9, 0.1]),
    }

    df = pd.DataFrame(data)

    n_fraud = int(n_transactions * fraud_ratio)
    fraud_indices = np.random.choice(n_transactions, n_fraud, replace=False)
    df["IS_FRAUD"] = 0
    df.loc[fraud_indices, "IS_FRAUD"] = 1

    for idx in fraud_indices:
        fraud_pattern = np.random.choice(
            ["high_amount", "odd_hours", "rapid_fire", "international"]
        )

        if fraud_pattern == "high_amount":
            df.loc[idx, "TRANSACTION_AMOUNT"] = np.random.uniform(500, 5000)
        elif fraud_pattern == "odd_hours":
            df.loc[idx, "TRANSACTION_HOUR"] = np.random.choice([2, 3, 4, 5])
        elif fraud_pattern == "rapid_fire":
            df.loc[idx, "TRANSACTION_AMOUNT"] = np.random.uniform(50, 200)
        elif fraud_pattern == "international":
            df.loc[idx, "IS_INTERNATIONAL"] = 1
            df.loc[idx, "CARD_PRESENT"] = 0

    df["TRANSACTION_AMOUNT"] = df["TRANSACTION_AMOUNT"].round(2)

    return df


def generate_customer_profiles(customer_ids: List[str], seed: int = 42) -> pd.DataFrame:
    """Generate customer profile data for feature enrichment.

    Args:
        customer_ids: List of unique customer IDs
        seed: Random seed

    Returns:
        DataFrame with customer profile features
    """
    np.random.seed(seed)
    n_customers = len(customer_ids)

    account_ages = np.random.exponential(scale=365 * 2, size=n_customers)

    data = {
        "CUSTOMER_ID": customer_ids,
        "ACCOUNT_AGE_DAYS": account_ages.astype(int),
        "CREDIT_SCORE": np.clip(
            np.random.normal(700, 80, n_customers), 300, 850
        ).astype(int),
        "AVG_MONTHLY_TRANSACTIONS": np.random.poisson(20, n_customers),
        "AVG_TRANSACTION_AMOUNT": np.abs(
            np.random.lognormal(3, 0.8, n_customers)
        ).round(2),
        "NUM_CARDS_ON_FILE": np.random.choice(
            [1, 2, 3, 4], n_customers, p=[0.5, 0.3, 0.15, 0.05]
        ),
        "IS_PREMIUM_MEMBER": np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
        "NUM_PREVIOUS_FRAUD_REPORTS": np.random.choice(
            [0, 1, 2, 3], n_customers, p=[0.85, 0.10, 0.04, 0.01]
        ),
    }

    return pd.DataFrame(data)


def compute_aggregated_features(
    transactions_df: pd.DataFrame, window_days: int = 7
) -> pd.DataFrame:
    """Compute aggregated transaction features per customer.

    Args:
        transactions_df: Raw transaction DataFrame
        window_days: Lookback window in days

    Returns:
        DataFrame with aggregated features per customer
    """
    df = transactions_df.copy()
    df["TRANSACTION_TIMESTAMP"] = pd.to_datetime(df["TRANSACTION_TIMESTAMP"])

    agg_features = (
        df.groupby("CUSTOMER_ID")
        .agg(
            {
                "TRANSACTION_AMOUNT": ["mean", "std", "max", "min", "sum", "count"],
                "TRANSACTION_HOUR": ["mean", "std"],
                "IS_INTERNATIONAL": ["sum", "mean"],
                "CARD_PRESENT": ["mean"],
            }
        )
        .reset_index()
    )

    agg_features.columns = [
        "CUSTOMER_ID",
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
    ]

    agg_features["STD_TXN_AMOUNT"] = agg_features["STD_TXN_AMOUNT"].fillna(0)
    agg_features["STD_TXN_HOUR"] = agg_features["STD_TXN_HOUR"].fillna(0)

    return agg_features


def prepare_model_features(
    transactions_df: pd.DataFrame,
    customer_profiles_df: pd.DataFrame,
    agg_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare final feature set for model training.

    Args:
        transactions_df: Raw transaction data
        customer_profiles_df: Customer profile features
        agg_features_df: Aggregated transaction features

    Returns:
        DataFrame ready for model training
    """
    df = transactions_df.merge(customer_profiles_df, on="CUSTOMER_ID", how="left")
    df = df.merge(agg_features_df, on="CUSTOMER_ID", how="left")

    df["AMOUNT_TO_AVG_RATIO"] = df["TRANSACTION_AMOUNT"] / (df["AVG_TXN_AMOUNT"] + 1)
    df["AMOUNT_TO_MAX_RATIO"] = df["TRANSACTION_AMOUNT"] / (df["MAX_TXN_AMOUNT"] + 1)
    df["IS_HIGH_VALUE_TXN"] = (
        df["TRANSACTION_AMOUNT"] > df["AVG_TXN_AMOUNT"] * 3
    ).astype(int)
    df["IS_ODD_HOUR"] = (
        (df["TRANSACTION_HOUR"] >= 0) & (df["TRANSACTION_HOUR"] <= 5)
    ).astype(int)
    df["RISK_SCORE"] = (
        df["IS_INTERNATIONAL"] * 2
        + df["IS_HIGH_VALUE_TXN"] * 1.5
        + df["IS_ODD_HOUR"] * 1
        + (1 - df["CARD_PRESENT"]) * 1.5
        + df["NUM_PREVIOUS_FRAUD_REPORTS"] * 3
    )

    return df


def get_feature_columns() -> Tuple[List[str], str]:
    """Return the list of feature columns and label column for model training.

    Returns:
        Tuple of (feature_columns, label_column)
    """
    feature_cols = [
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
    label_col = "IS_FRAUD"
    return feature_cols, label_col


def get_entity_key_columns() -> List[str]:
    """Return entity key columns for Feature Store."""
    return ["CUSTOMER_ID", "TRANSACTION_ID"]


def calculate_fraud_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict:
    """Calculate fraud-specific metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "auc_pr": average_precision_score(y_true, y_proba),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "fraud_detection_rate": recall_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
    }

    return metrics


def simulate_realtime_transaction(
    customer_ids: List[str],
    merchant_ids: List[str],
    merchant_categories: List[str],
    fraud_probability: float = 0.1,
) -> dict:
    """Simulate a single real-time transaction for inference demo.

    Args:
        customer_ids: List of valid customer IDs
        merchant_ids: List of valid merchant IDs
        merchant_categories: List of valid merchant categories
        fraud_probability: Probability of generating a suspicious transaction

    Returns:
        Dictionary representing a single transaction
    """
    is_suspicious = np.random.random() < fraud_probability

    timestamp = datetime.now()

    if is_suspicious:
        amount = np.random.uniform(500, 3000)
        hour = np.random.choice([1, 2, 3, 4, 23])
        is_international = np.random.choice([0, 1], p=[0.3, 0.7])
        card_present = 0
    else:
        amount = np.abs(np.random.lognormal(3.5, 1.2))
        hour = timestamp.hour
        is_international = np.random.choice([0, 1], p=[0.92, 0.08])
        card_present = np.random.choice([0, 1], p=[0.3, 0.7])

    return {
        "TRANSACTION_ID": f"TXN_{int(timestamp.timestamp() * 1000):015d}",
        "CUSTOMER_ID": np.random.choice(customer_ids),
        "MERCHANT_ID": np.random.choice(merchant_ids),
        "MERCHANT_CATEGORY": np.random.choice(merchant_categories),
        "TRANSACTION_TIMESTAMP": timestamp,
        "TRANSACTION_AMOUNT": round(amount, 2),
        "TRANSACTION_HOUR": hour,
        "TRANSACTION_DAY_OF_WEEK": timestamp.weekday(),
        "IS_WEEKEND": 1 if timestamp.weekday() >= 5 else 0,
        "CARD_PRESENT": card_present,
        "IS_INTERNATIONAL": is_international,
    }
