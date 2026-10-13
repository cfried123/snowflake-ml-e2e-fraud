"""Shared fraud-detection utilities: synthetic data, features, and evaluation.

This module is the single source of truth for the model's feature contract.
It is imported by ``source/train.py`` (ML Job), the notebooks, the pipeline
stored procedures and the simulator, so feature names, ordering and derived
feature logic stay identical between training and serving.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

AGG_WINDOW_DAYS = 7
LABEL_COL = "IS_FRAUD"
TIMESTAMP_COL = "TRANSACTION_TIMESTAMP"
ID_COL = "TRANSACTION_ID"
CUSTOMER_COL = "CUSTOMER_ID"

# Hand-built heuristic that partially mirrors the label-generation rules.
# Kept only as a baseline comparison column; never used as a model input.
BASELINE_SCORE_COL = "RISK_SCORE"

TRANSACTION_FEATURE_COLS = [
    "TRANSACTION_AMOUNT",
    "TRANSACTION_HOUR",
    "TRANSACTION_DAY_OF_WEEK",
    "IS_WEEKEND",
    "CARD_PRESENT",
    "IS_INTERNATIONAL",
]

PROFILE_FEATURE_COLS = [
    "ACCOUNT_AGE_DAYS",
    "CREDIT_SCORE",
    "AVG_MONTHLY_TRANSACTIONS",
    "AVG_TRANSACTION_AMOUNT",
    "NUM_CARDS_ON_FILE",
    "IS_PREMIUM_MEMBER",
    "NUM_PREVIOUS_FRAUD_REPORTS",
]

# Aggregates that are legitimately 0 when a customer has no prior history.
_AGG_ZERO_WHEN_EMPTY = ["TOTAL_TXN_AMOUNT", "TXN_COUNT", "INTL_TXN_COUNT"]
# Aggregates that are undefined (NaN) when there is no prior history.
_AGG_NAN_WHEN_EMPTY = [
    "AVG_TXN_AMOUNT",
    "STD_TXN_AMOUNT",
    "MAX_TXN_AMOUNT",
    "MIN_TXN_AMOUNT",
    "AVG_TXN_HOUR",
    "STD_TXN_HOUR",
    "INTL_TXN_RATIO",
    "CARD_PRESENT_RATIO",
]
_AGG_BASE_NAMES = [
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
HAS_HISTORY_COL = "HAS_HISTORY"


def agg_prefix(window_days: int = AGG_WINDOW_DAYS) -> str:
    return f"D{window_days}_"


def agg_col(name: str, window_days: int = AGG_WINDOW_DAYS) -> str:
    return f"{agg_prefix(window_days)}{name}"


def last_txn_timestamp_col(window_days: int = AGG_WINDOW_DAYS) -> str:
    """Helper column (not a model feature) used to derive time-since-last-txn."""
    return agg_col("LAST_TXN_TIMESTAMP", window_days)


def get_agg_feature_columns(window_days: int = AGG_WINDOW_DAYS) -> List[str]:
    return [agg_col(c, window_days) for c in _AGG_BASE_NAMES] + [HAS_HISTORY_COL]


def get_online_agg_columns(window_days: int = AGG_WINDOW_DAYS) -> List[str]:
    """Columns served by the online ``CUSTOMER_AGG_FEATURES`` Feature View."""
    return get_agg_feature_columns(window_days) + [last_txn_timestamp_col(window_days)]


def get_derived_feature_columns(window_days: int = AGG_WINDOW_DAYS) -> List[str]:
    return [
        "AMOUNT_TO_AVG_RATIO",
        "AMOUNT_TO_MAX_RATIO",
        "IS_HIGH_VALUE_TXN",
        "IS_ODD_HOUR",
        agg_col("MINUTES_SINCE_LAST_TXN", window_days),
    ]


def get_feature_columns(
    window_days: int = AGG_WINDOW_DAYS,
) -> Tuple[List[str], str]:
    """Return (feature_columns, label_column) in the exact order the model expects.

    The order matters: the SPCS ``/predict-proba`` endpoint takes positional
    rows matching the signature inferred from training data with this order.
    """
    feature_cols = (
        TRANSACTION_FEATURE_COLS
        + PROFILE_FEATURE_COLS
        + get_agg_feature_columns(window_days)
        + get_derived_feature_columns(window_days)
    )
    return feature_cols, LABEL_COL


FEATURE_COLS, _ = get_feature_columns()


def get_entity_key_columns() -> List[str]:
    """Return entity key columns for Feature Store."""
    return [CUSTOMER_COL, ID_COL]


def is_odd_hour(hour):
    """Odd hours are 23:00-05:59. Works on scalars, numpy arrays and Series."""
    return (hour < 6) | (hour > 22)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

FRAUD_PATTERNS = ["high_amount", "odd_hours", "rapid_fire", "international"]


def _set_time_fields_from_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df[TIMESTAMP_COL])
    df["TRANSACTION_HOUR"] = ts.dt.hour.astype(int)
    df["TRANSACTION_DAY_OF_WEEK"] = ts.dt.weekday.astype(int)
    df["IS_WEEKEND"] = (ts.dt.weekday >= 5).astype(int)
    return df


def generate_synthetic_fraud_data(
    n_transactions: int = 100000, fraud_ratio: float = 0.02, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic transaction data for fraud detection.

    Fraud patterns are injected on the raw fields (timestamp, amount, channel
    flags) and all time-derived fields are recomputed from the final
    timestamp, so every row is internally consistent. Customer-relative
    features (aggregates, ratios, ``IS_HIGH_VALUE_TXN``, ``RISK_SCORE``) are
    computed downstream from these final values by
    :func:`compute_aggregated_features` and :func:`add_derived_features`.

    Args:
        n_transactions: Number of transactions to generate
        fraud_ratio: Proportion of fraudulent transactions
        seed: Random seed for reproducibility

    Returns:
        DataFrame with transaction fields, a ``FRAUD_PATTERN`` column
        (``None`` for legitimate rows) and ``IS_FRAUD`` labels.
    """
    np.random.seed(seed)
    random.seed(seed)

    n_customers = max(n_transactions // 50, 1)
    n_merchants = max(n_transactions // 200, 1)

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
            days=int(np.random.randint(0, 365)),
            hours=int(np.random.randint(0, 24)),
            minutes=int(np.random.randint(0, 60)),
        )
        for _ in range(n_transactions)
    ]

    df = pd.DataFrame(
        {
            ID_COL: [f"TXN_{i:010d}" for i in range(n_transactions)],
            CUSTOMER_COL: np.random.choice(customer_ids, n_transactions),
            "MERCHANT_ID": np.random.choice(merchant_ids, n_transactions),
            "MERCHANT_CATEGORY": np.random.choice(merchant_categories, n_transactions),
            TIMESTAMP_COL: pd.to_datetime(timestamps),
            "TRANSACTION_AMOUNT": np.abs(np.random.lognormal(3.5, 1.2, n_transactions)),
            "CARD_PRESENT": np.random.choice([0, 1], n_transactions, p=[0.3, 0.7]),
            "IS_INTERNATIONAL": np.random.choice([0, 1], n_transactions, p=[0.9, 0.1]),
        }
    )

    n_fraud = int(n_transactions * fraud_ratio)
    fraud_indices = np.random.choice(n_transactions, n_fraud, replace=False)
    df["IS_FRAUD"] = 0
    df.loc[fraud_indices, "IS_FRAUD"] = 1
    df["FRAUD_PATTERN"] = None

    rows_by_customer = df.groupby(CUSTOMER_COL).indices

    for idx in fraud_indices:
        fraud_pattern = np.random.choice(FRAUD_PATTERNS)
        df.at[idx, "FRAUD_PATTERN"] = fraud_pattern

        if fraud_pattern == "high_amount":
            df.at[idx, "TRANSACTION_AMOUNT"] = np.random.uniform(500, 5000)
        elif fraud_pattern == "odd_hours":
            ts = df.at[idx, TIMESTAMP_COL]
            df.at[idx, TIMESTAMP_COL] = ts.replace(
                hour=int(np.random.choice([2, 3, 4, 5]))
            )
        elif fraud_pattern == "rapid_fire":
            # Burst right after another transaction from the same customer.
            siblings = [
                i for i in rows_by_customer[df.at[idx, CUSTOMER_COL]] if i != idx
            ]
            if siblings:
                anchor = df.at[int(np.random.choice(siblings)), TIMESTAMP_COL]
                df.at[idx, TIMESTAMP_COL] = anchor + pd.Timedelta(
                    minutes=int(np.random.randint(1, 11))
                )
            df.at[idx, "TRANSACTION_AMOUNT"] = np.random.uniform(50, 200)
        elif fraud_pattern == "international":
            df.at[idx, "IS_INTERNATIONAL"] = 1
            df.at[idx, "CARD_PRESENT"] = 0

    df["TRANSACTION_AMOUNT"] = df["TRANSACTION_AMOUNT"].round(2)
    df = _set_time_fields_from_timestamp(df)

    column_order = [
        ID_COL,
        CUSTOMER_COL,
        "MERCHANT_ID",
        "MERCHANT_CATEGORY",
        TIMESTAMP_COL,
        "TRANSACTION_AMOUNT",
        "TRANSACTION_HOUR",
        "TRANSACTION_DAY_OF_WEEK",
        "IS_WEEKEND",
        "CARD_PRESENT",
        "IS_INTERNATIONAL",
        "IS_FRAUD",
        "FRAUD_PATTERN",
    ]
    return df[column_order]


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
        CUSTOMER_COL: customer_ids,
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


# ---------------------------------------------------------------------------
# Feature engineering (pandas)
# ---------------------------------------------------------------------------


def compute_aggregated_features(
    transactions_df: pd.DataFrame, window_days: int = AGG_WINDOW_DAYS
) -> pd.DataFrame:
    """Compute point-in-time customer aggregates for every transaction.

    For each transaction ``t`` of customer ``c`` the aggregates use only
    transactions of ``c`` with ``t - window_days <= ts < t`` (strictly before,
    so the transaction itself and same-timestamp transactions are excluded).

    Customers without prior history in the window get ``HAS_HISTORY = 0``,
    zero counts/sums and NaN for mean/std/min/max/ratio aggregates (XGBoost
    handles missing values natively). Std aggregates are also NaN when only a
    single prior transaction exists.

    Args:
        transactions_df: Raw transactions with ``TRANSACTION_ID``,
            ``CUSTOMER_ID``, ``TRANSACTION_TIMESTAMP``, ``TRANSACTION_AMOUNT``,
            ``TRANSACTION_HOUR``, ``IS_INTERNATIONAL`` and ``CARD_PRESENT``.
        window_days: Lookback window in days.

    Returns:
        One row per input transaction: ``TRANSACTION_ID``, the ``D{w}_*``
        aggregates, ``HAS_HISTORY`` and ``D{w}_LAST_TXN_TIMESTAMP``.
    """
    df = transactions_df[
        [
            ID_COL,
            CUSTOMER_COL,
            TIMESTAMP_COL,
            "TRANSACTION_AMOUNT",
            "TRANSACTION_HOUR",
            "IS_INTERNATIONAL",
            "CARD_PRESENT",
        ]
    ].copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df["_TS_SECONDS"] = (df[TIMESTAMP_COL] - pd.Timestamp("1970-01-01")) / pd.Timedelta(
        seconds=1
    )
    for c in [
        "TRANSACTION_AMOUNT",
        "TRANSACTION_HOUR",
        "IS_INTERNATIONAL",
        "CARD_PRESENT",
    ]:
        df[c] = df[c].astype(float)

    df = df.sort_values([CUSTOMER_COL, TIMESTAMP_COL], kind="mergesort").reset_index(
        drop=True
    )

    rolling = (
        df.set_index(TIMESTAMP_COL)
        .groupby(CUSTOMER_COL, sort=False)
        .rolling(f"{window_days}D", closed="left")
    )
    amount = rolling["TRANSACTION_AMOUNT"]
    hour = rolling["TRANSACTION_HOUR"]
    intl = rolling["IS_INTERNATIONAL"]
    card = rolling["CARD_PRESENT"]

    def _vals(series: pd.Series) -> np.ndarray:
        return series.to_numpy()

    count = _vals(amount.count())
    out = pd.DataFrame({ID_COL: df[ID_COL].to_numpy()})
    out[agg_col("AVG_TXN_AMOUNT", window_days)] = _vals(amount.mean())
    out[agg_col("STD_TXN_AMOUNT", window_days)] = _vals(amount.std())
    out[agg_col("MAX_TXN_AMOUNT", window_days)] = _vals(amount.max())
    out[agg_col("MIN_TXN_AMOUNT", window_days)] = _vals(amount.min())
    out[agg_col("TOTAL_TXN_AMOUNT", window_days)] = _vals(amount.sum())
    out[agg_col("TXN_COUNT", window_days)] = count
    out[agg_col("AVG_TXN_HOUR", window_days)] = _vals(hour.mean())
    out[agg_col("STD_TXN_HOUR", window_days)] = _vals(hour.std())
    out[agg_col("INTL_TXN_COUNT", window_days)] = _vals(intl.sum())
    out[agg_col("INTL_TXN_RATIO", window_days)] = _vals(intl.mean())
    out[agg_col("CARD_PRESENT_RATIO", window_days)] = _vals(card.mean())
    last_seconds = _vals(rolling["_TS_SECONDS"].max())

    has_history = np.nan_to_num(count, nan=0.0) > 0
    for name in _AGG_ZERO_WHEN_EMPTY:
        col = agg_col(name, window_days)
        out[col] = np.where(has_history, out[col], 0.0)
    for name in _AGG_NAN_WHEN_EMPTY:
        col = agg_col(name, window_days)
        out[col] = np.where(has_history, out[col], np.nan)
    for name in ["TXN_COUNT", "INTL_TXN_COUNT"]:
        col = agg_col(name, window_days)
        out[col] = out[col].astype(int)
    out[HAS_HISTORY_COL] = has_history.astype(int)
    out[last_txn_timestamp_col(window_days)] = pd.to_datetime(
        np.where(has_history, last_seconds, np.nan), unit="s"
    )

    order = transactions_df[ID_COL].reset_index(drop=True)
    return out.set_index(ID_COL).loc[order].reset_index()


def add_derived_features(
    df: pd.DataFrame, window_days: int = AGG_WINDOW_DAYS
) -> pd.DataFrame:
    """Add derived features (and the excluded ``RISK_SCORE`` baseline) in place.

    Shared by offline training data preparation and online serving
    (:func:`build_serving_feature_row`). Mirrors
    :func:`add_derived_features_snowpark`.
    """
    avg = df[agg_col("AVG_TXN_AMOUNT", window_days)].astype(float)
    mx = df[agg_col("MAX_TXN_AMOUNT", window_days)].astype(float)
    amount = df["TRANSACTION_AMOUNT"].astype(float)
    hour = df["TRANSACTION_HOUR"].astype(int)

    df["AMOUNT_TO_AVG_RATIO"] = amount / (avg + 1)
    df["AMOUNT_TO_MAX_RATIO"] = amount / (mx + 1)
    df["IS_HIGH_VALUE_TXN"] = (amount > avg * 3).astype(int)
    df["IS_ODD_HOUR"] = is_odd_hour(hour).astype(int)

    last_ts = pd.to_datetime(df[last_txn_timestamp_col(window_days)])
    txn_ts = pd.to_datetime(df[TIMESTAMP_COL])
    df[agg_col("MINUTES_SINCE_LAST_TXN", window_days)] = (
        txn_ts - last_ts
    ) / pd.Timedelta(minutes=1)

    df[BASELINE_SCORE_COL] = (
        df["IS_INTERNATIONAL"] * 2
        + df["IS_HIGH_VALUE_TXN"] * 1.5
        + df["IS_ODD_HOUR"] * 1
        + (1 - df["CARD_PRESENT"]) * 1.5
        + df["NUM_PREVIOUS_FRAUD_REPORTS"] * 3
    )
    return df


def prepare_model_features(
    transactions_df: pd.DataFrame,
    customer_profiles_df: pd.DataFrame,
    agg_features_df: Optional[pd.DataFrame] = None,
    window_days: int = AGG_WINDOW_DAYS,
) -> pd.DataFrame:
    """Prepare the final, point-in-time correct feature set for training.

    Args:
        transactions_df: Raw transaction data
        customer_profiles_df: Customer profile features
        agg_features_df: Per-transaction output of
            :func:`compute_aggregated_features`; computed if omitted.
        window_days: Aggregation lookback window in days.

    Returns:
        DataFrame with all columns from :func:`get_feature_columns`, the
        label, identifiers, timestamp and the ``RISK_SCORE`` baseline.
    """
    if agg_features_df is None:
        agg_features_df = compute_aggregated_features(transactions_df, window_days)
    df = transactions_df.merge(customer_profiles_df, on=CUSTOMER_COL, how="left")
    df = df.merge(agg_features_df, on=ID_COL, how="left", validate="one_to_one")
    return add_derived_features(df, window_days)


def build_serving_feature_row(
    transaction: Dict,
    profile_features: Dict,
    agg_features: Dict,
    window_days: int = AGG_WINDOW_DAYS,
) -> Dict:
    """Build one model input row for real-time scoring.

    ``agg_features`` are the customer's current-window aggregates from the
    online Feature Store (all transactions strictly before now). A missing
    row means no history, matching the training-time semantics.
    """
    row = {c: transaction[c] for c in TRANSACTION_FEATURE_COLS}
    row[TIMESTAMP_COL] = transaction[TIMESTAMP_COL]
    for c in PROFILE_FEATURE_COLS:
        row[c] = profile_features.get(c)

    count = agg_features.get(agg_col("TXN_COUNT", window_days)) or 0
    has_history = count > 0
    for name in _AGG_BASE_NAMES:
        col = agg_col(name, window_days)
        value = agg_features.get(col) if has_history else None
        if value is None:
            value = 0 if name in _AGG_ZERO_WHEN_EMPTY else np.nan
        row[col] = value
    row[HAS_HISTORY_COL] = int(has_history)
    last_col = last_txn_timestamp_col(window_days)
    row[last_col] = agg_features.get(last_col) if has_history else None

    frame = add_derived_features(pd.DataFrame([row]), window_days)
    feature_cols, _ = get_feature_columns(window_days)
    return frame.iloc[0][feature_cols].to_dict()


def to_model_input_vector(feature_row: Dict) -> List:
    """Order a feature row as the model expects; NaN -> None for JSON."""
    vector = []
    for c in FEATURE_COLS:
        value = feature_row.get(c)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            vector.append(None)
        elif isinstance(value, (np.integer, np.bool_)):
            vector.append(int(value))
        else:
            vector.append(float(value))
    return vector


# ---------------------------------------------------------------------------
# Feature engineering (Snowpark) - mirrors the pandas implementation above
# ---------------------------------------------------------------------------


def _snowpark_agg_exprs(h, window_days: int):
    from snowflake.snowpark import functions as F
    from snowflake.snowpark.types import FloatType

    a = lambda name: agg_col(name, window_days)  # noqa: E731
    return [
        F.avg(h["TRANSACTION_AMOUNT"]).alias(a("AVG_TXN_AMOUNT")),
        F.stddev(h["TRANSACTION_AMOUNT"]).alias(a("STD_TXN_AMOUNT")),
        F.max(h["TRANSACTION_AMOUNT"]).alias(a("MAX_TXN_AMOUNT")),
        F.min(h["TRANSACTION_AMOUNT"]).alias(a("MIN_TXN_AMOUNT")),
        F.sum(h["TRANSACTION_AMOUNT"]).alias(a("TOTAL_TXN_AMOUNT")),
        F.count(h["TRANSACTION_AMOUNT"]).alias(a("TXN_COUNT")),
        F.avg(h["TRANSACTION_HOUR"]).alias(a("AVG_TXN_HOUR")),
        F.stddev(h["TRANSACTION_HOUR"]).alias(a("STD_TXN_HOUR")),
        F.sum(h["IS_INTERNATIONAL"]).alias(a("INTL_TXN_COUNT")),
        F.avg(h["IS_INTERNATIONAL"].cast(FloatType())).alias(a("INTL_TXN_RATIO")),
        F.avg(h["CARD_PRESENT"].cast(FloatType())).alias(a("CARD_PRESENT_RATIO")),
        F.max(h[TIMESTAMP_COL]).alias(last_txn_timestamp_col(window_days)),
    ]


def _with_has_history(df, window_days: int):
    from snowflake.snowpark import functions as F

    count = F.col(agg_col("TXN_COUNT", window_days))
    has_history = count > 0
    for name in ("TOTAL_TXN_AMOUNT", "INTL_TXN_COUNT"):
        col = agg_col(name, window_days)
        df = df.with_column(col, F.iff(has_history, F.col(col), F.lit(0)))
    return df.with_column(HAS_HISTORY_COL, F.iff(has_history, F.lit(1), F.lit(0)))


def build_point_in_time_agg_features_snowpark(
    transactions_df, window_days: int = AGG_WINDOW_DAYS
):
    """Per-transaction aggregates over prior transactions (Snowpark).

    Snowpark equivalent of :func:`compute_aggregated_features`: a LEFT
    self-join on ``CUSTOMER_ID`` restricted to
    ``spine.ts - window_days <= hist.ts < spine.ts``. Returns
    ``TRANSACTION_ID`` plus aggregate columns (no ``CUSTOMER_ID`` so it can
    back a ``TRANSACTION``-entity Feature View without key collisions).
    """
    from snowflake.snowpark import functions as F

    spine = transactions_df.select(
        F.col(ID_COL).alias("S_TXN_ID"),
        F.col(CUSTOMER_COL).alias("S_CUSTOMER_ID"),
        F.col(TIMESTAMP_COL).alias("S_TS"),
    )
    h = transactions_df.select(
        F.col(CUSTOMER_COL).alias("H_CUSTOMER_ID"),
        F.col(TIMESTAMP_COL),
        F.col("TRANSACTION_AMOUNT"),
        F.col("TRANSACTION_HOUR"),
        F.col("IS_INTERNATIONAL"),
        F.col("CARD_PRESENT"),
    )
    joined = spine.join(
        h,
        (spine["S_CUSTOMER_ID"] == h["H_CUSTOMER_ID"])
        & (h[TIMESTAMP_COL] < spine["S_TS"])
        & (h[TIMESTAMP_COL] >= F.dateadd("day", F.lit(-window_days), spine["S_TS"])),
        how="left",
    )
    aggregated = (
        joined.group_by("S_TXN_ID")
        .agg(*_snowpark_agg_exprs(h, window_days))
        .with_column_renamed("S_TXN_ID", ID_COL)
    )
    return _with_has_history(aggregated, window_days)


def build_current_agg_features_snowpark(
    transactions_df, window_days: int = AGG_WINDOW_DAYS
):
    """Per-customer aggregates over the last ``window_days`` as of now (Snowpark).

    Backs the online Feature View used for real-time serving: at request time
    every stored transaction is strictly before the one being scored, which
    matches :func:`build_point_in_time_agg_features_snowpark`.
    """
    from snowflake.snowpark import functions as F

    recent = transactions_df.filter(
        F.col(TIMESTAMP_COL)
        >= F.dateadd("day", F.lit(-window_days), F.current_timestamp())
    )
    aggregated = recent.group_by(CUSTOMER_COL).agg(
        *_snowpark_agg_exprs(recent, window_days)
    )
    return _with_has_history(aggregated, window_days)


def add_derived_features_snowpark(df, window_days: int = AGG_WINDOW_DAYS):
    """Snowpark equivalent of :func:`add_derived_features`."""
    from snowflake.snowpark import functions as F
    from snowflake.snowpark.types import FloatType

    amount = F.col("TRANSACTION_AMOUNT").cast(FloatType())
    avg = F.col(agg_col("AVG_TXN_AMOUNT", window_days)).cast(FloatType())
    mx = F.col(agg_col("MAX_TXN_AMOUNT", window_days)).cast(FloatType())
    hour = F.col("TRANSACTION_HOUR")
    odd = F.iff((hour < 6) | (hour > 22), F.lit(1), F.lit(0))
    high = F.iff(F.coalesce(amount > avg * 3, F.lit(False)), F.lit(1), F.lit(0))

    return (
        df.with_column("AMOUNT_TO_AVG_RATIO", amount / (avg + 1))
        .with_column("AMOUNT_TO_MAX_RATIO", amount / (mx + 1))
        .with_column("IS_HIGH_VALUE_TXN", high)
        .with_column("IS_ODD_HOUR", odd)
        .with_column(
            agg_col("MINUTES_SINCE_LAST_TXN", window_days),
            F.datediff(
                "second",
                F.col(last_txn_timestamp_col(window_days)),
                F.col(TIMESTAMP_COL),
            )
            / 60.0,
        )
        .with_column(
            BASELINE_SCORE_COL,
            F.col("IS_INTERNATIONAL") * 2
            + F.col("IS_HIGH_VALUE_TXN") * 1.5
            + F.col("IS_ODD_HOUR") * 1
            + (F.lit(1) - F.col("CARD_PRESENT")) * 1.5
            + F.col("NUM_PREVIOUS_FRAUD_REPORTS") * 3,
        )
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

DEFAULT_FPR_TARGETS = (0.01, 0.05)
DEFAULT_REVIEW_TARGET_RECALL = 0.90


def time_based_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    timestamp_col: str = TIMESTAMP_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into (train, validation, test).

    The most recent ``test_fraction`` of rows is the test set; the most
    recent ``val_fraction`` of the remainder is the validation set used for
    threshold selection.
    """
    ordered = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    test_start = int(round(n * (1 - test_fraction)))
    val_start = int(round(test_start * (1 - val_fraction)))
    return (
        ordered.iloc[:val_start],
        ordered.iloc[val_start:test_start],
        ordered.iloc[test_start:],
    )


def time_based_split_snowpark(
    df,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    timestamp_col: str = TIMESTAMP_COL,
):
    """Snowpark equivalent of :func:`time_based_split` -> (train, val, test)."""
    from snowflake.snowpark import Window
    from snowflake.snowpark import functions as F

    ranked = df.with_column(
        "_TIME_RANK", F.percent_rank().over(Window.order_by(F.col(timestamp_col)))
    )
    test_start = 1 - test_fraction
    val_start = test_start * (1 - val_fraction)
    rank = F.col("_TIME_RANK")
    return (
        ranked.filter(rank < val_start).drop("_TIME_RANK"),
        ranked.filter((rank >= val_start) & (rank < test_start)).drop("_TIME_RANK"),
        ranked.filter(rank >= test_start).drop("_TIME_RANK"),
    )


def recall_at_fpr(y_true: np.ndarray, y_proba: np.ndarray, max_fpr: float) -> float:
    """Highest recall achievable with false-positive rate <= ``max_fpr``."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def _fpr_key(max_fpr: float) -> str:
    return f"recall_at_{max_fpr * 100:g}pct_fpr".replace(".", "_")


def calculate_fraud_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    fpr_targets: Tuple[float, ...] = DEFAULT_FPR_TARGETS,
) -> Dict:
    """Threshold-free ranking metrics plus metrics at the operating threshold.

    Returns AUC-ROC, AUC-PR (average precision), recall at each FPR target
    (``recall_at_1pct_fpr``, ``recall_at_5pct_fpr``) and accuracy, precision,
    recall, F1 and the confusion matrix at ``threshold``.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "threshold": float(threshold),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
    }
    for max_fpr in fpr_targets:
        metrics[_fpr_key(max_fpr)] = recall_at_fpr(y_true, y_proba, max_fpr)
    metrics.update(
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "fraud_detection_rate": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        }
    )
    return metrics


def select_decision_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    review_target_recall: float = DEFAULT_REVIEW_TARGET_RECALL,
) -> Dict[str, float]:
    """Choose APPROVE/REVIEW/DECLINE band edges from validation data.

    - ``decline_threshold``: the precision-recall-curve threshold that
      maximizes F1. This is the model's single operating threshold, used for
      reported precision/recall/F1 and the binary ``predict`` decision.
    - ``review_threshold``: the highest threshold that still reaches
      ``review_target_recall`` (so REVIEW + DECLINE together catch at least
      that share of fraud), capped at ``decline_threshold``.

    Scores ``< review`` -> APPROVE, ``[review, decline)`` -> REVIEW,
    ``>= decline`` -> DECLINE.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where(
        precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
    )
    best = int(np.argmax(f1))
    decline = float(thresholds[best])

    meets_recall = recall >= review_target_recall
    review = (
        float(thresholds[meets_recall].max())
        if meets_recall.any()
        else float(thresholds.min())
    )
    review = min(review, decline)
    return {
        "review_threshold": review,
        "decline_threshold": decline,
        "review_target_recall": float(review_target_recall),
        "val_f1_at_decline": float(f1[best]),
    }


def get_decision(prob: float, review_threshold: float, decline_threshold: float) -> str:
    if prob >= decline_threshold:
        return "DECLINE"
    if prob >= review_threshold:
        return "REVIEW"
    return "APPROVE"


def compute_feature_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """XGBoost gain importance plus mean |SHAP| (TreeSHAP via ``pred_contribs``)."""
    import xgboost as xgb

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    contribs = booster.predict(xgb.DMatrix(X, missing=np.nan), pred_contribs=True)[
        :, :-1
    ]
    importance = pd.DataFrame(
        {
            "feature": list(X.columns),
            "gain": [float(gain.get(c, 0.0)) for c in X.columns],
            "mean_abs_shap": np.abs(contribs).mean(axis=0).astype(float),
        }
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Real-time simulation helpers
# ---------------------------------------------------------------------------


def most_recent_time_at_hour(now: datetime, hour: int) -> datetime:
    """Latest timestamp <= ``now`` whose hour is ``hour`` (keeps fields consistent)."""
    candidate = now.replace(hour=int(hour))
    if candidate > now:
        candidate -= timedelta(days=1)
    return candidate


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
        timestamp = most_recent_time_at_hour(
            timestamp, int(np.random.choice([1, 2, 3, 4, 23]))
        )
        is_international = np.random.choice([0, 1], p=[0.3, 0.7])
        card_present = 0
    else:
        amount = np.abs(np.random.lognormal(3.5, 1.2))
        is_international = np.random.choice([0, 1], p=[0.92, 0.08])
        card_present = np.random.choice([0, 1], p=[0.3, 0.7])

    return {
        ID_COL: f"TXN_{int(timestamp.timestamp() * 1000):015d}",
        CUSTOMER_COL: np.random.choice(customer_ids),
        "MERCHANT_ID": np.random.choice(merchant_ids),
        "MERCHANT_CATEGORY": np.random.choice(merchant_categories),
        TIMESTAMP_COL: timestamp,
        "TRANSACTION_AMOUNT": round(amount, 2),
        "TRANSACTION_HOUR": timestamp.hour,
        "TRANSACTION_DAY_OF_WEEK": timestamp.weekday(),
        "IS_WEEKEND": 1 if timestamp.weekday() >= 5 else 0,
        "CARD_PRESENT": card_present,
        "IS_INTERNATIONAL": is_international,
    }
