"""Point-in-time correctness of compute_aggregated_features."""

import numpy as np
import pandas as pd
import pytest

import utils as u

W = u.AGG_WINDOW_DAYS


def _txns(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "TRANSACTION_ID",
            "CUSTOMER_ID",
            "TRANSACTION_TIMESTAMP",
            "TRANSACTION_AMOUNT",
            "IS_INTERNATIONAL",
            "CARD_PRESENT",
        ],
    )
    df["TRANSACTION_TIMESTAMP"] = pd.to_datetime(df["TRANSACTION_TIMESTAMP"])
    df["TRANSACTION_HOUR"] = df["TRANSACTION_TIMESTAMP"].dt.hour
    return df


@pytest.fixture(scope="module")
def synthetic():
    return u.generate_synthetic_fraud_data(
        n_transactions=3000, fraud_ratio=0.05, seed=7
    )


def _brute_force(df, window_days):
    """Reference implementation: explicit scan over strictly-earlier rows."""
    out = {}
    ts = pd.to_datetime(df["TRANSACTION_TIMESTAMP"])
    for i, row in df.iterrows():
        t = ts[i]
        mask = (
            (df["CUSTOMER_ID"] == row["CUSTOMER_ID"])
            & (ts < t)
            & (ts >= t - pd.Timedelta(days=window_days))
        )
        hist = df[mask]
        out[row["TRANSACTION_ID"]] = {
            "count": len(hist),
            "sum": hist["TRANSACTION_AMOUNT"].sum(),
            "max": hist["TRANSACTION_AMOUNT"].max() if len(hist) else np.nan,
            "std": hist["TRANSACTION_AMOUNT"].std() if len(hist) > 1 else np.nan,
        }
    return out


def test_matches_brute_force_reference(synthetic):
    sample = synthetic[
        synthetic["CUSTOMER_ID"].isin(synthetic["CUSTOMER_ID"].unique()[:10])
    ]
    sample = sample.reset_index(drop=True)
    agg = u.compute_aggregated_features(sample).set_index("TRANSACTION_ID")
    ref = _brute_force(sample, W)
    for txn_id, r in ref.items():
        got = agg.loc[txn_id]
        assert got[u.agg_col("TXN_COUNT")] == r["count"]
        assert got[u.agg_col("TOTAL_TXN_AMOUNT")] == pytest.approx(r["sum"])
        np.testing.assert_allclose(
            got[u.agg_col("MAX_TXN_AMOUNT")], r["max"], equal_nan=True
        )
        np.testing.assert_allclose(
            got[u.agg_col("STD_TXN_AMOUNT")], r["std"], rtol=1e-9, equal_nan=True
        )
        assert got[u.HAS_HISTORY_COL] == int(r["count"] > 0)


def test_future_transactions_do_not_leak(synthetic):
    """Temporal causality: mutating/adding later transactions never changes earlier features."""
    cutoff = synthetic["TRANSACTION_TIMESTAMP"].quantile(0.5)
    before = u.compute_aggregated_features(synthetic).set_index("TRANSACTION_ID")

    perturbed = synthetic.copy()
    future = perturbed["TRANSACTION_TIMESTAMP"] >= cutoff
    perturbed.loc[future, "TRANSACTION_AMOUNT"] *= 100
    perturbed.loc[future, "IS_INTERNATIONAL"] = 1
    perturbed.loc[future, "CARD_PRESENT"] = 0
    extra = perturbed[future].copy()
    extra["TRANSACTION_ID"] = extra["TRANSACTION_ID"] + "_DUP"
    perturbed = pd.concat([perturbed, extra], ignore_index=True)
    after = u.compute_aggregated_features(perturbed).set_index("TRANSACTION_ID")

    past_ids = synthetic.loc[~future, "TRANSACTION_ID"]
    cols = u.get_agg_feature_columns() + [u.last_txn_timestamp_col()]
    pd.testing.assert_frame_equal(before.loc[past_ids, cols], after.loc[past_ids, cols])


def test_current_and_same_timestamp_rows_excluded():
    df = _txns(
        [
            ("T1", "C1", "2025-01-01 10:00", 10.0, 0, 1),
            ("T2", "C1", "2025-01-02 10:00", 20.0, 1, 0),
            ("T3", "C1", "2025-01-02 10:00", 999.0, 0, 1),
        ]
    )
    agg = u.compute_aggregated_features(df).set_index("TRANSACTION_ID")
    for txn in ("T2", "T3"):
        assert agg.loc[txn, u.agg_col("TXN_COUNT")] == 1
        assert agg.loc[txn, u.agg_col("MAX_TXN_AMOUNT")] == 10.0


def test_window_days_lookback_is_respected():
    df = _txns(
        [
            ("T1", "C1", "2025-01-01 00:00", 100.0, 0, 1),
            ("T2", "C1", "2025-01-05 00:00", 50.0, 0, 1),
            ("T3", "C1", "2025-01-09 00:00", 10.0, 0, 1),
        ]
    )
    agg7 = u.compute_aggregated_features(df, window_days=7).set_index("TRANSACTION_ID")
    agg3 = u.compute_aggregated_features(df, window_days=3).set_index("TRANSACTION_ID")
    assert agg7.loc["T3", "D7_TXN_COUNT"] == 1  # T1 is 8 days earlier
    assert agg7.loc["T3", "D7_MAX_TXN_AMOUNT"] == 50.0
    assert agg3.loc["T3", "D3_TXN_COUNT"] == 0
    assert agg3.loc["T3", u.HAS_HISTORY_COL] == 0


def test_no_history_flagged_instead_of_zero_filled():
    df = _txns(
        [
            ("T1", "C1", "2025-01-01 00:00", 100.0, 1, 0),
            ("T2", "C1", "2025-01-02 00:00", 50.0, 0, 1),
            ("T3", "C2", "2025-01-02 00:00", 70.0, 0, 1),
        ]
    )
    agg = u.compute_aggregated_features(df).set_index("TRANSACTION_ID")
    first = agg.loc["T1"]
    assert first[u.HAS_HISTORY_COL] == 0
    assert first[u.agg_col("TXN_COUNT")] == 0
    assert first[u.agg_col("TOTAL_TXN_AMOUNT")] == 0
    assert np.isnan(first[u.agg_col("AVG_TXN_AMOUNT")])
    assert np.isnan(first[u.agg_col("STD_TXN_AMOUNT")])
    # One prior transaction: std is undefined, not 0.
    assert agg.loc["T2", u.HAS_HISTORY_COL] == 1
    assert np.isnan(agg.loc["T2", u.agg_col("STD_TXN_AMOUNT")])
    assert agg.loc["T2", u.agg_col("INTL_TXN_RATIO")] == 1.0
    assert agg.loc["T3", u.HAS_HISTORY_COL] == 0


def test_output_is_one_row_per_transaction_in_input_order(synthetic):
    shuffled = synthetic.sample(frac=1.0, random_state=0).reset_index(drop=True)
    agg = u.compute_aggregated_features(shuffled)
    assert list(agg["TRANSACTION_ID"]) == list(shuffled["TRANSACTION_ID"])


def test_minutes_since_last_txn():
    df = _txns(
        [
            ("T1", "C1", "2025-01-01 10:00", 10.0, 0, 1),
            ("T2", "C1", "2025-01-01 10:05", 20.0, 0, 1),
        ]
    )
    profiles = pd.DataFrame({"CUSTOMER_ID": ["C1"], "NUM_PREVIOUS_FRAUD_REPORTS": [0]})
    for c in ("TRANSACTION_DAY_OF_WEEK", "IS_WEEKEND"):
        df[c] = 0
    feats = u.prepare_model_features(df, profiles).set_index("TRANSACTION_ID")
    col = u.agg_col("MINUTES_SINCE_LAST_TXN")
    assert np.isnan(feats.loc["T1", col])
    assert feats.loc["T2", col] == pytest.approx(5.0)
