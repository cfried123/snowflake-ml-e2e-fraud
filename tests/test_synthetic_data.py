"""Fraud-pattern injection keeps every field consistent."""

import pandas as pd
import pytest

import utils as u


@pytest.fixture(scope="module")
def data():
    txns = u.generate_synthetic_fraud_data(n_transactions=5000, fraud_ratio=0.1, seed=3)
    profiles = u.generate_customer_profiles(txns["CUSTOMER_ID"].unique().tolist())
    return txns, u.prepare_model_features(txns, profiles)


def test_time_fields_match_timestamp(data):
    txns, _ = data
    ts = pd.to_datetime(txns["TRANSACTION_TIMESTAMP"])
    assert (txns["TRANSACTION_HOUR"] == ts.dt.hour).all()
    assert (txns["TRANSACTION_DAY_OF_WEEK"] == ts.dt.weekday).all()
    assert (txns["IS_WEEKEND"] == (ts.dt.weekday >= 5).astype(int)).all()


def test_every_fraud_row_has_a_pattern(data):
    txns, _ = data
    assert txns.loc[txns["IS_FRAUD"] == 1, "FRAUD_PATTERN"].notna().all()
    assert txns.loc[txns["IS_FRAUD"] == 0, "FRAUD_PATTERN"].isna().all()
    assert set(txns["FRAUD_PATTERN"].dropna()) == set(u.FRAUD_PATTERNS)


def test_odd_hours_pattern(data):
    _, feats = data
    rows = feats[feats["FRAUD_PATTERN"] == "odd_hours"]
    assert rows["TRANSACTION_HOUR"].isin([2, 3, 4, 5]).all()
    assert (rows["IS_ODD_HOUR"] == 1).all()
    assert (
        pd.to_datetime(rows["TRANSACTION_TIMESTAMP"]).dt.hour
        == rows["TRANSACTION_HOUR"]
    ).all()


def test_international_pattern(data):
    _, feats = data
    rows = feats[feats["FRAUD_PATTERN"] == "international"]
    assert (rows["IS_INTERNATIONAL"] == 1).all()
    assert (rows["CARD_PRESENT"] == 0).all()
    avg = rows[u.agg_col("AVG_TXN_AMOUNT")]
    expected_high = (rows["TRANSACTION_AMOUNT"] > avg * 3).astype(int)
    assert (rows["IS_HIGH_VALUE_TXN"] == expected_high).all()


def test_high_amount_pattern(data):
    _, feats = data
    rows = feats[feats["FRAUD_PATTERN"] == "high_amount"]
    assert rows["TRANSACTION_AMOUNT"].between(500, 5000).all()


def test_rapid_fire_follows_same_customer_transaction(data):
    txns, feats = data
    rows = feats[feats["FRAUD_PATTERN"] == "rapid_fire"]
    assert rows["TRANSACTION_AMOUNT"].between(50, 200).all()
    minutes = rows[u.agg_col("MINUTES_SINCE_LAST_TXN")]
    assert (minutes.dropna() <= 10).mean() > 0.9


def test_risk_score_recomputed_from_final_fields(data):
    _, feats = data
    expected = (
        feats["IS_INTERNATIONAL"] * 2
        + feats["IS_HIGH_VALUE_TXN"] * 1.5
        + feats["IS_ODD_HOUR"] * 1
        + (1 - feats["CARD_PRESENT"]) * 1.5
        + feats["NUM_PREVIOUS_FRAUD_REPORTS"] * 3
    )
    pd.testing.assert_series_equal(feats["RISK_SCORE"], expected, check_names=False)
