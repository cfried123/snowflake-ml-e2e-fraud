"""One feature contract shared by training, serving and the baseline."""

import json
import os
from datetime import datetime

import numpy as np
import pytest

import utils as u

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_train_uses_utils_feature_columns():
    train = pytest.importorskip("train")
    feature_cols, label = u.get_feature_columns()
    assert train.FEATURE_COLS is u.FEATURE_COLS
    assert list(train.FEATURE_COLS) == feature_cols
    assert train.LABEL_COL == label == "IS_FRAUD"


def test_feature_columns_unique_and_exclude_baseline_and_ids():
    feature_cols, label = u.get_feature_columns()
    assert len(feature_cols) == len(set(feature_cols))
    for excluded in (
        u.BASELINE_SCORE_COL,
        label,
        u.ID_COL,
        u.CUSTOMER_COL,
        u.TIMESTAMP_COL,
    ):
        assert excluded not in feature_cols
    assert u.HAS_HISTORY_COL in feature_cols
    assert u.last_txn_timestamp_col() not in feature_cols


def test_prepare_model_features_produces_every_feature():
    txns = u.generate_synthetic_fraud_data(n_transactions=500, seed=1)
    profiles = u.generate_customer_profiles(txns["CUSTOMER_ID"].unique().tolist())
    feats = u.prepare_model_features(txns, profiles)
    assert set(u.FEATURE_COLS) <= set(feats.columns)
    assert u.BASELINE_SCORE_COL in feats.columns


def test_serving_row_matches_training_row():
    """Serving with as-of-now aggregates reproduces the training features."""
    txns = u.generate_synthetic_fraud_data(n_transactions=2000, seed=5)
    profiles = u.generate_customer_profiles(txns["CUSTOMER_ID"].unique().tolist())
    feats = u.prepare_model_features(txns, profiles)
    agg = u.compute_aggregated_features(txns).set_index("TRANSACTION_ID")
    profile_by_customer = profiles.set_index("CUSTOMER_ID")

    for _, row in feats.sample(50, random_state=0).iterrows():
        online_agg = agg.loc[
            row["TRANSACTION_ID"], u.get_online_agg_columns()
        ].to_dict()
        if not online_agg[u.HAS_HISTORY_COL]:
            online_agg = {}  # online store has no row for customers without history
        serving = u.build_serving_feature_row(
            row.to_dict(),
            profile_by_customer.loc[row["CUSTOMER_ID"]].to_dict(),
            online_agg,
        )
        expected = row[u.FEATURE_COLS].astype(float).to_numpy()
        got = np.array([serving[c] for c in u.FEATURE_COLS], dtype=float)
        np.testing.assert_allclose(got, expected, equal_nan=True)


def test_simulator_vector_order_and_length():
    inference = pytest.importorskip("simulator.inference", reason="needs snowpark")
    online = {
        inference.PROFILE_FV_NAME: {c: 1 for c in u.PROFILE_FEATURE_COLS},
        inference.AGG_FV_NAME: {
            u.agg_col("TXN_COUNT"): 3,
            u.agg_col("AVG_TXN_AMOUNT"): 40.0,
            u.agg_col("MAX_TXN_AMOUNT"): 80.0,
            u.last_txn_timestamp_col(): datetime(2026, 1, 1, 9, 0),
        },
    }
    vector = inference.build_feature_vector(
        online,
        amount=500.0,
        hour=3,
        card_present=0,
        is_intl=1,
        now=datetime(2026, 1, 1, 12),
    )
    assert len(vector) == len(u.FEATURE_COLS)
    by_name = dict(zip(u.FEATURE_COLS, vector))
    assert by_name["TRANSACTION_AMOUNT"] == 500.0
    assert by_name["TRANSACTION_HOUR"] == 3
    assert by_name["IS_ODD_HOUR"] == 1
    assert by_name[u.HAS_HISTORY_COL] == 1
    assert by_name["IS_HIGH_VALUE_TXN"] == 1
    assert by_name[u.agg_col("STD_TXN_AMOUNT")] is None  # missing -> JSON null
    # Scored at the most recent 03:00 (2025-12-31 03:00 is before "now"); last txn 2026-01-01 09:00
    # would be later, so the simulator clamps to 0 rather than sending a negative gap.
    assert by_name[u.agg_col("MINUTES_SINCE_LAST_TXN")] == 0.0
    json.dumps(vector)

    no_history = inference.build_feature_vector(
        {inference.PROFILE_FV_NAME: online[inference.PROFILE_FV_NAME]}, 10.0, 12, 1, 0
    )
    assert dict(zip(u.FEATURE_COLS, no_history))[u.HAS_HISTORY_COL] == 0


def test_monitoring_notebook_uses_shared_feature_columns():
    with open(os.path.join(ROOT, "notebooks", "05_model_monitoring.ipynb")) as f:
        source = "\n".join(
            c["source"] for c in json.load(f)["cells"] if c["cell_type"] == "code"
        )
    assert "from utils import FEATURE_COLS" in source
    assert "FEATURE_COLS = [" not in source
