import numpy as np
import pandas as pd
import pytest

import utils as u


def test_time_based_split_holds_out_most_recent():
    ts = pd.date_range("2025-01-01", periods=1000, freq="h")
    df = pd.DataFrame({"TRANSACTION_TIMESTAMP": ts[::-1], "X": range(1000)})
    train, val, test = u.time_based_split(df, test_fraction=0.2, val_fraction=0.2)
    assert (len(train), len(val), len(test)) == (640, 160, 200)
    assert train["TRANSACTION_TIMESTAMP"].max() < val["TRANSACTION_TIMESTAMP"].min()
    assert val["TRANSACTION_TIMESTAMP"].max() < test["TRANSACTION_TIMESTAMP"].min()
    assert test["TRANSACTION_TIMESTAMP"].max() == ts.max()


def test_recall_at_fpr():
    y = np.array([0] * 100 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.5, 100), np.linspace(0.45, 1.0, 10)])
    assert u.recall_at_fpr(y, p, 0.0) == pytest.approx(0.9)
    assert u.recall_at_fpr(y, p, 0.15) == 1.0  # ~10% of negatives score above 0.45


def test_calculate_fraud_metrics_keys_and_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 500), 0, 1)
    m = u.calculate_fraud_metrics(y, p, threshold=0.5)
    for key in (
        "auc_roc",
        "auc_pr",
        "recall_at_1pct_fpr",
        "recall_at_5pct_fpr",
        "precision",
        "recall",
        "f1",
    ):
        assert 0.0 <= m[key] <= 1.0
    assert m["threshold"] == 0.5
    assert m["true_positives"] == int(((p >= 0.5) & (y == 1)).sum())


def test_select_decision_thresholds_orders_bands():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.03).astype(int)
    p = np.clip(y * 0.5 + rng.beta(1, 8, 5000), 0, 1)
    th = u.select_decision_thresholds(y, p, review_target_recall=0.9)
    assert th["review_threshold"] <= th["decline_threshold"]
    review_recall = ((p >= th["review_threshold"]) & (y == 1)).sum() / y.sum()
    assert review_recall >= 0.9
    best_f1 = u.calculate_fraud_metrics(y, p, th["decline_threshold"])["f1"]
    assert best_f1 >= u.calculate_fraud_metrics(y, p, 0.5)["f1"] - 1e-9
    assert (
        u.get_decision(
            th["decline_threshold"], th["review_threshold"], th["decline_threshold"]
        )
        == "DECLINE"
    )
    assert (
        u.get_decision(0.0, th["review_threshold"], th["decline_threshold"])
        == "APPROVE"
    )


def test_feature_importance_has_every_feature():
    xgb = pytest.importorskip("xgboost")
    txns = u.generate_synthetic_fraud_data(n_transactions=2000, fraud_ratio=0.1, seed=2)
    profiles = u.generate_customer_profiles(txns["CUSTOMER_ID"].unique().tolist())
    feats = u.prepare_model_features(txns, profiles)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3).fit(
        feats[u.FEATURE_COLS], feats["IS_FRAUD"]
    )
    imp = u.compute_feature_importance(model, feats[u.FEATURE_COLS])
    assert set(imp["feature"]) == set(u.FEATURE_COLS)
    assert (imp["mean_abs_shap"] >= 0).all()
