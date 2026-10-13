import math

import dag_procs as d


def test_promotes_on_auc_pr_gain():
    r = d.decide_promotion(
        {"auc_pr": 0.62, "recall_at_1pct_fpr": 0.50},
        {"auc_pr": 0.60, "recall_at_1pct_fpr": 0.50},
    )
    assert r["promote"]


def test_promotes_on_recall_at_fpr_gain():
    r = d.decide_promotion(
        {"auc_pr": 0.60, "recall_at_1pct_fpr": 0.55},
        {"auc_pr": 0.60, "recall_at_1pct_fpr": 0.50},
    )
    assert r["promote"]


def test_requires_more_than_one_percent():
    r = d.decide_promotion(
        {"auc_pr": 0.605, "recall_at_1pct_fpr": 0.5},
        {"auc_pr": 0.60, "recall_at_1pct_fpr": 0.5},
    )
    assert not r["promote"]
    assert "KEEP_CHAMPION" in r["message"]


def test_blocks_when_other_metric_regresses():
    r = d.decide_promotion(
        {"auc_pr": 0.70, "recall_at_1pct_fpr": 0.40},
        {"auc_pr": 0.60, "recall_at_1pct_fpr": 0.50},
    )
    assert not r["promote"]
    assert "regressed" in r["message"]


def test_auc_roc_alone_does_not_promote():
    r = d.decide_promotion(
        {"auc_pr": 0.60, "recall_at_1pct_fpr": 0.50, "auc_roc": 0.99},
        {"auc_pr": 0.60, "recall_at_1pct_fpr": 0.50, "auc_roc": 0.90},
    )
    assert not r["promote"]


def test_legacy_champion_without_metrics_is_replaced():
    r = d.decide_promotion(
        {"auc_pr": 0.3, "recall_at_1pct_fpr": 0.2},
        {"auc_pr": None, "recall_at_1pct_fpr": 0.0},
    )
    assert r["promote"]
    assert math.isinf(r["changes"]["auc_pr"])


def test_missing_challenger_metric_blocks():
    r = d.decide_promotion(
        {"auc_pr": None, "recall_at_1pct_fpr": 0.9},
        {"auc_pr": 0.5, "recall_at_1pct_fpr": 0.5},
    )
    assert not r["promote"]


def test_drifted_features():
    assert d.drifted_features({"A": 0.05, "B": 0.25, "C": None}, 0.2) == {"B": 0.25}
