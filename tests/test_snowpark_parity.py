"""Snowpark feature SQL matches the pandas reference (Snowpark local testing mode)."""

import warnings

import numpy as np
import pytest

import utils as u

snowpark = pytest.importorskip("snowflake.snowpark")


@pytest.mark.slow
def test_point_in_time_snowpark_matches_pandas():
    warnings.filterwarnings("ignore")
    session = snowpark.Session.builder.config("local_testing", True).create()
    txns = u.generate_synthetic_fraud_data(n_transactions=300, fraud_ratio=0.1, seed=11)
    profiles = u.generate_customer_profiles(txns["CUSTOMER_ID"].unique().tolist())
    raw = txns.drop(columns=["FRAUD_PATTERN"])

    sdf = session.create_dataframe(raw)
    agg = u.build_point_in_time_agg_features_snowpark(sdf)
    full = sdf.join(session.create_dataframe(profiles), on="CUSTOMER_ID").join(
        agg, on="TRANSACTION_ID"
    )
    got = (
        u.add_derived_features_snowpark(full)
        .to_pandas()
        .set_index("TRANSACTION_ID")
        .sort_index()
    )
    expected = (
        u.prepare_model_features(raw, profiles).set_index("TRANSACTION_ID").sort_index()
    )

    for col in u.FEATURE_COLS + [u.BASELINE_SCORE_COL]:
        np.testing.assert_allclose(
            got[col].astype(float).to_numpy(),
            expected[col].astype(float).to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
            err_msg=col,
        )
