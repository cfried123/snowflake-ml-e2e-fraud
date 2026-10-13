"""
Fraud Detection Simulator Configuration
"""

import os

DATABASE = os.getenv("FRAUD_DB", "DEMO_FRAUD_DETECTION_DB")
SCHEMA = os.getenv("FRAUD_SCHEMA", "ML_PIPELINE")
FEATURE_STORE_SCHEMA = os.getenv("FRAUD_FS_SCHEMA", "FEATURE_STORE")
WAREHOUSE = os.getenv("FRAUD_WH", "ML_WH")
FEATURE_SERVING_WH = os.getenv("FRAUD_FS_WH", "FEATURE_SERVING_WH")

MODEL_NAME = os.getenv("FRAUD_MODEL", "FRAUD_DETECTION_MODEL")

TRANSACTIONS_TABLE = "TRANSACTIONS"
CUSTOMER_PROFILES_TABLE = "CUSTOMER_PROFILES"
AGGREGATED_FEATURES_TABLE = "AGGREGATED_FEATURES"

# Feature Store views served online (see notebooks/02_feature_engineering.ipynb)
PROFILE_FV_NAME = "CUSTOMER_PROFILE_FEATURES"
PROFILE_FV_VERSION = os.getenv("FRAUD_PROFILE_FV_VERSION", "V1")
AGG_FV_NAME = "CUSTOMER_AGG_FEATURES"
AGG_FV_VERSION = os.getenv("FRAUD_AGG_FV_VERSION", "V5")

# Decision bands. By default they are read from the deployed model version's
# registry metrics (review_threshold / decline_threshold, tuned on validation
# data by source/train.py). These env vars override; the fallback is 0.3/0.7.
REVIEW_THRESHOLD_OVERRIDE = os.getenv("FRAUD_REVIEW_THRESHOLD")
DECLINE_THRESHOLD_OVERRIDE = os.getenv("FRAUD_DECLINE_THRESHOLD")
DEFAULT_REVIEW_THRESHOLD = 0.3
DEFAULT_DECLINE_THRESHOLD = 0.7
