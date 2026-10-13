# Fraud Detection ML Pipeline

An end-to-end machine learning pipeline for real-time fraud detection built on Snowflake ML.

## Overview

This project demonstrates a production-ready fraud detection system using:
- **Snowflake Feature Store** for centralized feature management (auto-managed Dynamic Tables)
- **Snowflake Model Registry** for model versioning and deployment
- **Tuner API** for distributed hyperparameter optimization (BayesOpt on Ray)
- **SPCS** for model training (ML Jobs) and real-time inference (model services)
- **Model Monitor** for drift detection and ML observability
- **Task DAG** for automated champion/challenger retraining

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│   Transactions  │────▶│  Feature Store        │────▶│  Model Registry  │
│   (Raw Data)    │     │  (FeatureViews +      │     │  (Versioned      │
│                 │     │   auto-managed DTs)   │     │   XGBoost)       │
└─────────────────┘     └──────────────────────┘     └──────────────────┘
                                 │                           │
                                 ▼                           ▼
                        ┌──────────────────┐     ┌──────────────────┐
                        │  Training Data   │     │  SPCS Inference  │
                        │  (spine join)    │     │  Service         │
                        └──────────────────┘     └──────────────────┘
                                                         │
                                                         ▼
                                                 ┌──────────────────┐
                                                 │  Model Monitor   │
                                                 │  (drift metrics) │
                                                 └──────────────────┘
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_setup.ipynb` | Infrastructure setup, synthetic data generation, Feature Store initialization |
| `02_feature_engineering.ipynb` | Snowpark DataFrame aggregations, Feature View registration, derived features, training dataset |
| `03a_model_train.ipynb` | XGBoost training via ML Jobs (SPCS), experiment tracking |
| `03b_model_tune.ipynb` | Hyperparameter optimization with Tuner API (BayesOpt), model registration |
| `04_model_deployment.ipynb` | SPCS inference service deployment, old service retirement |
| `05_model_monitoring.ipynb` | Model Monitor setup, drift detection, inference autocapture |
| `pipeline/ml_dag.ipynb` | Champion/challenger DAG with automated retraining (daily) |

## Features

### Synthetic Data Generation
- 100,000 transactions with realistic fraud patterns
- 2% fraud rate (class imbalance)
- Fraud patterns: high-value, odd-hours, international, card-not-present

### Feature Engineering
All feature names, aggregation logic and derived features are defined once in `source/utils.py` (`get_feature_columns()` / `FEATURE_COLS`) and shared by training (`source/train.py`), the notebooks, the DAG procedures, the monitor and the simulator.
- **Customer Profile Features**: Account age, credit score, fraud history (daily refresh FeatureView)
- **Aggregated Features** (`D7_*`, 7-day window): AVG, STDDEV, MAX, MIN, sum, count, ratios, plus `HAS_HISTORY` and `D7_MINUTES_SINCE_LAST_TXN`
  - *Training*: `TRANSACTION_PIT_AGG_FEATURES` — **point-in-time** per transaction, using only that customer's transactions in `[ts - 7d, ts)` (no future or same-transaction leakage)
  - *Serving*: `CUSTOMER_AGG_FEATURES` — current 7-day window per customer, online-enabled, 1-minute refresh
  - No history → `HAS_HISTORY = 0`, zero counts/sums, NULL means/std/ratios (no silent zero-fill)
- **Derived Features**: Amount-to-average/max ratios, high-value and odd-hour flags, minutes since last transaction
- **Baseline only**: `RISK_SCORE` is a hand-built heuristic that partially encodes the label rules, so it is **excluded from model inputs** and only reported as a comparison baseline

### Model Evaluation
- **Time-based split**: sorted by `TRANSACTION_TIMESTAMP`; oldest 64% train, next 16% validation, most recent 20% test
- **Metrics**: AUC-PR, recall at 1% / 5% FPR, precision / recall / F1 at the chosen threshold, AUC-ROC (reference)
- **Decision thresholds** are selected on validation data and persisted with the model (pickle, registry metrics, `PIPELINE_RUN_LOG`):
  - `decline_threshold` — maximizes F1 on the precision-recall curve (the binary operating threshold)
  - `review_threshold` — highest score that still reaches 90% recall (`REVIEW_TARGET_RECALL`)
  - Bands: `score < review` → APPROVE, `review ≤ score < decline` → REVIEW, `score ≥ decline` → DECLINE. The simulator reads these from the deployed version's registry metrics (fixed 30%/70% only as a fallback)
- **Artifacts per run** (`@ML_MODELS/runs/<RUN_ID>/`): model + thresholds, metrics, `feature_importance.json` (XGBoost gain + mean |SHAP|), also logged to Experiment Tracking
- Handles 50:1 class imbalance with `scale_pos_weight`; BayesOpt HPO via Tuner API (10 trials, optimizes validation AUC-PR)

## Project Structure

```
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03a_model_train.ipynb
│   ├── 03b_model_tune.ipynb
│   ├── 04_model_deployment.ipynb
│   └── 05_model_monitoring.ipynb
├── pipeline/
│   ├── ml_dag.ipynb
│   └── dag_procs.py           # Stored-procedure handlers (train/validate/promote/drift check)
├── simulator/                 # Real-time scoring + load test (see simulator/README.md)
├── source/
│   ├── train.py               # XGBoost training script (runs as ML Job on SPCS)
│   └── utils.py               # Feature contract, feature engineering, synthetic data, evaluation
├── tests/                     # pytest: leakage, injection consistency, feature contract, promotion gate
└── README.md
```

## Snowflake Objects

### Database & Schemas
- `DEMO_FRAUD_DETECTION_DB.ML_PIPELINE` — Pipeline objects
- `DEMO_FRAUD_DETECTION_DB.FEATURE_STORE` — Feature Store

### Tables
| Table | Description |
|-------|-------------|
| `TRANSACTIONS` | Raw synthetic transaction data (100K rows) |
| `CUSTOMER_PROFILES` | Customer attributes (2K customers) |
| `FRAUD_TRAINING_DATA` | Final training dataset (30 model features + IDs, timestamp, `RISK_SCORE` baseline, label) |
| `FRAUD_TEST_DATA` | Holdout predictions from every training run, keyed by `RUN_ID` / `MODEL_VERSION` (appended, never overwritten) |
| `FRAUD_MONITORING_BASELINE` | Training distribution snapshot for drift detection |
| `MODEL_CHAMPION_TRACKER` | Current champion version, AUC-PR, recall@1%FPR, AUC-ROC and decision thresholds |
| `PIPELINE_RUN_LOG` | DAG execution history (train / validate / promote / drift check) |
| `MODEL_DRIFT_LOG` | Per-feature PSI from each drift check |

### Dynamic Tables
| Table | Refresh | Managed By |
|-------|---------|------------|
| `CUSTOMER_AGG_FEATURES$V5` | 1 minute | Feature Store (online serving aggregates) |
| `TRANSACTION_PIT_AGG_FEATURES$V1` | 1 day | Feature Store (point-in-time training aggregates) |
| `FRAUD_INFERENCE_AUTOCAPTURE` | 10 minutes | Model Monitor pipeline |

### Feature Views
- `CUSTOMER_PROFILE_FEATURES` (V1) — online enabled
- `CUSTOMER_AGG_FEATURES` (V5) — online enabled, 1-minute refresh, current 7-day window
- `TRANSACTION_PIT_AGG_FEATURES` (V1) — offline, per-transaction point-in-time aggregates for training

### Tasks (Champion/Challenger DAG)
| Task | Schedule | Description |
|------|----------|-------------|
| `TASK_TRAIN_CHALLENGER` | Daily at 2 AM | Train new challenger model (also scores the champion on the same holdout) |
| `TASK_VALIDATE_CHALLENGER` | After train | Compare challenger vs champion on AUC-PR and recall@1%FPR |
| `TASK_PROMOTE_IF_BETTER` | After validate | Promote if AUC-PR **or** recall@1%FPR improves >1% and neither regresses >1% |
| `TASK_CHECK_DRIFT` | Daily at 2 PM | Model Monitor PSI per feature; PSI > 0.2 → `EXECUTE TASK TASK_TRAIN_CHALLENGER` (12h cooldown) |

#### Promotion criteria
Implemented in `pipeline/dag_procs.py` (`decide_promotion`, unit-tested in `tests/test_dag_procs.py`):
- Baseline is the champion scored **on the challenger's time-based holdout** when its feature set matches; otherwise the stored `MODEL_CHAMPION_TRACKER` metrics
- Promote when AUC-PR or recall at 1% FPR improves by >1% (relative) and neither drops by >1%
- AUC-ROC is logged for reference only — at ~2% fraud it is close to saturated and insensitive to the precision changes that matter

#### Known gap (TODO): post-inference performance
Model Monitor currently tracks drift only. Measuring live precision/recall needs delayed ground-truth labels:
1. Send `TRANSACTION_ID` with each inference request (autocapture currently only stores the feature vector)
2. Join `FRAUD_INFERENCE_AUTOCAPTURE` to confirmed labels (`TRANSACTIONS.IS_FRAUD` / chargebacks) once they mature
3. Set `ACTUAL_CLASS_COLUMNS` on the Model Monitor (or a scheduled task) so performance metrics and the drift gate both feed `TASK_CHECK_DRIFT`

## Getting Started

### Prerequisites
- Snowflake account with ML features enabled
- Warehouse: `ML_WH`
- Compute pool: `ML_TRAINING_POOL` (for ML Jobs and Tuner)
- Required packages: `snowflake-ml-python`, `xgboost`, `scikit-learn`

### Quick Start

1. **Run Setup** — `00_setup.ipynb` creates infrastructure and generates synthetic data
2. **Engineer Features** — `02_feature_engineering.ipynb` creates Feature Views and training dataset
3. **Train Model** — `03a_model_train.ipynb` for baseline, `03b_model_tune.ipynb` for HPO
4. **Deploy** — `04_model_deployment.ipynb` creates SPCS inference service
5. **Monitor** — `05_model_monitoring.ipynb` sets up drift detection
6. **Automate** — `pipeline/ml_dag.ipynb` creates scheduled champion/challenger DAG

### Suspend Pipeline

```sql
ALTER TASK DEMO_FRAUD_DETECTION_DB.ML_PIPELINE.TASK_TRAIN_CHALLENGER SUSPEND;
ALTER TASK DEMO_FRAUD_DETECTION_DB.ML_PIPELINE.TASK_CHECK_DRIFT SUSPEND;
```

### Run Tests

```bash
uv sync --group dev
uv run pytest            # add -m "not slow" to skip the Snowpark local-testing parity test
```

### Upgrading an existing deployment
- Re-run `02_feature_engineering.ipynb` (new Feature Views + `FRAUD_TRAINING_DATA` schema); `train.py` fails fast if required columns are missing
- The first versioned training run renames an existing un-versioned `FRAUD_TEST_DATA` to `FRAUD_TEST_DATA_LEGACY_<ts>`
- Re-run `05_model_monitoring.ipynb` (monitored feature list changed) and `pipeline/ml_dag.ipynb` (procedures now take arguments; tracking tables gain columns automatically)

## Key Concepts Demonstrated

- **Feature Store**: FeatureViews with auto-managed Dynamic Tables (no manual DDL)
- **Snowpark DataFrames**: Server-side feature transformations pushed to warehouse
- **ML Jobs**: Training on SPCS compute pools via `submit_directory()`
- **Tuner API**: Distributed BayesOpt HPO on Ray (Container Runtime)
- **Experiment Tracking**: Run logging with metrics and parameters
- **Model Registry**: Versioning, metrics, SPCS deployment
- **Model Monitor**: PSI, Jensen-Shannon, Wasserstein drift detection, wired to drift-triggered retraining
- **Champion/Challenger**: Automated retraining DAG gated on AUC-PR / recall@1%FPR
- **Class Imbalance**: `scale_pos_weight` for rare event detection
