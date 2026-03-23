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
- **Customer Profile Features**: Account age, credit score, fraud history (daily refresh FeatureView)
- **Aggregated Features**: Transaction statistics via Snowpark DataFrame — AVG, STDDEV, MAX, MIN, counts, ratios (1-minute refresh FeatureView, auto-managed Dynamic Table)
- **Derived Features**: Amount ratios, risk score — computed inline at training time

### Model Performance
- Handles 50:1 class imbalance with `scale_pos_weight`
- BayesOpt HPO via Tuner API (10 trials)
- Tracked metrics: Precision, Recall, F1, AUC-ROC, AUC-PR

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
│   └── ml_dag.ipynb
├── source/
│   ├── train.py               # XGBoost training script (runs as ML Job on SPCS)
│   └── utils.py               # Synthetic data generation utilities
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
| `FRAUD_TRAINING_DATA` | Final training dataset (30 features) |
| `FRAUD_TEST_DATA` | Holdout test set from training job |
| `FRAUD_MONITORING_BASELINE` | Training distribution snapshot for drift detection |
| `MODEL_CHAMPION_TRACKER` | Current champion model version and AUC |
| `PIPELINE_RUN_LOG` | DAG execution history |

### Dynamic Tables
| Table | Refresh | Managed By |
|-------|---------|------------|
| `AGGREGATED_FEATURES` | 1 minute | Feature Store (auto-created from FeatureView) |
| `FRAUD_INFERENCE_AUTOCAPTURE` | 10 minutes | Model Monitor pipeline |

### Feature Views
- `CUSTOMER_PROFILE_FEATURES` (V1) — online enabled
- `CUSTOMER_AGG_FEATURES` (V1) — online enabled, 1-minute refresh

### Tasks (Champion/Challenger DAG)
| Task | Schedule | Description |
|------|----------|-------------|
| `TASK_TRAIN_CHALLENGER` | Daily at 2 AM | Train new challenger model |
| `TASK_VALIDATE_CHALLENGER` | After train | Compare challenger vs champion AUC |
| `TASK_PROMOTE_IF_BETTER` | After validate | Promote if >1% improvement |

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
```

## Key Concepts Demonstrated

- **Feature Store**: FeatureViews with auto-managed Dynamic Tables (no manual DDL)
- **Snowpark DataFrames**: Server-side feature transformations pushed to warehouse
- **ML Jobs**: Training on SPCS compute pools via `submit_directory()`
- **Tuner API**: Distributed BayesOpt HPO on Ray (Container Runtime)
- **Experiment Tracking**: Run logging with metrics and parameters
- **Model Registry**: Versioning, metrics, SPCS deployment
- **Model Monitor**: PSI, Jensen-Shannon, Wasserstein drift detection
- **Champion/Challenger**: Automated retraining DAG with promotion threshold
- **Class Imbalance**: `scale_pos_weight` for rare event detection
