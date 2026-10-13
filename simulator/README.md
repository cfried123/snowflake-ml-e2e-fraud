# Fraud Detection Simulator

Real-time fraud detection demo using Snowflake Online Feature Store + Model Inference on SPCS.

## How It Works

The simulator consists of scripts that serve different purposes in the ML pipeline:

### The Core Concept

1. **`transactions.py`** generates historical transaction data for customers
2. This data flows into the **Feature Store** via Dynamic Tables
3. **`predict.py`** and **`load_test.py`** score NEW transactions by comparing them against each customer's historical behavior

**Example:** Customer ABC has a history of 50 transactions averaging $85, mostly domestic, card-present, during business hours. When ABC suddenly attempts a $2,500 international transaction at 3am with card-not-present, the fraud detector flags it as anomalous.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE (Batch/Streaming)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  transactions.py                 Dynamic Tables         Feature Store        │
│  ───────────────                                                             │
│           │                                                                  │
│           ▼                        (refresh              Online Feature      │
│    ┌──────────────┐                ~1 minute)            Tables              │
│    │ TRANSACTIONS │  ─────────────────────────────────►  (low-latency       │
│    │    TABLE     │                                       lookups)           │
│    └──────────────┘                                                          │
│                                                                              │
│    Writes raw transactions:                                                  │
│    • Customer ID, Amount, Hour                                               │
│    • Card present, International                                             │
│    • Merchant info                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Features aggregated (7-day window):
                                        │ • D7_AVG_TXN_AMOUNT, D7_STD_TXN_AMOUNT
                                        │ • D7_TXN_COUNT, D7_INTL_TXN_RATIO
                                        │ • D7_CARD_PRESENT_RATIO, HAS_HISTORY, etc.
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE (Real-time)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  predict.py / load_test.py                                                   │
│  ─────────────────────────                                                   │
│                                                                              │
│  New Transaction ──► 1. Raw SQL against Online Feature Tables               │
│  (Customer ABC,      │  (parallel queries via ThreadPoolExecutor)           │
│   $2500, 3am,        │                                                       │
│   international)     ├► 2. Compute derived features in real-time            │
│                      │     (source/utils.py: AMOUNT_TO_AVG_RATIO, ...)      │
│                      │                                                       │
│                      ├► 3. Call Model API on SPCS (/predict-proba)          │
│                      │     (30 features, FEATURE_COLS order)                 │
│                      ▼                                                       │
│               ┌─────────────────┐                                            │
│               │ APPROVE  < rev  │  bands tuned on validation data,           │
│               │ REVIEW  rev-dec │  read from the deployed version's          │
│               │ DECLINE >= dec  │  registry metrics (fallback 30%/70%)       │
│               └─────────────────┘                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Scripts

### 1. Single Prediction (`predict.py`)

**Purpose:** Score a single transaction with detailed output showing feature lookup and model inference.

```bash
export SNOWFLAKE_PAT="your-token"
python simulator/predict.py

# Specific customer
python simulator/predict.py -c CUST_000123
```

| Flag | Description | Default |
|------|-------------|---------|
| `-c, --customer` | Customer ID | Random |
| `--connection` | Snowflake connection name | default |

Transaction parameters (amount, hour, international, card_present) are randomly generated each run, with 30% chance of suspicious patterns.

### 2. Load Test (`load_test.py`)

**Purpose:** Demonstrate the model endpoint handling high-volume transaction scoring.

```bash
export SNOWFLAKE_PAT="your-token"
python simulator/load_test.py -n 100 -c 10
```

| Flag | Description | Default |
|------|-------------|---------|
| `-n` | Total transactions to process | 100 |
| `-c` | Concurrent requests | 5 |
| `--connection` | Snowflake connection name | default |

**Sample Output:**
```
================================================================================
RESULTS: 100/100 successful | 29.2 TPS

  Latency (ms)       Avg      p50      p95      p99
  ────────────────────────────────────────────────────
  Feature Store      412      398      520      610
  Model Inference     89       85      120      145
  Total              501      483      640      755

  Decisions: APPROVE=78 REVIEW=15 DECLINE=7
```

### 3. Transaction Simulator (`transactions.py`)

**Purpose:** Generate synthetic transaction history to populate the Feature Store.

```bash
python simulator/transactions.py -n 100 --delay 0.5

# Higher fraud rate
python simulator/transactions.py -n 50 -f 0.25
```

| Flag | Description | Default |
|------|-------------|---------|
| `-n, --num` | Number of transactions to generate | 50 |
| `-d, --delay` | Seconds between transactions | 0.2 |
| `-f, --fraud-rate` | Probability of suspicious patterns | 0.15 |
| `--connection` | Snowflake connection name | default |

## Supporting Files

| File | Description |
|------|-------------|
| `config.py` | Database, schema, warehouse, model, and table configuration. All values support env var overrides (see below). |
| `inference.py` | Shared by `predict.py` and `load_test.py`: session creation, parameterized online feature lookups, feature-vector building (via `source/utils.py`), service/endpoint resolution, model calls and decision bands. |

## Model Input Features

The model receives the **30 features** in `FEATURE_COLS` (`source/utils.py`), in that order. The simulator builds them with `build_serving_feature_row()`, the same derived-feature code used to build training data, so there is no hand-maintained copy of the feature list here.

| Group | Features | Source |
|-------|----------|--------|
| Transaction | `TRANSACTION_AMOUNT`, `TRANSACTION_HOUR`, `TRANSACTION_DAY_OF_WEEK`, `IS_WEEKEND`, `CARD_PRESENT`, `IS_INTERNATIONAL` | Request |
| Profile | `ACCOUNT_AGE_DAYS`, `CREDIT_SCORE`, `AVG_MONTHLY_TRANSACTIONS`, `AVG_TRANSACTION_AMOUNT`, `NUM_CARDS_ON_FILE`, `IS_PREMIUM_MEMBER`, `NUM_PREVIOUS_FRAUD_REPORTS` | `CUSTOMER_PROFILE_FEATURES$V1$ONLINE` |
| Aggregates (7-day) | `D7_AVG/STD/MAX/MIN/TOTAL_TXN_AMOUNT`, `D7_TXN_COUNT`, `D7_AVG/STD_TXN_HOUR`, `D7_INTL_TXN_COUNT`, `D7_INTL_TXN_RATIO`, `D7_CARD_PRESENT_RATIO`, `HAS_HISTORY` | `CUSTOMER_AGG_FEATURES$V5$ONLINE` |
| Derived | `AMOUNT_TO_AVG_RATIO`, `AMOUNT_TO_MAX_RATIO`, `IS_HIGH_VALUE_TXN`, `IS_ODD_HOUR` (23:00-05:59), `D7_MINUTES_SINCE_LAST_TXN` | Computed at request time |

Customers with no transactions in the window have no online aggregate row: the simulator sends `HAS_HISTORY = 0`, zero counts and JSON `null` for undefined aggregates, exactly as in training. `RISK_SCORE` is not a model input.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SNOWFLAKE_PAT` | Programmatic Access Token for SPCS endpoint | For predict.py, load_test.py |
| `FRAUD_DB` | Override database name (default: `DEMO_FRAUD_DETECTION_DB`) | No |
| `FRAUD_SCHEMA` | Override schema name (default: `ML_PIPELINE`) | No |
| `FRAUD_FS_SCHEMA` | Override Feature Store schema (default: `FEATURE_STORE`) | No |
| `FRAUD_WH` | Override warehouse (default: `ML_WH`) | No |
| `FRAUD_FS_WH` | Override feature serving warehouse (default: `FEATURE_SERVING_WH`) | No |
| `FRAUD_MODEL` | Override model name (default: `FRAUD_DETECTION_MODEL`) | No |
| `FRAUD_AGG_FV_VERSION` / `FRAUD_PROFILE_FV_VERSION` | Online Feature View versions (default: `V5` / `V1`) | No |
| `FRAUD_REVIEW_THRESHOLD` / `FRAUD_DECLINE_THRESHOLD` | Override the decision bands (both required) | No |
| `FRAUD_USE_OFFLINE_FEATURES` | `true` reads the Feature Views' dynamic tables instead of `$ONLINE` tables (trial accounts don't support the hybrid tables behind online features) | No |
