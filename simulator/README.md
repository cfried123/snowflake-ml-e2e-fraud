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
                                        │ Features aggregated:
                                        │ • AVG_TXN_AMOUNT, STD_TXN_AMOUNT
                                        │ • TXN_COUNT, INTL_TXN_RATIO
                                        │ • CARD_PRESENT_RATIO, etc.
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
│                      │     (AMOUNT_TO_AVG_RATIO, IS_ODD_HOUR, RISK_SCORE)   │
│                      │                                                       │
│                      ├► 3. Call Model API on SPCS (/predict-proba)          │
│                      │     (29 features)                                     │
│                      ▼                                                       │
│               ┌─────────────────┐                                            │
│               │ APPROVE (<30%)  │                                            │
│               │ REVIEW (30-70%) │                                            │
│               │ DECLINE (>70%)  │                                            │
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
| `utils.py` | Legacy shared module with session creation, feature vector building, and model call helpers. Note: `predict.py` and `load_test.py` use their own inline implementations. |

## Model Input Features

The fraud detection model receives **29 features** at prediction time:

### Features from Online Feature Store (Customer History)

| Feature | Source | Description |
|---------|--------|-------------|
| `ACCOUNT_AGE_DAYS` | Profile | How old the account is |
| `CREDIT_SCORE` | Profile | Customer's credit score |
| `AVG_MONTHLY_TRANSACTIONS` | Profile | Typical monthly volume |
| `AVG_TRANSACTION_AMOUNT` | Profile | Typical spend |
| `NUM_CARDS_ON_FILE` | Profile | Cards registered |
| `IS_PREMIUM_MEMBER` | Profile | Premium status |
| `NUM_PREVIOUS_FRAUD_REPORTS` | Profile | Past fraud history |
| `AVG_TXN_AMOUNT` | Aggregates | Mean transaction amount |
| `STD_TXN_AMOUNT` | Aggregates | Std dev of transaction amounts |
| `MAX_TXN_AMOUNT` | Aggregates | Largest transaction |
| `MIN_TXN_AMOUNT` | Aggregates | Smallest transaction |
| `TOTAL_TXN_AMOUNT` | Aggregates | Sum of all transactions |
| `TXN_COUNT` | Aggregates | Number of transactions |
| `AVG_TXN_HOUR` | Aggregates | Typical hour of day |
| `STD_TXN_HOUR` | Aggregates | Variation in transaction times |
| `INTL_TXN_COUNT` | Aggregates | International transaction count |
| `INTL_TXN_RATIO` | Aggregates | % of international transactions |
| `CARD_PRESENT_RATIO` | Aggregates | % of in-person transactions |

### Features from Current Transaction

| Feature | Description |
|---------|-------------|
| `amount` | This transaction's amount |
| `hour` | Hour of day (0-23) |
| `day_of_week` | Day of week (0-6) |
| `is_weekend` | Weekend flag (0/1) |
| `card_present` | Card present flag (0/1) |
| `is_intl` | International flag (0/1) |

### Computed Risk Signals

| Feature | Formula | Meaning |
|---------|---------|---------|
| `amount_ratio` | `amount / (avg_txn + 1)` | How unusual is the amount? |
| `max_ratio` | `amount / (max_txn + 1)` | Does it exceed their max? |
| `exceeds_3x_avg` | `amount > avg_txn * 3` | Big spike flag |
| `odd_hour` | `hour in [0-5]` | Middle of night flag |
| `risk_score` | `intl*2 + no_card*1.5` | Combined risk heuristic |

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
