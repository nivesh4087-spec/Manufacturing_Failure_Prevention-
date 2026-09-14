# How to work this system

A practical guide to running, using, and extending the platform. Read section 1
to get it running; the rest you can come back to.

**Contents**

1. [Get it running](#1-get-it-running)
2. [The mental model](#2-the-mental-model)
3. [Using the dashboard](#3-using-the-dashboard)
4. [Reading the outputs](#4-reading-the-outputs)
5. [Bringing your own data](#5-bringing-your-own-data)
6. [Training](#6-training)
7. [Configuration](#7-configuration)
8. [Tests](#8-tests)
9. [Generating reports and figures](#9-generating-reports-and-figures)
10. [Extending it](#10-extending-it)
11. [When something breaks](#11-when-something-breaks)

---

## 1. Get it running

### Prerequisites

Python 3.10 or newer. Check with `python --version`.

### Install

```bash
git clone <repository-url>
cd Manufacturing_Failure_Prevention-

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS or Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
streamlit run app/main.py
```

The dashboard opens at `http://localhost:8501`.

**On a fresh clone there is no trained model yet.** The app will say so and offer
you a button — *Train baseline model* — that trains one in about thirty seconds
and reloads. Click it. You do not need to run anything in a terminal first.

If you would rather train from the command line:

```bash
python scripts/train_pipeline.py --mode baseline
```

### The three training profiles

| Command | Time | Use it when |
|:--|:--|:--|
| `--mode baseline` | ~30 sec | First run, or you just changed code and want to check nothing broke |
| `--mode fast` | ~5-10 min | You changed the features or the pipeline and want a real comparison |
| `--mode full` | 10-20 min | Producing figures or numbers anyone else will see |

Add `--skip-plots` or `--skip-shap` to any of them to cut the slowest steps.

---

## 2. The mental model

The plant has two production lines, and the whole system exists because of the
link between them:

```
  LINE 1  CNC stamping and milling          LINE 2  Finishing and inspection
  ┌──────────────────────────────┐          ┌──────────────────────────────┐
  │ spindle, press, tooling      │  tiles   │ optical inspection cell      │
  │                              │ ───────► │                              │
  │ sensors: air temperature,    │          │ rejects: edge chipping,      │
  │ process temperature, speed,  │          │ water staining, warp,        │
  │ torque, accumulated wear     │          │ grid misalignment            │
  └───────────────┬──────────────┘          └──────────────┬───────────────┘
                  │                                        │
                  └────────────────────────────────────────┘
       worn tooling upstream becomes a rejected tile downstream
```

Line 2 catches bad tiles after the material and machine time are spent. The
defect was actually decided upstream, by the state of the tooling. So the system
predicts **machine condition** in order to prevent **product defects**.

### What happens to a reading

```
  raw sensor readings
         │
         ▼
  validation ......... schema, plausible ranges, class balance
         │
         ▼
  feature engineering  5 derived values (thermal margin, power, strain, ...)
         │
         ▼
  scaling ............ using the scaler fitted during training
         │
         ▼
  calibrated model ... failure probability, 0 to 1
         │
         ├──────────────► SHAP ........... which readings drove it
         │
         ▼
  risk score 0-100 ──► risk band ──► ranked maintenance actions
```

Each stage lives in its own module under `src/`, and each is separately tested.

---

## 3. Using the dashboard

Eight pages, in the order you would normally use them.

### Asset Health — start here

The fleet at a glance. Five tiles across the top: how many assets, how many are
critical, how many are above routine, overall fleet health, and what share of
known failures the model catches.

Below that, the operating envelope — three views of where risk sits in the
sensor space — and a watchlist of the highest-risk assets, which you can export
as CSV.

### Risk Assessment — one machine

Enter the current readings for a single machine and get a probability, a risk
score, and the reasoning.

- The three **scenario buttons** load representative operating states. Use them
  to see the system respond without typing numbers.
- The **thermal margin** readout under the inputs updates live — it is the most
  diagnostic single value, so it is shown before you even run the assessment.
- After assessing you get the risk gauge, the SHAP breakdown, and a ranked list
  of actions tied to specific stations on the line.

Values persist when you change something else, so you can load a scenario, nudge
one reading, and re-assess.

### Data Explorer — the underlying record

Distributions split by outcome, a correlation matrix, a pairwise scatter, and a
filterable table. Useful for sanity-checking data before trusting a model on it.

### Model Diagnostics — why the model behaves as it does

- **Across the fleet**: which readings move predictions in general.
- **One asset**: step through real failing and passing assets and see each
  breakdown, with the model's call against the recorded ground truth.
- **Station mapping**: what each feature corresponds to physically.

### Defect Inspection — the quality side

The simulated inspection cell, the defect catalogue with the upstream cause of
each defect, and — most importantly — the **measured** relationship between tool
wear and failure rate, from which the recommended replacement interval is
derived.

### Model & Cost Analysis — which model, and what it is worth

Six tabs: leaderboard, ROC and PR curves, confusion matrix, calibration,
feature-engineering impact, and cost.

The **Cost** tab is the one to show a manager. Put your plant's real figures into
the five inputs and it prices every confusion-matrix cell. The threshold sweep at
the bottom finds the alert threshold that minimises total cost, which is usually
not 0.5.

### Batch Analysis — many machines at once

Upload a CSV or Excel file, connect a database, or load a sample. Column headers
are matched automatically; anything unmatched you map by hand. Then score
everything at once and export the assets that need attention.

### Alerts — the session log

Everything assessed this session, with the high and critical items raised to the
top. Resets when the server restarts — it is a working log, not an audit trail.

---

## 4. Reading the outputs

### Risk score

A 0-100 transform of the calibrated failure probability.

| Score | Band | What to do |
|:--|:--|:--|
| 0-30 | Routine | Nothing; normal schedule |
| 30-60 | Moderate | Check at the next planned stop |
| 60-80 | High | Intervene this shift |
| 80-100 | Critical | Stop and inspect |

The transform is a deliberate design choice, not a physical quantity — it spreads
moderate probabilities across a readable range. The band edges are in
`config/config.yaml` under `risk.thresholds`.

### Failure probability

The calibrated model output. **Calibrated** means it can be read as a real
probability: assets scored at 80% fail roughly 80% of the time. This is what
makes the cost arithmetic meaningful.

### SHAP values

One number per feature per prediction.

- **Positive** — this reading pushed the prediction toward failure.
- **Negative** — it pushed away.
- **Magnitude** — how hard.

They sum to the difference between this prediction and the average one. SHAP
explains what drove the model, which is not proof of physical causation.

### Recommendations

Generated by matching the top risk-increasing features against rules in
`src/recommendations/engine.py`, ordered by priority. Each names the station and
the physical check to perform.

---

## 5. Bringing your own data

### Required columns

| Reading | Unit | Typical range |
|:--|:--|:--|
| Air temperature | K | 295-305 |
| Process temperature | K | 305-315 |
| Rotational speed | rpm | 1150-2900 |
| Torque | Nm | 3-77 |
| Tool wear | min | 0-260 |

Product grade (L/M/H) is optional and defaults to the middle tier.

### Header matching

Headers are matched case-insensitively against a list of known spellings, so
`Air temperature [K]`, `air_temp`, `AirTemp` and `ambient_temperature` all
resolve to the same feature. Anything unmatched is offered as a manual dropdown.

Download a correctly shaped template from the **Batch Analysis** page.

### Databases

The sidebar and the Batch Analysis page both accept a SQLAlchemy connection URI:

```
postgresql://user:password@host:5432/plant
mysql+pymysql://user:password@host:3306/telemetry
sqlite:///local/plant.db
```

Install the matching driver (`psycopg2-binary`, `pymysql`, and so on). Once
connected, the imported table replaces the bundled telemetry across every page.

---

## 6. Training

```bash
python scripts/train_pipeline.py --mode full
```

What it does, in order: load and validate data → engineer features → split →
train each enabled model with randomised hyperparameter search → select on F1 →
calibrate probabilities → evaluate on the held-out test split → compute SHAP →
run the feature-engineering ablation → save artifacts and figures.

### What it writes

| Path | Contents |
|:--|:--|
| `models/*.joblib` | Models, scaler, feature metadata, test results |
| `reports/figures/*.png` | ROC, PR, confusion, calibration, SHAP plots |
| `reports/results/*.json` | Metrics, calibration, ablation, SHAP importance |

`models/` and `reports/results/` are gitignored — they are build output, and
regenerating them is one command.

### Optional dependencies

XGBoost and SHAP are optional. If either is missing the pipeline logs a warning,
skips that part, and continues with the scikit-learn models. Install them for the
full comparison:

```bash
pip install xgboost shap
```

---

## 7. Configuration

Almost everything lives in `config/config.yaml`. The blocks worth knowing:

| Block | Controls |
|:--|:--|
| `data` | Dataset location, target column, columns excluded as leakage |
| `preprocessing` | Scaling method, product-grade encoding |
| `features.engineered` | The derived features and their formulas |
| `models` | Which model families are enabled, and their search grids |
| `hpo` | Search iterations, cross-validation folds, scoring metric |
| `calibration` | Isotonic or sigmoid, and folds |
| `risk` | Score transform exponent, band thresholds, band colours |
| `business` | Downtime cost, recovery time, intervention costs |
| `industry_mapping` | Feature → the station an engineer would walk to |
| `inspection` | Camera setup, product list, defect catalogue |
| `demo_scenarios` | The three scenario buttons |

**Two things to change first when adapting this to a real plant:**

1. `business` — the cost figures drive every currency number in the system.
2. `industry_mapping` — so recommendations name your stations, not ours.

The dashboard theme is in `.streamlit/config.toml` and
`app/components/styles.py`. Change both together; the second one says so.

---

## 8. Tests

```bash
python -m pytest tests/ -q            # all
python -m pytest tests/ -v            # verbose
python -m pytest tests/test_risk.py   # one file
python -m pytest tests/ --cov=src     # with coverage
```

| File | Covers |
|:--|:--|
| `test_data_loader.py` | Loading, config parsing |
| `test_preprocessing.py` | Splitting, scaling, encoding |
| `test_features.py` | Engineered feature formulas |
| `test_risk.py` | Risk scoring and banding |
| `test_prediction.py` | End-to-end single prediction |
| `test_recommendations.py` | Rule matching and priority |
| `test_inference.py` | Batch scoring path |
| `test_tool_wear_profile.py` | Tool-wear interval derivation |
| `test_failure_cases.py` | Malformed input handling |
| `test_train_modes.py` | Training profile presets |
| `test_pages_smoke.py` | Every dashboard page renders without error |

---

## 9. Generating reports and figures

```bash
python scripts/generate_project_report.py    # FALSE_CEILING_PROJECT_REPORT.md
python scripts/generate_project_figures.py   # reports/figures/*.png
```

Both read the trained artifacts and compute their numbers. If no model exists
they stop with an error rather than filling in a figure — that is deliberate. A
report should never contain a number nobody measured.

---

## 10. Extending it

### Add a feature

1. Add the formula under `features.engineered` in `config/config.yaml`.
2. Implement it in `src/features/engineer.py`.
3. Add its physical meaning to `industry_mapping`.
4. Add a test in `tests/test_features.py`.
5. Retrain.

### Add a model

1. Add a block under `models` in `config/config.yaml` with a search grid.
2. Register it in `get_model_registry()` in `src/models/trainer.py`.
3. Retrain. It appears in the comparison automatically.

### Add a dashboard page

1. Create `app/pages/your_page.py` with a `render_page(...)` function.
2. Add it to `PAGES` in `app/main.py` and to the router below it.
3. Use the shared components from `app/components/styles.py` — `panel()`,
   `render_stat_tile()`, `plotly_layout()` — so it matches everything else.

### Add a recommendation rule

Add an entry to `RECOMMENDATION_RULES` in `src/recommendations/engine.py` with
its feature patterns, priority, action list, and plant context.

---

## 11. When something breaks

### "Model artifacts not found"

No model has been trained. Use the button on the cold-start screen, or run
`python scripts/train_pipeline.py --mode baseline`.

### "No module named 'xgboost'" or "'shap'"

Both are optional; the platform runs without them. To enable them:
`pip install xgboost shap`.

### "No module named 'statsmodels'"

`pip install -r requirements.txt` — it is declared there.

### Dataset not found

Put `ai4i2020.csv` in `data/raw/` or the project root. The loader also tries to
download it automatically, which needs an internet connection.

### The app shows raw HTML instead of the styled interface

Usually more than one Streamlit server running at once, competing over
`__pycache__`. Stop them all, delete `__pycache__` directories, and start one:

```bash
find . -name __pycache__ -type d -exec rm -rf {} +
streamlit run app/main.py
```

### A page is slow

Expensive work is cached in `app/components/data_access.py`. If you changed
preprocessing or retrained, clear the cache from the Streamlit menu (**Clear
cache**) or restart the server.

### Predictions look wrong after changing config

Feature engineering and scaling statistics are saved with the model. Changing
`config.yaml` without retraining leaves them inconsistent. Retrain after any
change to `features`, `preprocessing`, or `data`.

---

## Project layout

```
Manufacturing_Failure_Prevention-/
├── app/
│   ├── main.py                  entry point, sidebar, router, cold start
│   ├── components/
│   │   ├── styles.py            design tokens, components, Plotly theme
│   │   └── data_access.py       cached expensive operations
│   └── pages/                   one module per dashboard page
├── src/
│   ├── data/                    loading and validation
│   ├── preprocessing/           splitting, scaling, encoding
│   ├── features/                engineered features
│   ├── models/                  training, baseline, persistence
│   ├── inference/               batch scoring
│   ├── evaluation/              metrics and plots
│   ├── explainability/          SHAP engine
│   ├── risk/                    scoring, banding, history
│   ├── recommendations/         rule engine
│   └── reporting/               report metric collection
├── scripts/                     training, report and figure generation
├── config/config.yaml           central configuration
├── tests/                       test suite
└── reports/                     generated figures and results
```
