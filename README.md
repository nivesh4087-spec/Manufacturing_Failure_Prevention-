<div align="center">

# Manufacturing Failure Prevention Platform

### Predictive maintenance and defect inspection for a suspended-ceiling systems plant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B.svg)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-8b5cf6.svg)](https://shap.readthedocs.io)
[![Tests](https://img.shields.io/badge/tests-161%20passing-0ca30c.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Predicts machine failures before they happen, explains every prediction, and
prices the decision — so a maintenance call is backed by a reason and a number.*

</div>

---

## What this is

A plant making suspended-ceiling systems runs two connected lines. Line 1 stamps
and mills the board; Line 2 inspects the finished tile. When Line 1's tooling
wears past its limit, Line 2 starts rejecting tiles for edge chipping — by which
point the material and machine time are already spent.

This platform predicts the **machine** condition in order to prevent the
**product** defect.

```
  LINE 1  CNC stamping and milling          LINE 2  Finishing and inspection
  ┌──────────────────────────────┐          ┌──────────────────────────────┐
  │ spindle, press, tooling      │  tiles   │ optical inspection cell      │
  │                              │ ───────► │                              │
  │ sensors: air temp, process   │          │ rejects: edge chipping,      │
  │ temp, speed, torque, wear    │          │ staining, warp, misalignment │
  └───────────────┬──────────────┘          └──────────────┬───────────────┘
                  └────────────────────────────────────────┘
       worn tooling upstream becomes a rejected tile downstream
```

**The clearest result we found:** binning the dataset by accumulated tool wear
shows the failure rate sits flat near 2.3% until about **200 minutes**, then
climbs to 13.8% and beyond. Replacing tooling at that point would have avoided
**44 of 46 tool-wear failures (96%)**. That interval is derived from the data at
runtime, not asserted — it moves if the data moves.

---

## Contents

- [Quick start](#quick-start)
- [What it does](#what-it-does)
- [Results](#results)
- [The dashboard](#the-dashboard)
- [How it works](#how-it-works)
- [Training](#training)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Authors](#authors)

---

## Quick start

```bash
git clone <repository-url>
cd Manufacturing_Failure_Prevention-

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/main.py
```

There is no trained model on a fresh clone, and the app says so rather than
crashing: it offers a **Train baseline model** button that produces a working
model in about thirty seconds and reloads. You do not need to run anything else
first.

Prefer the terminal?

```bash
python scripts/train_pipeline.py --mode baseline   # ~30 seconds
python scripts/train_pipeline.py --mode fast       # ~5-10 minutes
python scripts/train_pipeline.py --mode full       # 10-20 minutes
```

**New here?** [`USER_GUIDE.md`](USER_GUIDE.md) walks through the whole system —
running it, reading its output, bringing your own data, and extending it.

---

## What it does

| Capability | How |
|:--|:--|
| Predicts failure from sensor telemetry | Calibrated ensemble, selected on F1 |
| Explains every prediction | SHAP attribution, global and per-asset |
| Turns explanations into instructions | Rule engine mapping features to stations |
| Scores a whole file or database table | Shared batch inference path |
| Prices the decision | Full cost model with adjustable plant figures |
| Finds the cost-optimal alert threshold | Threshold sweep over the test split |
| Links machine health to product quality | Tool-wear interval measured from data |

---

## Results

Measured on a held-out test split. Regenerate with
`python scripts/generate_project_report.py`.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|:--|--:|--:|--:|--:|--:|--:|
| **Random Forest** | 0.9583 | 0.9020 | **0.9293** | 0.9417 | 0.9806 | 0.0081 |
| HistGradientBoosting | 0.8679 | 0.9020 | 0.8846 | 0.9466 | 0.9784 | 0.0055 |

*Figures above come from the `baseline` training profile. The `full` profile
searches hyperparameters and includes XGBoost and Logistic Regression.*

In plain terms: of every 51 genuine failures the model catches **46** and misses
**5**, while raising **2** unnecessary inspections.

**Accuracy is deliberately absent.** With 96.6% of assets running normally, a
model that always predicts "no failure" scores 96.6% accurate and catches
nothing. F1 and PR-AUC stay honest under that imbalance.

### What this does not claim

- These scores describe the **AI4I 2020 public dataset**, not a real plant. The
  pipeline ingests real telemetry and has been exercised end to end, but live
  performance must be re-measured after deployment.
- The risk score is an engineering design choice, not a physical quantity.
- SHAP explains what drove the model. That is not proof of physical causation.
- The split is random, not chronological — the dataset has no reliable time
  ordering. On live data a time-based split would be the honest test.

---

## The dashboard

Eight modules, in the order you would normally use them.

| Page | The question it answers |
|:--|:--|
| **Asset Health** | How is the fleet doing, and what needs attention? |
| **Risk Assessment** | What is this machine's failure risk, and why? |
| **Data Explorer** | What does the underlying sensor record look like? |
| **Model Diagnostics** | What drives the model's decisions? |
| **Defect Inspection** | Is the product within tolerance, and what upstream cause explains a reject? |
| **Model & Cost Analysis** | Which model, and what is it worth in currency? |
| **Batch Analysis** | Score a whole file, table or database query |
| **Alerts** | What was flagged this session, and what was recommended? |

### Reading a risk score

| Score | Band | Action |
|:--|:--|:--|
| 0-30 | Routine | Normal schedule |
| 30-60 | Moderate | Check at the next planned stop |
| 60-80 | High | Intervene this shift |
| 80-100 | Critical | Stop and inspect |

Band edges are configurable in `config/config.yaml`.

---

## How it works

```
  telemetry ──► validation ──► feature engineering ──► calibrated model
   (file,         (schema,        (5 derived            (selected on F1,
    database,      range,          features)             isotonic calibration)
    stream)        balance)                                      │
                                                                 ▼
                                              ┌──────────────────────────────┐
                                              │ probability ──► risk score   │
                                              │      │              │        │
                                              │      ▼              ▼        │
                                              │   SHAP          risk band    │
                                              │      │              │        │
                                              │      └──────┬───────┘        │
                                              │             ▼                │
                                              │   ranked maintenance action  │
                                              └──────────────────────────────┘
```

### Engineered features

| Feature | Formula | What it captures |
|:--|:--|:--|
| Thermal margin | process temp − air temp | Whether heat is leaving the machine |
| Mechanical power | torque × speed × 2π ÷ 60 | Work actually done at the cut |
| Load per unit speed | torque ÷ speed | Whether the drive is straining |
| Accumulated strain | tool wear × torque | Cumulative stress on the tooling |
| Thermal-speed stress | thermal margin × speed | Heat generated at operating speed |

Plus binned wear severity and high-torque / low-speed / overload flags.

None of these adds information — each combines readings the model already has.
They help because a decision tree needs many splits to approximate a ratio or a
product, and stating it directly costs none.

### Models

Logistic Regression (interpretable baseline), Random Forest, XGBoost, and
HistGradientBoosting. XGBoost and SHAP are **optional** — if either is missing
the pipeline logs a warning, skips it, and continues with the rest.

### Class imbalance

The dataset is 3.4% failures. Handled with `class_weight="balanced"`, with SMOTE
available as a configurable alternative applied to training data only.

---

## Training

```bash
python scripts/train_pipeline.py --mode full
```

Load and validate → engineer features → split → hyperparameter search per model
→ select on F1 → calibrate probabilities → evaluate on the held-out split →
compute SHAP → ablation study → save artifacts and figures.

| Flag | Effect |
|:--|:--|
| `--mode baseline` | Fixed hyperparameters, ~30 seconds |
| `--mode fast` | Small search, capped grids and workers |
| `--mode full` | The configured search (default) |
| `--skip-plots` | No figure generation |
| `--skip-shap` | Skip SHAP, which dominates runtime |

Outputs land in `models/`, `reports/figures/` and `reports/results/`, all of
which are gitignored build output — one command regenerates them.

### Reports

```bash
python scripts/generate_project_report.py    # FALSE_CEILING_PROJECT_REPORT.md
python scripts/generate_project_figures.py   # analysis figures
```

Both compute their numbers from the trained artifacts. If no model exists they
stop with an error rather than filling in a figure.

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
│   ├── data/                    loading, database connectors, validation
│   ├── preprocessing/           splitting, scaling, encoding
│   ├── features/                engineered features
│   ├── models/                  training, baseline, persistence
│   ├── inference/               shared batch scoring
│   ├── evaluation/              metrics and evaluation plots
│   ├── explainability/          SHAP engine
│   ├── risk/                    scoring, banding, session history
│   ├── recommendations/         rule engine
│   └── reporting/               report metric collection
├── scripts/                     training, report and figure generation
├── config/config.yaml           central configuration
├── tests/                       161 tests
└── reports/                     generated figures and results
```

---

## Documentation

| Document | For |
|:--|:--|
| [`USER_GUIDE.md`](USER_GUIDE.md) | Running, using and extending the system |
| [`FALSE_CEILING_PROJECT_REPORT.md`](FALSE_CEILING_PROJECT_REPORT.md) | Project status, generated from measured results |
| [`FINAL_PROJECT_AGILE_KANBAN_SUBMISSION.md`](FINAL_PROJECT_AGILE_KANBAN_SUBMISSION.md) | Agile process and sprint tracking |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution workflow |

---

## Tests

```bash
python -m pytest tests/ -q
```

Covering data loading, preprocessing, feature formulas, risk scoring, the
recommendation rules, the batch inference path, the tool-wear interval
derivation, training mode presets, and malformed-input handling — plus headless
smoke tests that render every dashboard page and fail the build if any one of
them cannot open.

---

## Authors

VIT Pune

| Member | Responsibility |
|:--|:--|
| **Nivesh Manoj Jain** | ML architecture and pipeline |
| **Hasan Rupawalla** | Computer vision and inspection |
| **Rachit Ingole** | Telemetry ingestion and data |

Dataset: [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset),
UCI Machine Learning Repository.

---

<div align="center">

**Predict early. Explain the reason. Act before it breaks.**

</div>
