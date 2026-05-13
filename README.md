# Restaurant Survival Analysis
### Spring 2026 Data Challenge : UW-Madison Statistics Club

> **Result: 1st place, most accurate model - 94% accuracy (2nd place: 74%)**

---

## Overview

A data-driven consulting analysis using the Yelp Open Dataset to answer two questions:

1. **Which restaurant attributes** are most associated with business survival?
2. **Can we predict** whether a restaurant is currently open or permanently closed?

---

## Key Findings

### Predicting Survival
Business survival is overwhelmingly predicted by **recency of customer activity** — not star ratings or review counts.

| Feature | Correlation with is_open |
|---|---|
| Days since last review | −0.730 |
| Days since last checkin | −0.686 |
| Days since last tip | −0.470 |
| Review count | +0.128 |
| Star rating | +0.017 |

Star ratings had near-zero correlation with survival — a restaurant's historical quality tells us almost nothing about whether it's still open. What matters is whether people are still visiting and talking about it.

### Attribute Recommendations
Using class-stratified prevalence rates (controlling for 80/20 class imbalance):

**Invest in:**
| Attribute | Open Rate | Closed Rate | Difference |
|---|---|---|---|
| Drive Thru | 67.3% | 26.2% | +41.1% |
| Restaurants Delivery | 72.9% | 37.0% | +35.8% |
| Has TV | 82.0% | 64.4% | +17.6% |
| Good For Kids | 86.1% | 80.4% | +5.6% |

**Avoid:**
| Attribute | Open Rate | Closed Rate | Difference |
|---|---|---|---|
| Happy Hour | 60.7% | 78.8% | −18.1% |
| Reservations | 29.1% | 40.4% | −11.4% |

Formal sit-down dining models carry measurably higher closure risk than accessible, convenience-oriented formats.

---

## Model Performance

Three models were trained to validate findings at different levels:

| Model | Features | ROC-AUC |
|---|---|---|
| Full model | All 55 features | 0.9809 |
| Activity-stripped | No recency proxies | 0.9100 |
| Attributes-only | Attributes + location | 0.8300 |

**5-fold cross-validation: 0.9820 ± 0.0005** — confirming results are not attributable to a favorable split.

The attributes-only model (0.83 AUC) validates Part 1a recommendations independently of activity signals — all four recommended attributes appear in the top SHAP features even when the model cannot see any temporal data.

---

## Methodology

### Data Processing
- Filtered 150,346 raw businesses to **56,293 food-related restaurants** using category keywords
- Merged all 5 Yelp dataset files into one feature table per restaurant
- Engineered 55 features across four dimensions: activity recency, activity velocity, review quality, and business attributes

### Leakage Detection
`avg_reviewer_experience` showed −0.496 raw correlation with survival — but a partial correlation test (residualizing against `days_since_last_review`) dropped it to 0.037. Confirmed leakage: the feature encoded *when* businesses closed, not reviewer quality. Excluded from the model.

### Feature Engineering
Key engineered features:
- **Recency:** days since last review, checkin, tip
- **Velocity:** reviews/month, checkins/month, tips/month
- **Quality:** average stars, std of stars, useful/funny/cool votes
- **Attributes:** 15 boolean attributes + categorical encoding of Alcohol, WiFi, NoiseLevel, RestaurantsAttire

### Model
- **XGBoost** (n_estimators=300, learning_rate=0.05, max_depth=6)
- 80/20 stratified train-test split preserving class imbalance
- NaN values left as-is — XGBoost learns optimal missing value treatment natively
- SHAP (SHapley Additive exPlanations) for feature importance

### Attribute Analysis
Rather than Pearson correlation (inappropriate for binary variables with class imbalance), attribute prevalence was computed separately within open and closed restaurants:

```
open_rate  = # open restaurants with attribute / # total open restaurants
closed_rate = # closed restaurants with attribute / # total closed restaurants
difference  = open_rate - closed_rate
```

This gives P(attribute | open) vs P(attribute | closed) — directly answering the business question while controlling for the 80/20 class imbalance.

---

## Repository Structure

```
project/
│
├── data/                          # Raw Yelp parquet files (not included)
│   ├── business_y_removed.parquet
│   ├── review-001.parquet
│   ├── checkin.parquet
│   ├── tip.parquet
│   └── user-002.parquet
│
├── notebooks/
│   └── analysis.ipynb             # Full analysis notebook
│
├── outputs/
│   ├── holdout_predictions.csv    # Predictions on held-out test set
│   ├── dashboard.html             # Interactive restaurant dashboard
│   ├── correlation_chart.png      # Feature correlation visualization
│   └── leakage_detection.png      # Leakage detection visualization
│
├── report.pdf                     # Executive report (2 pages)
├── presentation.pptx              # Presentation slides
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Dependencies

Key libraries:
- `pandas`, `numpy` — data processing
- `xgboost` — gradient boosted classifier
- `scikit-learn` — train/test split, cross-validation, metrics
- `shap` — feature importance via Shapley values
- `plotly` — interactive dashboard
- `matplotlib`, `seaborn` — static visualizations
- `pyarrow` — parquet file reading

Full list in `requirements.txt`.

---

## Limitations

- **Recency tautology** — closed businesses stop receiving reviews by definition, so recency features are partly proxies for closure rather than purely predictive signals
- **New restaurant gap** — for restaurants with no activity history, the attributes-only model (0.83 AUC) is more relevant than the full model
- **Geographic scope** — Yelp data concentrates in specific metropolitan areas; results may not generalize universally
- **Unparsed hours** — operating hours column was not used; features like weekend availability or late-night service could add meaningful signal

---

## Dataset

Yelp Open Dataset — [https://www.yelp.com/dataset](https://github.com/uw-madison-statclub/Sp_2026_Data_Challenge?tab=readme-ov-file)

---
