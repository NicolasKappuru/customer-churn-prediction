# Workshop 1 — KKBox Churn Prediction

Churn prediction for KKBox (music streaming) using tabular features engineered from subscription transactions, member demographics, and listening logs. The target is whether a user does not renew their subscription within 30 days after expiration.

Two experiments are reported: one on the full February cohort (~993K users, 6.4% churn) and one focused on the **manual renewal segment** (`state='manual'`, ~111K users, 31.8% churn). The manual segment was chosen because it concentrates the majority of churners and presents a much more tractable class imbalance (2.1:1 vs 14.6:1). Three models are compared in each experiment: Logistic Regression (M1), Random Forest (M2), and XGBoost with early stopping (M3).

A secondary ablation (`_54f`) re-runs the manual experiment after removing 5 features that become degenerate within the manual segment.

---

## Structure

```
w1/
│
├── notebooks/
│   ├── w1_complete_data/           # Full-dataset experiment
│   │   ├── What You Want Me to Stay_ - KKBox EDA.ipynb   # EDA (distributions, correlations, MI)
│   │   ├── KKBox_Correlation_Churn.ipynb                  # Feature–churn correlation analysis
│   │   ├── w1_data_preprocessing.ipynb                    # Feature engineering pipeline
│   │   └── w1_tabular_models.ipynb                        # M1/M2/M3 on full cohort
│   │
│   └── w1_manual_data/             # Manual-segment experiment
│       ├── 00_why_manual_segment.ipynb                    # Motivation: churn by subscription state
│       ├── KKBox_Correlation_Churn_Manual.ipynb           # EDA restricted to manual users
│       ├── w1_manual_tabular_models.ipynb                 # M1/M2/M3 on manual segment (59 features)
│       └── w1_manual_tabular_models_54f.ipynb             # Ablation: 54 features (5 degenerate removed)
│
├── artifacts/
│   ├── sherlock_w1/                # Full-dataset run (models, metrics, optuna studies)
│   ├── sherlock_w1_manual/         # Manual segment run — 59 features
│   └── sherlock_w1_manual_54f/     # Manual segment run — 54 features (ablation)
│
├── figures/
│   ├── eda/                        # EDA figures for the full dataset
│   └── eda_manual/                 # EDA figures for the manual segment + model outputs
│
├── src/
│   ├── data_processing/            # Feature engineering: members, transactions, user logs
│   └── models/
│       ├── baselines/              # LR, RF, XGBoost implementations
│       ├── evaluation/             # Metrics, confusion matrices, ROC/PR curves, artifacts
│       ├── feature_preparation/    # Splits, preprocessor config, data loading
│       └── tunning/                # Hyperparameter search registry
│
└── export-tabular-v5/              # Preprocessed parquet files (train_feb, test)
```
