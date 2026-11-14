# EDA series · KKBox churn

Exploratory analysis that **decides** the features of w1 (tabular) and w2 (sequences), rather than describing
features that already exist. It replaces `workshops/w1/eda_legacy/`.

**Rules for every notebook**

- Population: February cohort, `fit` part of the development split only (744,699 customers; identical to the
  rows the w1/w2 models are fitted on). `stop`, `tune` and the March cohort are never inspected with labels.
- As-of: only records dated on or before the cutoff (2017-01-31); listening window 2016-11-03 … 2017-01-31.
- Unit of analysis: the customer. Samples, when needed, are random by customer.
- Each notebook ends with a findings → decisions table, written to `DECISIONS.md` (the specification for
  `w1/src/data_processing` and `w2/src/data_processing`).

| # | Notebook | Question |
|---|---|---|
| 00 | `00_population_label_coverage` | Who is predicted, what does the label mean, which tables know each customer? |
| 01 | `01_members` | Which demographic fields carry signal, and how to encode them (age, profile, channel)? |
| 02 | `02_transactions_quality` | Which transaction records are duplicated, incomplete or ambiguous? |
| 03 | `03_transactions_state` | How much does the current subscription state explain? |
| 04 | `04_transactions_history` | What does the payment history add on top of the state? |
| 05 | `05_user_logs_quality` | What cleaning and scaling do the daily listening counters need? |
| 06 | `06_user_logs_aggregates` | Which listening summaries add signal, over which window? |
| 07 | `07_trajectories` | Is there a temporal dynamic that justifies a sequence model, and how should w2 be set up? |
| 08 | `08_integration_and_drift` | Final 20-feature set, ablation, comparison with current w1, Feb → Mar drift |
| 09 | `09_xgboost_old_vs_new` | Do the 20 EDA features beat the 108 current w1 columns with the same XGBoost? |
| 10 | `10_per_cycle_prediction` | Scoring each customer k days before their own expiry: AP/F1 vs lead time k |
| 11 | `11_label_audit` | What exactly does `is_churn` mean? Early renewers labelled as churn: noise or rule? |

Notebooks 09, 10 and 11 are the only ones that look at `tune` and March with labels (to evaluate, never to choose
features). Notebook 10 needs about 8 GB of RAM; do not run it next to another large kernel.

**Running.** Notebooks read `workshops/data/parquet/` (created by the w1 preprocessing notebook). Shared code
lives in `_common.py`; intermediate results are cached in `eda/cache/` (git-ignored). Run 06 before 07 (07 reads
the listening features cached by 06). Figures go to `eda/figures/`.

`PLAN.md` is the original working plan (Spanish).
