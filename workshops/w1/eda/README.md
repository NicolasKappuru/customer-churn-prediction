# EDA series (w1) · KKBox churn

The exploratory analysis of the KKBox churn data, split into ten short notebooks that are meant to be read in order.
Each notebook opens with the question it answers and closes with its conclusions.

| # | Notebook | Question it answers | Cohort |
|---|---|---|---|
| 01 | `01_target_overview` | How many customers churn? How imbalanced is the target? | March (`train_v2`) |
| 02 | `02_members` | What do age, city, gender and registration channel look like? What needs cleaning? | March |
| 03 | `03_transactions` | How do plans, prices and payment methods behave? | March |
| 04 | `04_user_logs` | What is the shape of the daily listening counters and their outliers? | March |
| 05 | `05_dates` | Do registration, transaction and expiry dates carry information? | March |
| 06 | `06_bivariate_raw` | Which raw variables already separate churners from non-churners? | March |
| 07 | `07_feature_catalogue_and_target` | What are the 59 model features? Target imbalance in the development cohort | February |
| 08 | `08_numeric_and_categorical_vs_churn` | Pearson/Spearman/ROC AUC for numeric features, Cramér's V for nominal ones | February |
| 09 | `09_binary_discrete_continuous_vs_churn` | Churn rate by value (flags, counts) and by decile (continuous features) | February |
| 10 | `10_redundancy_and_feature_ranking` | Redundant feature pairs and the unified mutual-information ranking | February |

**Part I (01-06)** works on the raw tables before feature engineering; **Part II (07-10)** works on the 59 engineered
features that the models in `../w1_tabular_models.ipynb` consume.

## Running them

1. Place the nine `.csv.7z` archives in `workshops/data/` (see the Workshop 1 report).
2. Run `01_target_overview` first: it converts the archives to `workshops/data/parquet/`.
3. Notebooks 07-10 read `workshops/w1/export-tabular-v5/train_feb.parquet`, produced by `../w1_data_preprocessing.ipynb`.
4. The notebooks can run from `workshops/w1` or from `workshops/w1/eda`; paths are resolved from the `w1` folder.

Notebooks 05 and 06 include a short "setup carried over" block that reloads the tables they need, so each notebook is
self-contained.

## Figures

Notebooks 07-10 write their plots to `workshops/w1/figures/eda/` (target distribution, numeric correlations,
categorical, binary, discrete and continuous relationships with churn, feature correlation matrix, mutual
information, and `feature_churn_ranking.csv`). The Workshop 1 report uses these files.

## Relation to the earlier notebooks

These notebooks were produced by splitting `../What You Want Me to Stay_ - KKBox EDA.ipynb` (01-06) and
`../KKBox_Correlation_Churn.ipynb` (07-10). Code and saved outputs are unchanged apart from the path handling, the
introductions and the conclusions. The two original notebooks are kept as they were.
