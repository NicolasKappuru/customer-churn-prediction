# Workshop 2 — KKBox Churn Prediction with Sequence Models

Churn prediction for KKBox (music streaming) using daily sequences instead of aggregated tabular features. Each user is represented by the 90 days before the cutoff, with 14 channels per day: 8 listening channels (song-completion counts, unique songs, total seconds, active flag) and 6 transaction channels (count, amount paid, plan days, state known, days since last transaction, days to expiry). The target is the same as in Workshop 1: the user does not renew within 30 days after expiration.

All reported experiments use the **manual renewal segment** (`state='manual'`), following the Workshop 1 decision. Development is done on the February cohort (cutoff 2017-01-31, split `fit / stop / tune` = 75 / 12.5 / 12.5) and the test set is the March cohort (cutoff 2017-02-28). Three approaches are compared, each with LSTM and GRU:

1. **Calendar**: the 90 calendar days before the cutoff.
2. **Cycle**: the sequence is aligned to each user's subscription expiry (`k = 7` days before expiry).
3. **Trajectory**: the model forecasts the next days of listening and transactions, and churn is read from the forecast trajectory.

A first run on the full cohort (all renewal states) is kept as a reference.

---

## Structure

```
w2/
│
├── notebooks/
│   ├── w2_complete_data/           # Full-cohort reference run
│   │   └── w2_neural_models.ipynb                 # LSTM/GRU on all users
│   │
│   ├── w2_calendar/                # Approach 1: calendar-aligned sequences
│   │   ├── w2_sequence_preprocessing.ipynb        # Raw parquet -> export-sequential-v6-manual
│   │   └── w2_neural_models_manual.ipynb          # LSTM/GRU: Optuna tuning, training, test
│   │
│   ├── w2_cycle/                   # Approach 2: expiry-aligned sequences
│   │   ├── w2_cycle_preprocessing.ipynb           # Raw parquet -> export-sequential-v6-cycle-k7-manual
│   │   └── w2_neural_models_cycle.ipynb           # LSTM/GRU on cycle sequences, compared with calendar
│   │
│   └── w2_trajectory/              # Approach 3: trajectory forecasting
│       └── w2_trajectory_models.ipynb             # Forecaster, event curves, churn from trajectories
│
├── artifacts/
│   ├── sherlock_w2_v6_manual/              # Approach 1 (reported)
│   ├── sherlock_w2_v6_manual_cycle_k7/     # Approach 2 (reported)
│   ├── sherlock_w2_v6_manual_traj/         # Approach 3 (reported): forecasts, event curves, calibration
│   └── sherlock_w2_v6_manual_traj_to_churn/  # Churn read from trajectories (test A/B, bootstrap)
│
├── src/
│   ├── data_processing/            # Members, transactions, user logs -> daily sequences
│   │   └── sequences/              # Channel builders, calendar/cycle export, causal checks
│   └── models/
│       ├── constants.py            # Run selection (W2_ALIGNMENT, W2_CYCLE_K), channels, split
│       ├── feature_preparation/    # Splits, scaling, prepared tensors
│       ├── recurrent_models/       # LSTM/GRU classifier: tuning, training, test
│       ├── trajectory_models/      # Sequence forecaster: loss, decoding, tuning, training, evaluation
│       └── evaluation/             # Metrics, confusion matrices, ROC/PR, gain/lift, bootstrap CI, DeLong
│
├── data/parquet/                   # Raw KKBox tables as parquet (not versioned)
├── export-sequential-v6-manual/            # Calendar sequences (seq_lstm_*.npy, labels_*.parquet)
└── export-sequential-v6-cycle-k7-manual/   # Cycle sequences (+ cycle_index_*.parquet)
```


**Running:** paths in `src` are relative to `workshops/w2`; the first cell of every notebook switches the working directory there. The calendar/cycle choice is made with the environment variables `W2_ALIGNMENT` (`calendar` | `cycle`) and `W2_CYCLE_K` before the first `import src.models` (restart the kernel after changing them). Run the calendar notebooks first: the cycle and trajectory notebooks reuse its split and scalers.
