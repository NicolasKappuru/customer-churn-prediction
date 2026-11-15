# EDA decisions

One section per notebook, written by `write_decisions` in its last cell. This table is the contract for `w1/src/data_processing` and `w2/src/data_processing`.

<!-- nb:00 -->
## 00 · Population, label and coverage

| Finding | Decision | Kind |
|---|---|---|
| Both transaction files contain records after 2017-01-31; customers without one churn 100 % | Every feature uses only `transaction_date <= cutoff` and logs inside the 90-day window (already true in w1/w2 pipelines; keep a test for it) | cleaning |
| EDA population = fit split, identical to the indices used by the w1/w2 models | All later notebooks use `C.load_fit_population()`; stop/tune/March stay unseen | no action |
| 11.6 % have no member record and churn less than the base (4.9 %) | Keep them; encode the absence explicitly (`has_member_record`) instead of imputing average demographics | new feature |
| 0.2 % have no transaction before the cutoff and churn 24-56 % | Add `has_txn`; subscription-state features stay null (not 0) for them | new feature |
| 11.3 % of the cohort does not expire in February (lapsed, paid through March, far future) with very different churn | Candidate categorical `expiry_state` (lapsed / February / March / later / no txn) from the last record; final encoding in notebook 03 | open |
| Last record's expiry differs from max expiry for 1.9 % (cancellations move it back) | Current-state features come from the last record (by date, then source order), never from max() | cleaning |
| No listening in 90 days with a member record churns 7.8 % vs 6.4 % | Keep `has_listening_logs`; activity features analysed in notebooks 06-07 | open |
<!-- /nb:00 -->

<!-- nb:01 -->
## 01 · Members

| Finding | Decision | Kind |
|---|---|---|
| Age has a threshold shape (19 % at <=18, ~7 % from 25 on); mean imputation is the weakest encoding | Replace `bd` mean + `bd_missing` by `age_band` (<=18, 19-21, 22-24, 25-29, 30-44, 45+, unknown) for the linear model; trees get `bd` with NaN for unknown (no imputation) | transformation |
| 403 ages outside 1..100 (excluding 0) | Treat as unknown (NaN / `unknown` band) | cleaning |
| `bd = 0`, `gender = null` and `city = 1` are the same empty-profile event (48 % of fit) | New `profile_state` (complete / incomplete / no member); `city = 1` with an incomplete profile is recoded to `unknown` | new feature |
| Profile state alone adds nothing next to channel; channel x profile does (+0.009 AUC), driven by channel 9 | Linear model gets the `registered_via x profile_state` interaction; trees get both columns | new feature |
| Channel is the strongest member signal (3 % for 7, 18 % for 4); only 5 levels, no negatives | Keep `registered_via` as categorical with `no member` level; drop the dead 'negative -> mode' rule | cleaning |
| Gender value: male 8.9 % vs female 8.6 % | Do not use the gender value; its absence lives in `profile_state` | drop |
| Among complete profiles cities differ little (7-10 %) except city 1; city adds +0.002 AUC | Keep `city` (after the recoding above) and group cities with < 1,000 fit customers into `other`; low priority | transformation |
| Tenure lowers churn inside channels 3/4/9; raw curve is distorted by channel mix | Keep `tenure_days` (trees) and `tenure_band` (linear); interpret only next to channel | transformation |
| 744 registration dates fall after the cutoff (snapshot overwritten), 18 % churn | Treat their tenure as unknown (NaN), never as a band of their own | cleaning |
| Registration month: 5.5-7.4 % once new customers are excluded | Drop `registration_month` and `registration_year` (tenure already covers them) | drop |
<!-- /nb:01 -->

<!-- nb:02 -->
## 02 · Transactions: record quality

| Finding | Decision | Kind |
|---|---|---|
| 2,149 exact duplicate records (0.02 %), all in the first file | Concatenate both files, drop exact duplicates on msno + 8 fields, keep first | cleaning |
| 450k records (Mar-Oct 2015) have plan days = 0 and list price = 0 but were paid ~149 for ~30 days; 31 % of customers affected | Set plan days and list price to null on those records (keep amount paid); exclude them from list-price, plan-length and discount statistics | cleaning |
| 'Paid above list price' is an artifact of the missing list price (39 real cases) | Drop `difference` (mean paid - mean list) and `last_discount` as defined today; use price types instead | drop |
| Price types: full 94 %, missing fields 4 %, promotion 149->119 1.5 %, free (7-day trials, free months) 0.8 % | Record-level `price_type` (full / discounted / free / unknown) as the basis for discount features (notebooks 03-04) | new feature |
| Expiry 1970 (110 records) and non-cancel records expiring before the transaction (2.2k) | Set those expiries to null (invalid); cancellation records may legitimately expire before the transaction date | cleaning |
| Expiry is cumulative (stacked renewals): expiry - transaction date exceeds the plan by > 1 year for 104k records | Drop `duration`, `more_than_30` and `plan_coverage_ratio` as defined today; plan length from `payment_plan_days`, paid-ahead from `days_to_expire` at the cutoff | drop |
| Cancellation = switching off auto-renew (all have is_auto_renew = 1); 62 % move the expiry back | Cancellations are state changes: excluded from payment counts/sums; their expiry replaces the previous one | cleaning |
| `is_auto_renew = 0 & is_cancel = 1` never occurs | Drop the constant `not_autorenew_cancel` | drop |
| File order is not time order within a day; 1.2 % of customers have a mixed cancel + payment last day (21 % churn, independent of order) | Collapse records per day before deriving state: last-day state = payment / cancel / cancel + payment; last expiry = latest expiry of that day | cleaning |
| 38 payment methods; 17 have < 1,000 customers (0.55 % of records) | Group rare methods into `other` before encoding (threshold fixed in notebook 03 on the last method) | transformation |
<!-- /nb:02 -->

<!-- nb:03 -->
## 03 · Transactions: subscription state

| Finding | Decision | Kind |
|---|---|---|
| State of the last transaction day: auto-renew 2.1 %, manual 32 %, cancel 83 %, cancel + payment 21 %, no txn 31 %; AUC 0.814 alone | New categorical `state` (5 levels), computed per day; replaces `last_is_auto_renew` and `last_is_cancel` (file order) | new feature |
| Expiry splits the small states (cancel + payment: 6 % if February, 99 % if lapsed); +0.074 AP | New `expiry_state` (lapsed / february / march / april+ / unknown) plus `state x expiry_state` for the linear model; `days_to_expire` kept only as a numeric input for trees and w2 | new feature |
| Manual payment with expiry in April or later churns 97 % (prepaid long plans in the February cohort) | No special handling; covered by `state x expiry_state`. Mention as a label-definition quirk in the report | no action |
| Non-monthly plans churn 45-73 %; plan length matters only among manual customers | `plan_band` (7 or less / 30-31 / 32-200 / 201+ / unknown) replaces `last_plan_days` | transformation |
| Inside auto-renew, list price orders churn (99: 1.7 %, 180: 3.9 %); free last payment churns 34-61 % | New `last_list_price` level (99/100/129/149/150/180/other) and `last_price_type` (full / discounted / free) | new feature |
| Payment method is ~pure auto-renew or manual, but adds +0.012 AUC within auto-renew | Keep `last_payment_method_id` from the last *payment* record, methods with < 1,000 fit customers grouped into `other` (16 + other) | transformation |
| Days since the last transaction adds ~0 AUC once expiry is known | Keep only as a band (0-7 / 8-31 / 32-90 / 91+), low priority; re-check in notebook 08 | open |
| Proposed state block beats current w1 state features with the same GBM (AP 0.509 vs 0.487, log-loss 0.156 vs 0.160) | Adopt the proposed block in w1; w2 state channels should follow the same day-level rules | transformation |
| `expire_dayofweek` (current w1) has no rationale in any analysis | Drop unless notebook 08 shows a gain | open |
<!-- /nb:03 -->

<!-- nb:04 -->
## 04 · Transactions: history

| Finding | Decision | Kind |
|---|---|---|
| History adds +0.036 AUC / +0.058 AP on top of the state block; 4 features carry almost all of it | Keep `payer_days`, `n_payments` (payments only, no cancellations), `n_lapses`, `total_lapse_days` (payment day after the previous expiry) | new feature |
| Whole history beats a 365-day window (AUC 0.894 vs 0.881) despite truncation at 2015-01-01 (31 % censored) | Use the whole history; state the censoring as a limitation (`payer_days` capped at ~25 months) | no action |
| Inside manual payment more payments / lapses mean less churn (0 lapses 50 %, 4+ lapses 22 %); inside auto-renew history barely matters | Linear model: interact history bands with `state` (manual vs rest); trees find it alone | transformation |
| `total_paid` ~ `n_payments` ~ `payer_days` (rho 0.91-0.93), `max_lapse_days` ~ `total_lapse_days` (0.99), `n_method_changes` ~ `n_methods` (0.96) | Drop `total_paid`, `max_lapse_days`, `n_method_changes` | drop |
| Cancellation counts and price-history counts add ~0 once the state block is known | Drop `n_cancels`, `days_since_last_cancel`, `n_discounted`; `n_free`, `n_methods`, `auto_share` optional (+0.0006 AP), re-check in notebook 08 | open |
| Current w1 history aggregates (19 columns) reach AP 0.560 vs 0.566 for the 4 proposed, and several are broken (2015 zeros, cumulative duration, a constant) | Replace them: drop transaction_count, average_*, sum_*, n_list_prices, n_amounts_paid, n_plans, max_plan_days, is_*_change_flag, autorenew_not_cancel | drop |
<!-- /nb:04 -->

<!-- nb:05 -->
## 05 · Listening logs: quality and scale

| Finding | Decision | Kind |
|---|---|---|
| `user_logs_v2` holds only March 2017; both feature windows (Feb and test) come entirely from `user_logs` | Feature code reads `user_logs` for the windows; `user_logs_v2` is not needed (keep reading both only if harmless) | no action |
| Window is structurally clean: one row per customer-day, no negatives, no non-finite seconds, num_unq <= plays | Keep the negative / non-finite guards as assertions only | no action |
| 7,410 customer-days (0.03 %) exceed 24 h, with counts inflated in proportion (~220 s per play) | On those days rescale all 7 channels by 86,400 / total_secs | cleaning |
| p99.9 per-channel clipping trims real heavy listeners (3.3 % of customers lose > 10 % of their 90-day total) | Drop the fitted p99.9 clipping (`log_clip_p999`) in w1 and any equivalent in w2 | drop |
| All counters are heavily right-skewed; log1p makes num_100, num_unq, total_secs compact | Use log1p for every listening channel (w2 sequences) and for numeric listening aggregates in the linear model | transformation |
| 18.6 % of fit customers have no listening day; churn by activity only ranges 4.9-8.5 % (silent customers 6.2 %) | Keep `has_listening_logs`; do not treat silence as a risk flag on its own; within-state value tested in notebook 06 | open |
<!-- /nb:05 -->

<!-- nb:06 -->
## 06 · Listening logs: customer aggregates

| Finding | Decision | Kind |
|---|---|---|
| Raw listening features are weak (4.5-10 % churn across bins) because 87 % auto-renew regardless of use | Never evaluate listening features alone; always next to `state` | no action |
| Inside manual payment, silence is strong: 27 % (listened on the last day) to 88 % (no logs in 90 days) | Linear model: interact recency bands with `state` (manual vs rest); trees find it alone | transformation |
| Recency (has_logs, days_since_last_log, longest_gap) gives +0.028 AP alone; 5 features match all 33 (AP 0.596) | Keep `has_logs`, `days_since_last_log` (90 if none), `longest_gap`, `active_days_30`, `secs_30` (log1p for linear) | new feature |
| secs / plays / unique songs are the same quantity (rho 0.95-0.98); adjacent windows overlap (0.92-0.94) | One volume measure (`secs_30`) and one window (30 days) | drop |
| Skip, completion and unique-song ratios are flat against churn | Drop the ratio features (`ratio_25`, `ratio_100`, `trend_ratio*` in w1) | drop |
| A sharp rise in listening goes with more churn (new / trial customers ramping up) | Drop `trend_*`; the effect is covered by tenure and free last payment | drop |
| Current w1 monthly blocks rank active months, not calendar windows, hiding recency | Replace the w1 monthly listening blocks (last / second / third month sums, means, logins, login_difference) with the 5 features above | drop |
<!-- /nb:06 -->

<!-- nb:07 -->
## 07 · Daily trajectories (w2)

| Finding | Decision | Kind |
|---|---|---|
| Churners listen less throughout the window (gap present from day 0); no calendar-aligned fade-out | Frame w2 as 'level + recent recency' rather than 'detecting a decline'; report this honestly | no action |
| Within-cycle decay is visible only when aligned to the expiry date (auto-renew churners drift down in the last ~20 days) | Give the LSTM a daily `days_to_expire` channel (already in w2 as `tx_days_to_expire`; keep it, cleaned with the notebook-02/03 rules) | transformation |
| Separation saturates at 14-21 days (auto-renew) and ~45 days (manual); 90 days is slightly worse than 45-60 | Sequence length 60 days is enough; keep 90 only if the model is tuned on it (test 60 vs 90 in w2) | open |
| Weekly trajectory adds nothing over the 5 tabular listening summaries (AP 0.5960 -> 0.5953) | Expect little gain from sequences per se; w2 must be compared against w1 with the same information | no action |
| A model limited to a 90-day sequence + current state reaches AUC 0.875 vs 0.900 with the history block | Add a static branch to the LSTM: state block, history block (payer_days, n_payments, n_lapses, total_lapse_days) and member block, concatenated to the final hidden state | new feature |
| Auto-renew payments land on month ends for everyone; daily cancellations are too sparse (<= 0.06 %/day) | Keep transaction channels as state channels (days to expire, auto-renew, cancel state) rather than relying on daily event counts | transformation |
| Shared calendar dips (New Year, Lunar New Year 27-31 Jan 2017) affect all customers | No action; note that the March test window (Dec-Feb) contains the same holidays at different positions | no action |
<!-- /nb:07 -->

<!-- nb:08 -->
## 08 · Integration, redundancy and drift

| Finding | Decision | Kind |
|---|---|---|
| Ablation: state -0.20 AP, listening -0.024, history -0.012, members -0.0014 when removed | Final set = 4 blocks, 20 features (table in section 7) | new feature |
| dsl_band, expire_dow, city_group, n_free, n_methods, auto_share each move AUC/AP by <= 0.0005 | Drop all open candidates (closes the open items of notebooks 03 and 04) | drop |
| Logistic regression: interactions add +0.012 AP (0.556 -> 0.568); GBM 0.597 | w1 linear baseline uses bands + state x expiry, via x profile, history/recency x manual; GBM is the reference model | transformation |
| 20 proposed features beat the 108 current w1 columns with the same GBM (AUC 0.9001 vs 0.8976, AP 0.597 vs 0.588) | Rewrite w1/src/data_processing to produce the 20-feature table | transformation |
| payer_days PSI 0.85 Feb -> Mar because the 2015-01-01 log start caps it at a cutoff-dependent value | Cap `payer_days` at 730 days (PSI 0.004, no performance loss) | transformation |
| expiry_state PSI 0.12 because fixed day counts meet months of 28 vs 31 days | Replace by `expiry_month` (lapsed / label month / next month / later); PSI 0.001, no performance loss | transformation |
| All other features stable (PSI < 0.04) without using March labels | No drift handling needed beyond the two fixes; keep a PSI check in the pipeline | no action |
| Correlated pairs remain (payer_days-n_payments 0.91, active_days_30-secs_30 0.93, ...) | Keep for trees; if the linear baseline is unstable drop n_payments and secs_30 first | open |
<!-- /nb:08 -->

<!-- nb:10 -->
## 10 · Per-cycle prediction (as-of = expiry - k)

| Finding | Decision | Kind |
|---|---|---|
| Scoring k days before each customer's expiry beats the calendar cutoff from k <= 14: AP +0.04 (k=14), +0.08-0.11 (k=7), +0.14-0.24 (k=1), CIs exclude 0 (Feb holdout and March) | Adopt as_of = expiry - k per customer for w1 and w2 (default k = 7; final k is a business choice); calendar cutoff only for the 11 % out of scope | open |
| k = 28 is slightly worse than the calendar cutoff (-0.025 / -0.006 AP): the calendar lead time is already 1-28 days | Do not use k > 14; report the AP-vs-k curve (figures/10_k_curve.png) as the lead-time menu | no action |
| The gain comes from auto-renew churners who switch off auto-renew between the cutoff and the expiry (recall 1-6 % -> 22-28 % at k=7, precision 0.83-0.90) | w1/w2 data processing take a per-customer as_of (`_common` helpers accept an `as_of` column; tested equal to the global cutoff) | transformation |
| Everyone in scope expires in the label month, so expiry_month is constant | In the per-cycle setting replace expiry_month by numeric days_to_expire at as_of | transformation |
| Eligibility (expiry known at as_of) >= 99.9 % except March k=28 (85 %); no record on or after the expiry is used | Keep the eligibility rule and the leakage assertion in the pipeline | cleaning |
| E3c: six features describing a payment made before the expiry (paid_ahead, days_early, plan days, auto, count, extension) change AP by -0.001..+0.002 at every k; days_to_expire and the last-payment block already carry them | Do not add them; the KKBox early-payment rule (nb 11) is already learned by the 20-feature model when the payment is visible at as_of | drop |
| Early payers extending 26-30 days churn 6 % (Feb) vs 42 % (Mar): composition of month lengths (nb 11 s5); March auto-renew early payers get a 0.07-0.26 % FP rate, like February (diagnostic A) | Keep days_to_expire; no calendar-robust encoding needed | no action |
| March F1 plateau is a threshold problem: the model under-predicts March (7.0-7.8 % churn vs 5.1-5.6 % mean p); with the oracle March threshold F1 rises to 0.612 / 0.629 / 0.641 at k = 7 / 3 / 1, the Feb threshold loses 0.008 / 0.033 / 0.049 | Apply the EM prior correction (Saerens 2002) before thresholding: label-free, +0.004 / +0.009 / +0.011 F1, never worse | transformation |
| EM estimates the March prior at 6.0-6.7 % vs 7.9 % true: the shift is not a pure prior shift (auto-renew early payers 5.4 % vs 2.2 %) | Recalibrate monthly on the latest month with matured labels; in a campaign set the cut-off by contact budget | open |
<!-- /nb:10 -->

<!-- nb:11 -->
## 11 · Label audit

| Finding | Decision | Kind |
|---|---|---|
| is_churn = no payment dated in [expiry, expiry + 30] (99.4 % agreement); a payment made before the expiry does not count as renewal even when it extends the subscription | Keep the official label as target and metric; document the definition in the w1/w2 reports | no action |
| Manual early renewers without cancellation churn 40 % (45 % March); stable across fit/stop/tune; 26 % of manual churners, ~17 % of all in-scope churners | Treat them as a systematic part of the label, not noise; do not drop them from training | no action |
| Inside that group the label is learnable (AUC 0.93 tree / 0.94 GBM on the Feb holdout) from payment timing and plan length | Per-cycle model: add paid_ahead, days_early, plan days of the early payment and payments in the current cycle (all dated <= as_of) | new feature |
| Early payers extending 26-30 days churn 6 % (Feb) vs 42 % (Mar) only because an early monthly auto-renew charge extends +28 days in February and +31 in March; at the exact extension both months match (2 % auto, 46 % manual) | No label mechanic to model; but days_to_expire after an early payment carries month length (k+28 vs k+31): test a calendar-robust encoding in the per-cycle model | open |
| A business label (any payment extending past the expiry) differs for ~1 in 6 churners and needs data to expiry + 30 (unavailable for March) | Report a February-only diagnostic with the business label next to the official metrics | open |
<!-- /nb:11 -->

<!-- nb:12 -->
## 12 · Why the per-cycle model under-predicts March

| Finding | Decision | Kind |
|---|---|---|
| The March shortfall sits in three segments that are calibrated in February: manual payers on plans > 31 d (2.5 % of customers, ~50 % of the deficit, churn 44-47 % -> 82-83 %, mean p 45-47 %), auto-renewers (~30 %, 1.2-1.7 % -> 1.8-2.2 %) and manual 30-day payers (~17 %, 25 % -> 27 %) | Report calibration by these four state x plan-length segments whenever a model is moved to another month | no action |
| Not covariate shift: March features alone predict slightly less churn (covariate effect -2..+2 points per segment); the churn rate given the features rises (+0.6 / +3.5 / +37 points). The only strong feature differences are calendar artefacts (payer_days, n_payments, days_to_expire) that do not move predictions | No feature change and no covariate reweighting | no action |
| The label definition differs between train (Feb) and train_v2 (Mar) while renewal behaviour is identical: a renewal with a plan > 31 d in the 30-day window is 2 % churn in Feb and 97 % in Mar (9,700 March churners); a renewal followed by a cancellation is 48 % vs 96 %. Applying both rules to February moves its churn rate 5.6 % -> 7.4 % (March 7.3 %), every segment on its March rate | For models evaluated on March, rebuild the February training target from transactions with the March rule (official label OR long-plan-only renewal OR renew-then-cancel). Rule found by auditing March labels: needs sign-off, or confirmation of the train_v2 definition from its source | open |
| Trained on the relabelled February target (same features, settings, protocol), March F1 at the February threshold goes 0.592 -> 0.613 (k=7) and 0.590 -> 0.674 (k=1), AP 0.618 -> 0.661 and 0.682 -> 0.730; the February threshold is within 0.002 F1 of the best March threshold and every segment is calibrated | No threshold transfer or recalibration step is needed once the label is consistent; the March F1 plateau of notebook 10 was the label mismatch | transformation |
| Label-free corrections fail on the full eligible population: EM prior shift estimates 5.0-5.3 % (below the training rate) and costs -0.009 / -0.005 F1, per-segment EM -0.012 / -0.015. The gain reported in notebook 10 came from the common-customers filter | Drop the EM prior correction and the monthly recalibration item (supersedes the last two rows of section 10) | drop |
| The calendar-cutoff model (whole cohort, notebook 09 setting) has the same mismatch: 6.4 % predicted vs 9.0 % real in March. Rules 1 + 2 give F1 0.584 -> 0.592 and AP 0.601 -> 0.627; also labelling customers prepaid beyond the next month as churn (40 % churn in Feb, 90 % in Mar) gives F1 0.628, AP 0.661. Customers expiring in the next month stay under-predicted (4.3 % vs 9.0 %) | The relabelled target is needed for any model trained on train and evaluated on train_v2, not only per-cycle; the rule for customers whose expiry is outside the label month is only partly identified (rule 3 is exploratory) | open |
<!-- /nb:12 -->

<!-- nb:13 -->
## 13 · Relabelled February target on the w1 (59 features) and EDA (20 features) models

| Finding | Decision | Kind |
|---|---|---|
| Trained on February relabelled with rules 1 + 2 and evaluated on the official March label, both calendar models gain the same: F1 0.584 -> 0.592 (20 features) and 0.578 -> 0.586 (w1 M3v2, 59 features), AP +0.026 in both; customers expiring in the label month become calibrated (7.3 % predicted vs 7.3 % real) | Use the relabelled February target for w1 models evaluated on train_v2; official February label only for February reporting | transformation |
| Adding rule 3 (prepaid beyond the next month = churn) gives most of the gain: F1 0.628 / AP 0.661 with 20 features, F1 0.610 / AP 0.603 with 59 features (d F1 +0.044 and +0.032, bootstrap CIs exclude 0) | Train with rules 1 + 2 + 3; rule 3 stays flagged as exploratory (found on one group, 90 % pure in March) until the train_v2 definition is confirmed | open |
| With rule 3 the 59-feature model learns days_to_expire >= 60 from February; in March the boundary is 62, so 3,960 customers expiring on April 29-30 (5.7 % churn) get p = 1.00 (~3,700 false positives) and AP falls 0.622 -> 0.603. The 20-feature model uses expiry_month and is not affected | Encode the expiry by calendar month (expiry_month) instead of numeric days_to_expire in any calendar-cutoff model trained with rule 3 | transformation |
| The 20-feature model beats the tuned 59-feature model under every training target (F1 0.584 vs 0.578, 0.592 vs 0.586, 0.628 vs 0.610) | Keep the 20-feature set as the w1 feature table (confirms notebook 09) | no action |
| Customers expiring in the next month stay under-predicted in both models (4.3-4.5 % vs 9.0 %) | No rule identified; report this segment separately | open |
<!-- /nb:13 -->

<!-- nb:14 -->
## 14 · Relabelled target with a February-only split (70 / 15 / 15)

| Finding | Decision | Kind |
|---|---|---|
| On a February-only split (train 70 / tune 15 / test 15, March never read) the 20 and 59 features tie: F1 0.562 vs 0.561 and AP 0.601 vs 0.600 on the official label, 0.653 vs 0.653 with rules 1 + 2 + 3 | Choose the feature set on transfer to another month (notebooks 09 and 13), not on a February holdout | no action |
| Relabelled targets are learned better on their own label (F1 0.56 -> 0.63 -> 0.65, AP 0.60 -> 0.68 -> 0.71) and stay calibrated, but partly mechanically: higher base rate, and rule 3 positives are fully determined by expiry_month = later | Never compare targets by their own-label metrics; compare them on a common evaluation label | no action |
| On the official February label rules 1 + 2 cost nothing measurable (d F1 -0.001 / -0.003, CIs include 0); rule 3 costs about 0.02 F1 (-0.019 / -0.021) | Rules 1 + 2 are the default training target (neutral on train, better on train_v2); rule 3 only when the evaluation label is train_v2, still exploratory | open |
<!-- /nb:14 -->
