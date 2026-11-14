# Predicting the subscription state: option A vs option B

The trajectory model predicts the next 7 days of each user. The 20 channels fall into two groups:

- **Events** (listening, payments, cancellations): what happens on a given day.
- **States** (`days_since_last`, `days_to_expire`, flags): the situation of the subscription at the end of the day.

States are a deterministic function of the events and the previous state:

- **With a payment:** `days_since_last` goes back to 0, the flags take the values of the payment and the expiry moves.
- **Without a payment:** `days_since_last` goes up by 1 and `days_to_expire` goes down by 1.

There were two ways to obtain the future state.

## Option A: predict the state directly

Add output heads for the 7 state channels and train them like the rest.

**Pros**

- One model and no post-processing.
- Simple to implement: only the size of the output layer changes.

**Cons**

1. **It predicts averages, not possible values.** Take a user with auto-renewal and 1 day of membership left:
   - Renews: `days_to_expire = 30`, with probability p.
   - Does not renew: `days_to_expire = 0`, with probability 1 − p.

   Under squared loss the optimal prediction is the conditional mean:

   ```text
   argmin_c E[(Y − c)²] = E[Y] = 30·p
   ```

   With p = 0.75 the model predicts 22.5 days, a value that no real trajectory takes.
2. **No coherence guarantee.** Nothing forces the predicted state to agree with the predicted events. The network can predict a payment while `days_since_last` keeps growing.
3. **It does not fit the environment.** In `wfinal` the customer is stored as raw transactions and logs, and `SequenceBuilder` recomputes the states from the payments. A predicted state has nowhere to go.
4. **It is hard to beat a trivial rule.** On days without a payment, the arithmetic rule (+1 / −1 per day) is almost always right. Option A can at best tie with it on those days.

### Example (made-up numbers)

A user with auto-renewal has 4 days left on day 82 and actually pays on day 86.

| Day | Real payment | p(pay) | `days_to_expire` real | pred A | pred B | `days_since_last` real | pred A | pred B |
|---|---|---|---|---|---|---|---|---|
| 83 | 0 | 0.02 | 3 | 3.1 | 3 | 27 | 27.0 | 27 |
| 84 | 0 | 0.05 | 2 | 2.0 | 2 | 28 | 28.1 | 28 |
| 85 | 0 | 0.21 | 1 | 1.4 | 1 | 29 | 27.5 | 29 |
| 86 | **1** | **0.78** | 30 | **22.5** | 30 | 0 | **6.2** | 0 |
| 87 | 0 | 0.06 | 29 | 24.0 | 29 | 1 | 7.0 | 1 |
| 88 | 0 | 0.03 | 28 | 23.1 | 28 | 2 | 8.1 | 2 |
| 89 | 0 | 0.02 | 27 | 22.0 | 27 | 3 | 9.0 | 3 |

The bold values of A (22.5 days, 6.2 days) mix "renewed" with "did not renew". B gets every day right here because the sampled payment falls on the real day.

## Option B: predict events and rebuild the state (chosen)

The network predicts only the 13 event targets. To simulate a week:

1. Sample `active`, `pay`, `cancel` and `auto_renew` from their probabilities.
2. Write the predicted amounts on the sampled days.
3. Recompute the 7 state channels with the export's rules.

**Pros**

- **Coherent by construction:** a payment always resets the counters.
- **Only possible values:** each simulated week is one plausible future, not an average.
- **Fits the environment:** simulated events are payments and listening days, exactly what `wfinal` stores.
- **Fewer outputs:** 13 targets instead of 20, so the network spends its capacity on what it cannot compute.

**Cons**

1. **Fragile to timing errors.** If the payment is sampled one day late, the state is off by a lot on those days. Option A would be off by a little on every day. The evaluation therefore also measures payments within a ±1 day / weekly window.
2. **Reconstruction happens after training.** Sampling is not differentiable, so the state never enters the loss.
3. **Approximate expiry.** The export uses the real expiry date. B derives it as `max(days_to_expire − 1, 0) + plan_days`. The `oracle_events` source in the evaluation, which rebuilds the state from the real events, measures how much error this rule alone introduces.

## Why B

The model is used as a **simulator for the agents**. An environment must move forward through valid states, so it needs **one possible future**, not the average of all of them. Option A can look competitive on a mean error. It still produces states that do not exist and that can contradict the events, which breaks the environment.

Option A could still be reported as an ablation (B vs A on state error), but it is not part of the implemented notebook.
