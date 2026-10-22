"""Demo — renders the KKBox visualization with synthetic data.

Run from the visualization/ folder:

    cd visualization
    python demo.py

Controls
--------
Q / W / E       Switch panel tabs  (Profile / Transactions / Logs)
↑ / ↓           Scroll current tab
1 / 2 / 3       Simulate an action  (accept/reject randomised)
R               Reset episode       (new random sprite, churn = 0.72)
ESC             Quit
"""

from __future__ import annotations
import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pygame
from renderer      import KKBoxRenderer
from customer_data import CustomerDataModel, ALL_FEATURES

# ── synthetic 59-feature baseline ────────────────────────────────────────────

_BASE_FEATURES: dict = {
    # ── Members ──
    "city":                       1,
    "registered_via":             9,
    "bd":                      1990,
    "bd_missing":                 0,
    "tenure_days":              670,
    "registration_year":       2015,
    "registration_month":         3,
    "has_member_record":          1,
    # ── Transactions ──
    "is_cancel_change_flag":      1,
    "is_auto_renew_change_flag":  1,
    "transaction_count":         12,
    "average_amount_paid":     144.3,
    "average_amount_charged":  149.0,
    "average_plan":             30.0,
    "n_payment_methods":          1,
    "n_plans":                    1,
    "n_list_prices":              1,
    "n_amounts_paid":             2,
    "duration":                 455,
    "plan_coverage_ratio":      0.92,
    "sum_plan_days":            360,
    "sum_list_price":          1788.0,
    "sum_amount_paid":         1731.6,
    "more_than_30":               0,
    "autorenew_not_cancel":      10,
    "not_autorenew_cancel":       2,
    "max_plan_days":             30,
    "last_payment_method_id":    38,
    "last_plan_days":            30,
    "last_is_auto_renew":         0,
    "last_is_cancel":             1,
    "last_discount":              0.0,
    "days_to_expire":            14,
    "expire_dayofweek":           6,
    "days_since_last_transaction":15,
    "difference":               -15,
    "has_transactions":           1,
    # ── User logs ──
    "sum_last_month_unq":        22,
    "sum_last_month_total_secs": 3840.0,
    "mean_last_month_total_secs":960.0,
    "last_login":          20161201,
    "last_sum_total_songs":       5,
    "last_ratio_100":           0.42,
    "last_ratio_25":            0.12,
    "sum_last_second_month_total_secs": 5120.0,
    "last_second_login":   20161105,
    "last_second_sum_total_songs":      8,
    "last_second_ratio_100":    0.51,
    "last_second_ratio_25":     0.09,
    "last_third_sum_total_songs":      11,
    "last_third_ratio_100":     0.58,
    "last_third_ratio_25":      0.07,
    "login_difference":           7,
    "trend_songs":              -3.0,
    "trend_secs":             -640.0,
    "trend_ratio100":          -0.08,
    "trend_ratio25":            0.02,
    "days_since_last_log":       30,
    "has_listening_logs":         1,
}

_ACTION_COSTS    = [0.0, 0.10, 0.15]
_ACTION_ACCEPT_P = [0.40, 0.70, 0.85]


def _new_model() -> CustomerDataModel:
    return CustomerDataModel(
        features=dict(_BASE_FEATURES),
        churn_probability=0.72,
    )


def _simulate_action(model: CustomerDataModel, action: int) -> None:
    accepted  = random.random() < _ACTION_ACCEPT_P[action]
    reduction = random.uniform(0.15, 0.30) if accepted else 0.0

    new_feat = copy.copy(_BASE_FEATURES)

    if accepted:
        # Simulate feature deltas depending on action
        if action == 0:   # renewal
            new_feat["last_is_auto_renew"]         = 1
            new_feat["last_is_cancel"]             = 0
            new_feat["transaction_count"]         += 1
            new_feat["days_since_last_transaction"] = 0
        elif action == 1:  # annual plan
            new_feat["last_plan_days"]             = 365
            new_feat["more_than_30"]               = 1
            new_feat["last_is_auto_renew"]         = 1
            new_feat["last_is_cancel"]             = 0
            new_feat["transaction_count"]         += 1
            new_feat["sum_last_month_total_secs"] += 2200.0
        elif action == 2:  # discount
            new_feat["last_discount"]              = 0.05
            new_feat["transaction_count"]         += 1
            new_feat["average_amount_paid"]       *= 0.95

    new_churn = max(0.0, model.churn_probability - reduction)
    reward    = reduction * (1.0 - _ACTION_COSTS[action]) if accepted else 0.0

    model.apply_action_result(
        new_features = new_feat,
        new_churn    = new_churn,
        reward       = reward,
        action       = action,
        accepted     = accepted,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    print(__doc__)
    renderer = KKBoxRenderer()
    model    = _new_model()
    renderer.set_sprite(random.randint(0, 23))

    running = True
    while running:
        for event in pygame.event.get():
            if not renderer.handle_event(event):
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_1:
                    _simulate_action(model, 0)
                elif event.key == pygame.K_2:
                    _simulate_action(model, 1)
                elif event.key == pygame.K_3:
                    _simulate_action(model, 2)
                elif event.key == pygame.K_r:
                    model = _new_model()
                    renderer.set_sprite(random.randint(0, 23))
                    # reset scroll
                    renderer._scroll    = {0: 0, 1: 0, 2: 0}
                    renderer._max_scroll= {0: 0, 1: 0, 2: 0}

        renderer.render(model)
        renderer.clock.tick(60)

    renderer.close()


if __name__ == "__main__":
    run()
