"""Preprocessing parameters: fit orchestration and persistence.

The unified ``params`` state is the merge of the per-domain fitted values:
member imputation (``members.fit_member_params``) and listening clipping /
sequence standardization (``user_logs.fit_sequence_params``).
"""

import json
from pathlib import Path

from w1.src.data_processing.members import fit_member_params
from w1.src.data_processing.user_logs import fit_sequence_params


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_preprocess_params(members, user_day, train_users, train_cutoff):
    """Fit the full preprocessing state on the training cohort only.

    Merges the member imputation values with the listening clipping and sequence
    standardization values into a single ``dict``.
    """
    return {
        **fit_member_params(members, train_users),
        **fit_sequence_params(user_day, train_users, train_cutoff),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_preprocess_params(params, path):
    """Write the preprocessing state as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def load_preprocess_params(path):
    """Read a preprocessing state previously written by ``save_preprocess_params``."""
    with open(path) as f:
        return json.load(f)
