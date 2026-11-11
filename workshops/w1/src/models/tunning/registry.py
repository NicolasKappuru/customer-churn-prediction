from dataclasses import dataclass

import numpy as np

from ..constants import EPS


class Registry:
    """Collect model records and clipped probability arrays."""

    def __init__(self, eps=EPS):
        self.eps = eps
        self.records: dict[str, dict] = {}
        self.proba: dict[str, dict[str, np.ndarray]] = {}

    def register(self, model_id, **record):
        self.records[model_id] = {"model_id": model_id, **record}
        return self.records[model_id]

    def set_proba(self, model_id, part, proba):
        p = np.clip(np.asarray(proba, dtype=np.float64), self.eps, 1 - self.eps)
        self.proba.setdefault(model_id, {})[part] = p


@dataclass
class SplitData:
    """Provide row positions and labels for the shared model split."""

    split: dict
    y_all: np.ndarray

    def rows(self, part):
        return self.split[part]

    def y(self, part):
        return self.y_all[self.split[part]]
