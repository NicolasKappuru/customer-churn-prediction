"""Runtime environment.

Seeds the libraries, detects the usable accelerators and records the library
versions, so runs stay reproducible and the environment is part of the
artifacts.
"""

import random

import numpy as np
import pandas as pd


def model_environment(seed):
    """Seed libraries, detect accelerators and return ``(xgb_device, env)``."""
    import optuna
    import sklearn
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    random.seed(seed)
    np.random.seed(seed)
    # XGBoost reports at build time whether CUDA support was compiled in.
    try:
        xgb_cuda = bool(xgb.build_info().get("USE_CUDA", False))
    except Exception:
        xgb_cuda = False
    xgb_device = "cuda" if xgb_cuda else "cpu"

    env = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "optuna": optuna.__version__,
        "xgb_device": xgb_device,
    }
    return xgb_device, env
