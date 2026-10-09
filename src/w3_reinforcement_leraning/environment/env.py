from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd



from w3_reinforcement_leraning.environment.constants import (  # noqa: E402
    ACTION_DEFINITIONS,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    ENGINEERED_FEATURES,
    INTERNET_DEPENDENT_SERVICES,
    OBSERVATION_FEATURES,
    RAW_FEATURES,
    SERVICE_PRICES,
)
from w3_reinforcement_leraning.environment.supervised_model.preprocessing_data import (  
    TelcoPreprocessor,
)
from w3_reinforcement_leraning.environment.supervised_model.supervised_model import (  
    MODEL_ARTIFACT,
)

class TelcoRetentionEnv(gym.Env[np.ndarray, int]):
    def __init__():
        pass





    