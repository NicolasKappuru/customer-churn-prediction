from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd



from w3_reinforcement_leraning.environment.constants import (  
    ACTION_DEFINITIONS,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    ENGINEERED_FEATURES,
    INTERNET_DEPENDENT_SERVICES,
    OBSERVATION_FEATURES,
    RAW_FEATURES,
    SERVICE_PRICES,
)
from w3_reinforcement_leraning.environment.supervised_model.preprocessor_customer_data import (  
    PreprocessorCustomerData,
)
from w3_reinforcement_leraning.environment.supervised_model.supervised_model import (  
    SupervisedModel
)

class TelcoRetentionEnv(gym.Env):
    def __init__(self):

        
        
        super().__init__()

        self.action_space = spaces.Discrete(len(ACTION_DEFINITIONS))
        self.observation_space = spaces.Box( # Observation space is the vector of information processed
            low=-np.inf,
            high=np.inf,
            shape=(48,),
            dtype=np.float32
        )

        def reset(self):
            pass


        def step(self):
            pass





    