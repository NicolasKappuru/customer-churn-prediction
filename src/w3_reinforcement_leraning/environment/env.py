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

from w3_reinforcement_leraning.environment.customer_selector import (
    CustomerSelector
)

from w3_reinforcement_leraning.environment.supervised_model.preprocessor_customer_data import (  
    PreprocessorCustomerData
)
from w3_reinforcement_leraning.environment.supervised_model.supervised_model import (  
    SupervisedModel
)
from w3_reinforcement_leraning.environment.actions_manager import (
    ActionsManager
)

class TelcoRetentionEnv(gym.Env):
    def __init__(self):
        
        super().__init__()
        self.customer_selector = CustomerSelector()
        self.preprocessor_customer_data = PreprocessorCustomerData()
        self.supervised_model = SupervisedModel()
        self.action_manager = ActionsManager()

        self.customer_data_df: pd.DataFrame | None = None

        self.probability_churn_before = 0
        self.probability_churn_after = 0
        
        self.action_space = spaces.Discrete(len(ACTION_DEFINITIONS))
        self.observation_space = spaces.Box( # Observation space is the vector of information processed
            low=-np.inf,
            high=np.inf,
            shape=(48,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """ Start each episode with this reset """
        
        self.customer_data_df = self.customer_selector.get_random_customer() # select a customer random
        customer_dic = self.customer_data_df.iloc[0].to_dict()

        customer_features = self.preprocessor_customer_data.preprocess_customer(customer_dic) # preprocess the data of customer

        self.probability_churn_before = self.supervised_model.predict_probability(customer_features) # get probability initial of be churn

        observation_df = customer_features.copy() # create a copy of df and add the feat of churn probability
        observation_df["churn_probability"] = self.probability_churn_before

        observation = observation_df.to_numpy(dtype=np.float32).flatten() # create observation like defined vector of 48 floats

        return observation, {}


    def step(self, action):
        self.customer_data_df, action_cost = self.action_manager.apply_action(self.customer_data_df, action)
        customer_dic = self.customer_data_df.iloc[0].to_dict()  

        customer_features = self.preprocessor_customer_data.preprocess_customer(customer_dic)

        self.probability_churn_after = (self.supervised_model.predict_probability(customer_features))

        observation_df = customer_features.copy()
        observation_df["churn_probability"] = self.probability_churn_after

        observation = observation_df.to_numpy(dtype=np.float32).flatten()


        churn_disminution = self.probability_churn_before - self.probability_churn_after
        cost = 1 - action_cost

        reward = churn_disminution * cost

        terminated = True
        truncated = False
        info = {
            "probability_before": self.probability_churn_before,
            "probability_after": self.probability_churn_after,
            "action": action,
            "action_cost": action_cost,
        }

        return observation, reward, terminated, truncated, info





    