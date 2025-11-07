from __future__ import division
from __future__ import generators
from __future__ import print_function

from functools import partial

import logging
import copy
import numpy as np
import scipy.stats as st
import pandas
import random

from collections import defaultdict
from base.environment import Environment
from util.arm_dict import ArmDict, Arm

##############################################################################

class ClassRecommenderEnvWithRng(Environment):

    def __init__(self, 
                 seed,
                 n_datapoints,
                 observation_function=None,
                 arm_update_function=None,
                 advance_function=None,                 
                 advance_initially=True,
                 iteration_data_function=None,
                 label_set=None):
        
        self.rng_prp_eps_greedy = np.random.default_rng(seed=seed+3)  #selecting same random actions for all agents
        self.rng_dp = np.random.default_rng(seed=seed+4)  #selecting same datapoints for all agents
        self.dp_list = list(self.rng_dp.permutation(n_datapoints))
        self.rng_bt = np.random.default_rng(seed=seed+5)  #breaking ties identically
        self.advance_function = advance_function
        self.iteration_data_function = iteration_data_function
        self.arm_update_function = arm_update_function
        self.advance_initially = advance_initially

        # Create arm sets
        self._create_arm_set(label_set)
        self._create_observation_arm_set(observation_function)

        # Initial advance of the environment
        if advance_initially:
            self.advance()

        self.optimal_reward = 1
            
    def _create_arm_set(self, label_set):
        self.arm_set = ArmDict()
        for arm in label_set:
            self.arm_set[arm] = Arm(arm, self.arm_set)

    def _create_observation_arm_set(self, observation_function):
        self.observation_arm_set = copy.deepcopy(self.arm_set)
        self.observation_arm_set.set_arm_weight_function()
        if observation_function is None:
            self.observation_arm_set.set_arm_weight_function(lambda _: np.array([[0]]))
        else:
            self.observation_arm_set.set_arm_weight_function(observation_function)

    def overwrite_arm_weight(self, arm_weight):
        """Overwrites the existing arms weights with specified values"""
        
        for arm in arm_weight:
            self.arm_set[arm].weight = arm_weight[arm]

    def get_observation(self):
        """Returns an observation from the environment."""
        return self.observation_arm_set
    
    def get_optimal_action(self):
        """Returns the optimal action for the environment at that point."""

        optimal_action = []
        for arm in self.arm_set:
            if not optimal_action:
                optimal_action.append(self.arm_set[arm])
            else:
                if self.arm_set[arm].weight > optimal_action[0].weight:
                    #print("overwriting optimal arm with", arm)
                    optimal_action = [self.arm_set[arm]]
                elif self.arm_set[arm].weight == optimal_action[0].weight:
                    optimal_action.append(self.arm_set[arm])
        optimal_action = self.rng_bt.choice(optimal_action)
        return optimal_action


    def get_optimal_reward(self):
        """Returns optimal reward arm
        
        Returns
        -------
        Optimal reward
        """
        return self.optimal_reward
    
    def get_stochastic_reward(self, action):
        """Gets the reward for the action."""

        return action.reward

    def get_expected_reward(self, action):
        """Gets the expected reward of an action."""
        return self.get_stochastic_reward(action)
    
    def pick_random_arm(self):
        """Picks a random arm from the arm set."""
        arms = list(self.arm_set.values())
        random_arm = self.rng_prp_eps_greedy.choice(arms)
        return random_arm

    def advance(self, *args, **kwargs):
        """Advance the environment"""
        if self.advance_function is not None:
            advance_data = self.advance_function(dp_list=self.dp_list)
        else:
            advance_data = None
        self.advance_data = advance_data

        if self.arm_update_function is not None:
            for arm in self.arm_set:
                if advance_data is not None:
                    self.arm_update_function(self.arm_set[arm], advance_data)
                    self.arm_update_function(self.observation_arm_set[arm], advance_data)
                else:
                    self.arm_update_function(self.arm_set[arm])
                    self.arm_update_function(self.observation_arm_set[arm])

    def get_iteration_data(self):
        if self.iteration_data_function is not None:
            return self.iteration_data_function(self)
        else:
            return {}