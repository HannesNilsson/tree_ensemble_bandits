from __future__ import print_function

import logging

import argparse
import copy
import numpy as np
import pandas as pd

from base.experiment import BaseExperiment


class ExperimentLoggingVisualizationNoAction(BaseExperiment):
    def run_step_maybe_log(self, t):
        # Evolve the bandit (potentially contextual) for one step and pick action
        observation = self.environment.get_observation()

        action = observation[self.agent.pick_action(observation)]

        # Compute useful stuff for regret calculations
        optimal_reward = self.environment.get_optimal_reward()
        
        expected_reward = self.environment.get_expected_reward(action)
        
        reward = self.environment.get_stochastic_reward(action)

        # Update the agent using realized rewards + bandit learing
        self.agent.update_observation(observation, action, reward)

        # Log whatever we need for the plots we will want to use.
        instant_regret = optimal_reward - expected_reward
        self.cum_regret += instant_regret

        # Advance the environment (used in nonstationary experiment)
        self.environment.advance(action, reward)

        if (t + 1) % self.rec_freq == 0:
            self.data_dict = {'t': (t + 1),
                            'instant_regret': instant_regret,
                            'cum_regret': self.cum_regret,
                            'action': action.label,
                            'unique_id': self.unique_id}
            self.results.append(self.data_dict)

        if (t + 1) % 10 == 0:
            logging.info('Seed: ' + str("%02d" % (self.seed,)) + ', job: ' + str("%02d" % (self.unique_id,)) + ', t: ' + str(t + 1))


    def run_experiment(self):
        """Run the experiment for n_steps and collect data"""
        np.random.seed(self.seed)
        self.cum_regret = 0
        self.cum_optimal = 0

        for t in range(self.n_steps):
            self.run_step_maybe_log(t)

        self.results = pd.DataFrame(self.results)