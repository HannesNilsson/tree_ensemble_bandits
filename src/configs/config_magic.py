from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import logging
import os
import argparse
import collections
import functools
import numpy as np
import scipy.stats as st
import pandas as pd
from collections import defaultdict
from pathlib import Path
import math

from base.config_lib import Config

from experiments.experiment_cr import ExperimentLoggingVisualizationNoAction

from agents.bernoulli_agents import BernoulliLinUCB, BernoulliLinTS, BernoulliNeuralUCB, BernoulliNeuralTS

from environments.environment_cr import ClassRecommenderEnvWithRng

EPSILON = np.finfo(float).eps


def get_config():
    """Generates the config for the experiment."""
    name = 'magic'

    parser = argparse.ArgumentParser(description='Run server simulation')

    # Problem environment command line arguments below
    parser.add_argument('--start_seed', help='start_seed', type=int, default=0)

    # Neural network command line arguments below
    parser.add_argument('--nn_num_layers', help='nn_num_layers', type=int, default=3)
    parser.add_argument('--nn_num_hidden_units', help='nn_num_hidden_units', type=int, default=10)
    parser.add_argument('--nn_regularization_factor', help='nn_regularization_factor', type=float, default=0.1)
    parser.add_argument('--nn_ucb_delta', help='nn_ucb_delta', type=float, default=0.1)
    parser.add_argument('--nn_ucb_confidence_factor', help='nn_ucb_confidence_factor', type=float, default=1.)
    parser.add_argument('--nn_TS_exploration_variance', help='nn_TS_exploration_variance', type=float, default=1.)
    parser.add_argument('--nn_dropout_probability', help='nn_dropout_probability', type=float, default=0.2)
    parser.add_argument('--nn_learning_rate', help='nn_learning_rate', type=float, default=0.05)
    parser.add_argument('--nn_num_epochs', help='nn_num_epochs', type=int, default=10)
    parser.add_argument('--nn_use_cuda', help='nn_use_cuda', type=bool, default=True)
    parser.add_argument('--nn_training_windows_size', help='nn_training_windows_size', type=int, default=10)
    parser.add_argument('--nn_batch_size_factor', help='nn_batch_size_factor', type=int, default=10)
    parser.add_argument('--nn_replay_buffer_size', help='nn_replay_buffer_size', type=int, default=4000)
    parser.add_argument('--nn_UCB_extra_t_factor', help='nn_UCB_extra_t_factor', type=int, default=0)
    parser.add_argument('--nn_early_stopping', help='nn_early_stopping', type=bool, default=True)
    parser.add_argument('--nn_epsilon', help='nn_epsilon', type=float, default=0.05)

    parser.add_argument('--xgb_exploration_factor', help='xgb_exploration_factor', type=float, default=1.0)
    parser.add_argument('--xgb_exploration_variance', help='xgb_exploration_variance', type=float, default=1.0)
    parser.add_argument('--xgb_max_depth', help='xgb_max_depth', type=int, default=6)
    parser.add_argument('--xgb_n_estimators', help='xgb_n_estimators', type=int, default=100)
    parser.add_argument('--xgb_learning_rate', help='xgb_learning_rate', type=float, default=0.3)
    parser.add_argument('--xgb_gamma', help='xgb_gamma', type=float, default=100.0)
    parser.add_argument('--xgb_lambda', help='xgb_lambda', type=float, default=1.0)
    parser.add_argument('--xgb_enable_categorical', help='xgb_enable_categorical', type=bool, default=False)

    parser.add_argument('--linucb_alpha', help='linucb_alpha', type=float, default=1.0)

    parser.add_argument('--initial_random_selections', help='initial_random_selections', type=int, default=10)

    args, _ = parser.parse_known_args()

    # Read data
    cr_data = _read_data()

    # Shuffle data
    cr_data = cr_data.sample(frac=1, random_state=args.start_seed).reset_index(drop=True)

    logging.debug('cr_data: ', cr_data)

    set_classes = cr_data[10].unique()
    n_classes = len(set_classes)
    n_features = cr_data.shape[1] - 1

    # Requires even number of features for NeuralUCB weight initiations
    context_length = n_features * n_classes


    def cr_sample(data=cr_data, dp_list=[0]):
        """Generates a single sample from the cr dataset

        Parameters
        ----------
        data - 2D numpy array
            Matrix with data points as rows and features as columns
        rng - Numpy random number generator
            Numpy random generator (e.g., with a specified seed, for reproducibility)

        Returns
        -------
        Sample as a row vector (numpy array)
        """
        idx = dp_list.pop(0)
        return data.iloc[idx]
        

    def env_arm_context_transform_function(arm):
        """Function which generates the complete contextual feature vector for each arm,
        when accessed by the agent. In this case, combines dynamic information () for positional 
        encoding, together with static information from the arm (label) for positional encoding.

        Parameters
        ----------
        arm - Arm object
            Object containing arm-specific data

        Returns
        -------
        Column vector (as numpy array) containing the arm-specific contextual features
        """

        # Not used here, but needs to exist for the agent to work

        return
        

    def env_arm_context_and_reward_update_function(arm, advance_data, rng=np.random.default_rng()):
        """Function which is executed once per iteration and arm ID, at the end of each
        iteration and (optionally) once before the first iteration. The function can modify
        the arm object (below, the global contextual feature available through "advance_data"
        is assigned to "arm.context". Note that, for practical reasons (context is supplied
        to the agent in the form of a "context dict"), two duplicate arms are identically
        updated here.

        Parameters
        ----------
        arms - List of arm objects
            For this specific arm ID: One arm from the internal environment, and one
            from the "context dict" supplied to the agent in each iteration.
        advance_data - Object
            Global contextual feature of the environment.
        rng - Numpy random number generator
            Numpy random generator (e.g., with a specified seed, for reproducibility)
        """
        true_class = advance_data[10]
        context = np.array([advance_data[:-1]])
        pre_padding = np.zeros(((arm.label) * n_features,1))
        post_padding = np.zeros(((n_classes - arm.label - 1) * n_features,1))
        context = np.concatenate((pre_padding, context.T, post_padding), axis=0)
        if len(context) % 2 == 1:
            context = np.append(context, np.zeros((1,1)), axis=0)
        context /= np.linalg.norm(context)
        context /= np.linalg.norm(context)
        arm.set_context(context)
        arm.set_reward(int(true_class == arm.label) + rng.normal(0, 0.001))


    def env_iteration_data_function(env):
        """Function which assigns values to the iteration data store, for debug purposes.
        They can be output in the results CSV file as separate columns.

        Parameters
        ----------
        env - Environment
            The environment

        Returns
        -------
        Iteration debug data as a dict of string (column name) to arbitrary data.
        """
        if env.advance_data is not None:
            return {'advance_data': env.advance_data[-1]}
        else:
            return {}
    
    
    def no_context_observation_function(arm):
      """Function which returns a constant observation for each arm"""
      return np.array([[1.0]])


    env_constructor = functools.partial(ClassRecommenderEnvWithRng,
                                        args.start_seed,
                                        len(cr_data),
                                        observation_function=env_arm_context_transform_function,
                                        arm_update_function=env_arm_context_and_reward_update_function,
                                        advance_function=cr_sample,
                                        advance_initially=True,
                                        iteration_data_function=env_iteration_data_function,
                                        label_set=set_classes
                                        )
                                        
    # ---------------------------------------------- EXPERIMENT SETUP ----------------------------------------------
    
    # Agent specifications, each will be run once for each random seed (to ensure fair comparison)
    agents = collections.OrderedDict(
        [
            ('LinUCB',
              functools.partial(BernoulliLinUCB,
                                env_constructor, context_length, args.nn_use_cuda, args.linucb_alpha)),

            ('LinTS',
              functools.partial(BernoulliLinTS,
                                env_constructor, context_length, args.nn_use_cuda, args.linucb_alpha)),

            ('NeuralUCB',
             functools.partial(BernoulliNeuralUCB,
                               env_constructor, context_length, args.nn_num_layers,
                               args.nn_num_hidden_units, args.nn_regularization_factor, args.nn_ucb_delta,
                               args.nn_ucb_confidence_factor, args.nn_dropout_probability, args.nn_learning_rate,
                               args.nn_num_epochs, args.nn_batch_size_factor, args.nn_replay_buffer_size, args.nn_early_stopping, 
                               args.nn_UCB_extra_t_factor, args.nn_use_cuda, args.initial_random_selections)),
              
            ('NeuralTS',
             functools.partial(BernoulliNeuralTS,
                               env_constructor, context_length, args.nn_num_layers,
                               args.nn_num_hidden_units, args.nn_regularization_factor, args.nn_TS_exploration_variance, 
                               args.nn_dropout_probability, args.nn_learning_rate, args.nn_num_epochs, args.nn_batch_size_factor,
                               args.nn_replay_buffer_size, args.nn_early_stopping, args.nn_use_cuda,
                               args.initial_random_selections)),
        ]
    )

    # Environment specification(s)
    environments = collections.OrderedDict(
        [('env',
          env_constructor)]
    )

    # Experiment specification(s)
    experiments = collections.OrderedDict(
        [(name, ExperimentLoggingVisualizationNoAction)]
    )

    # Number of steps and max number of random seeds (these can be ignored, since they are typically
    # assigned through command line arguments).
    n_steps = 400
    n_seeds = 1000

    # Experiment configuration to be returned
    config = Config(name, agents, environments, experiments, n_steps, n_seeds)
    return config


def _read_data():
    # fetch dataset 
    filepath = '../datasets/magic04.data'
    data = pd.read_csv(filepath, header=None)

    # convert class labels to 0 and 1
    data[10] = data[10].map({'g': 1, 'h': 0})

    return data
