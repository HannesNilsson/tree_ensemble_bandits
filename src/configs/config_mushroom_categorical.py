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

from agents.bernoulli_agents import BernoulliXGBoostTEUCB, BernoulliXGBoostTETS, \
    BernoulliRandomForestTEUCB, BernoulliRandomForestTETS, BernoulliTreeBootstrapXGBoost, \
    BernoulliTreeBootstrapRandomForest, BernoulliTreeBootstrapDecisionTree

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
    parser.add_argument('--xgb_enable_categorical', help='xgb_enable_categorical', type=bool, default=True)

    parser.add_argument('--linucb_alpha', help='linucb_alpha', type=float, default=1.0)

    parser.add_argument('--initial_random_selections', help='initial_random_selections', type=int, default=10)

    args, _ = parser.parse_known_args()

    # Read data
    cr_data = _read_data()

    # Shuffle data
    cr_data = cr_data.sample(frac=1, random_state=args.start_seed).reset_index(drop=True)

    logging.debug('cr_data: ', cr_data)

    set_classes = cr_data['label'].unique()
    n_classes = len(set_classes)
    n_features = cr_data.shape[1] - 1

    context_length = n_features + 1  #extra feature is arm-ID

    #all features categorical
    feature_types = ['c' for i in range(context_length)]

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
        #idx = rng.integers(0, len(data) - 1)
        idx = dp_list.pop(0)
        #logging.info('cr_sample: data.iloc[idx]: ' + str(data.iloc[idx]))
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
        """Function which is executed once per iteration and edge ID, at the end of each
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
        true_class = advance_data.iloc[-1]
        context = np.array([advance_data.iloc[:-1]])
        label = np.zeros((1,1)) + arm.label
        context = np.concatenate((label, context.T), axis=0)
        context = context.astype(int)
        arm.set_context(context)
        arm.set_reward(int(true_class == arm.label))

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
            return {'advance_data': env.advance_data.iloc[-1]}
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
            ('TreeBootstrapRF',
                functools.partial(BernoulliTreeBootstrapRandomForest,
                                    env_constructor, context_length, 
                                    args.nn_use_cuda, args.initial_random_selections)),

            ('TreeBootstrapXGBoost',
                functools.partial(BernoulliTreeBootstrapXGBoost,
                                    env_constructor, context_length, 
                                    args.nn_use_cuda, args.xgb_enable_categorical,
                                    args.initial_random_selections, feature_types)),

            ('TreeBootstrapDT',
                functools.partial(BernoulliTreeBootstrapDecisionTree,
                                    env_constructor, context_length, 
                                    args.nn_use_cuda, args.initial_random_selections)),

            ('XGBoostTEUCB',
            functools.partial(BernoulliXGBoostTEUCB,
                              env_constructor, context_length, args.nn_use_cuda, 
                              args.xgb_exploration_factor, args.xgb_max_depth, args.xgb_n_estimators, 
                              args.xgb_learning_rate, args.xgb_gamma, args.xgb_lambda, args.xgb_enable_categorical,
                              args.initial_random_selections, feature_types)),

            ('XGBoostTETS',
             functools.partial(BernoulliXGBoostTETS,
                               env_constructor, context_length, args.nn_use_cuda,
                               args.xgb_exploration_variance, args.xgb_max_depth, args.xgb_n_estimators,
                               args.xgb_learning_rate, args.xgb_gamma, args.xgb_lambda, args.xgb_enable_categorical,
                               args.initial_random_selections, feature_types)),
            
            ('RandomForestTEUCB',
                functools.partial(BernoulliRandomForestTEUCB,
                                    env_constructor, context_length, 
                                    args.nn_use_cuda, args.xgb_exploration_factor, args.xgb_max_depth,
                                    args.xgb_n_estimators, args.initial_random_selections)),

            ('RandomForestTETS',
                functools.partial(BernoulliRandomForestTETS,
                                    env_constructor, context_length,
                                    args.nn_use_cuda, args.xgb_exploration_variance, args.xgb_max_depth,
                                    args.xgb_n_estimators, args.initial_random_selections)),

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
    filepath = '../datasets/agaricus-lepiota.data'
    columns = ["label", "cap-shape", "cap-surface", "cap-color", "bruises", 
               "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", 
               "stalk-shape", "stalk-root", "stalk-surface-above-ring", 
               "stalk-surface-below-ring", "stalk-color-above-ring", 
               "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", 
               "ring-type", "spore-print-color", "population", "habitat"]
    
    dataset = pd.read_csv(filepath, names=columns)

    dataset.dropna(subset=['label'])

    target = dataset.columns[0]

    dataset[target] = dataset.pop(target)

    dataset['stalk-root'] = dataset['stalk-root'].fillna('m')  #'m' for missing

    dataset = dataset.astype("category").apply(lambda x: x.cat.codes).astype(int)

    return dataset
