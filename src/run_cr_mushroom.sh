#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1420
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 24:00:00

#job parameters
n_steps=8123
n_jobs=4
n_processes=1
config="configs.config_mushroom"
res_path="../results/results_mushroom"
res_prefix="mushroom"
logging_level="info"
start_seed=0

#model parameters
nn_drop_prob=0.2
nn_num_layers=1
nn_num_hidden_units=100
nn_regularization_factor=0.0001
nn_learning_rate=0.001
nn_num_epochs=10
nn_batch_size_factor=10
nn_replay_buffer_size=100000
nn_ucb_confidence_factor=0.01
nn_UCB_extra_t_factor=-1
nn_TS_exploration_variance=0.00001
nn_use_cuda=True
nn_early_stopping=True
nn_epsilon=0.05
linucb_alpha=0.1
xgb_exploration_factor=1.0
xgb_max_depth=10
xgb_n_estimators=100
xgb_learning_rate=0.05
xgb_gamma=0.0
xgb_lambda=0.0
xgb_enable_categorical=False

initial_random_selections=10


python server_runner_cr.py --config ${config} --result_path ${res_path} --n_jobs ${n_jobs} --n_steps ${n_steps} --n_processes ${n_processes} --result_prefix ${res_prefix} --logging_level ${logging_level} --start_seed ${start_seed} --nn_dropout_probability ${nn_drop_prob} --nn_num_layers ${nn_num_layers} --nn_num_hidden_units ${nn_num_hidden_units} --nn_regularization_factor ${nn_regularization_factor} --nn_learning_rate ${nn_learning_rate} --nn_num_epochs ${nn_num_epochs} --nn_batch_size_factor ${nn_batch_size_factor} --nn_replay_buffer_size ${nn_replay_buffer_size} --nn_ucb_confidence_factor ${nn_ucb_confidence_factor} --nn_UCB_extra_t_factor ${nn_UCB_extra_t_factor} --nn_TS_exploration_variance ${nn_TS_exploration_variance} --nn_use_cuda ${nn_use_cuda} --nn_early_stopping ${nn_early_stopping} --nn_epsilon ${nn_epsilon} --linucb_alpha ${linucb_alpha} --xgb_exploration_factor ${xgb_exploration_factor} --xgb_max_depth ${xgb_max_depth} --xgb_n_estimators ${xgb_n_estimators} --xgb_learning_rate ${xgb_learning_rate} --xgb_gamma ${xgb_gamma} --xgb_lambda ${xgb_lambda} --xgb_enable_categorical ${xgb_enable_categorical} --initial_random_selections ${initial_random_selections}
