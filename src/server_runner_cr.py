from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import importlib
import time
import sys
import shutil
import multiprocessing
import logging
import numpy as np
import pandas as pd
import plotnine as gg
import matplotlib
from base import config_lib
from util import helpers_cr

matplotlib.use('agg')

sys.path.append(os.getcwd())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run server simulation')
    parser.add_argument('--predefined_config_id', help='job_id', type=int)
    parser.add_argument('--config', help='config', type=str)
    parser.add_argument('--n_jobs', help='n_jobs', type=int)
    parser.add_argument('--n_steps', help='n_steps', type=int)
    parser.add_argument('--result_path', help='result_path', type=str)
    parser.add_argument('--result_prefix', help='result_prefix', type=str)
    parser.add_argument('--result_pdf', help='result_pdf', type=bool, default=False)
    parser.add_argument('--n_processes', help='n_processes', type=int)
    parser.add_argument('--start_seed', help='start_seed', type=int, default=0)
    parser.add_argument('--logging_level', help='logging_level', type=str, default='info')

    args, unknown_args = parser.parse_known_args()

    # Set logging level
    logging_level = None
    if args.logging_level == 'debug' or args.logging_level == 'DEBUG':
        logging_level = logging.DEBUG
    elif args.logging_level == 'info' or args.logging_level == 'INFO':
        logging_level = logging.INFO
    elif args.logging_level == 'warning' or args.logging_level == 'WARNING':
        logging_level = logging.WARNING
    elif args.logging_level == 'error' or args.logging_level == 'ERROR':
        logging_level = logging.ERROR
    elif args.logging_level == 'critical' or args.logging_level == 'CRITICAL':
        logging_level = logging.CRITICAL

    if logging_level is None:
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
        logging.warning('Unknown logging level, defaulting to INFO')
    else:
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging_level)

    # Set the configuration
    config_path = args.config
    n_jobs = args.n_jobs
    
    # Set the result output path
    if args.result_path is not None:
        result_path = args.result_path 
    else:
        result_path = '../results'
        
    # Set the experiment result folder prefix
    if args.result_prefix is not None:
        result_prefix = args.result_prefix
    else:
        result_prefix = 'results'
        
    result_path = result_path + '/' + result_prefix + '_' + time.strftime("%Y%m%d-%H%M%S-") + str(time.time())[-6:-1]
    os.mkdir(result_path)
    logging.info('Results-path' + result_path)
    
    # Print command line arguments to file
    with open(result_path + '/experiment_args.txt', 'w') as experiment_args:
        experiment_args.write('## Base Arguments ##\n')
        experiment_args.write(str(args) + '\n')
        experiment_args.write('## Config-specific Arguments ##\n')
        experiment_args.write(str(unknown_args) + '\n')
    
    # Loading in the experiment config file
    config_module = importlib.import_module(config_path)
    config = config_module.get_config()
    
    logging.info('Starting experiments with config \'' + config_path + '\', n_jobs ' + str(n_jobs))

    # Either run in a single process or in multiple processes
    if args.n_processes is not None and args.n_processes > 1:
        if args.n_processes > n_jobs:
            raise ValueError("More processes than jobs!")
        pool = multiprocessing.Pool(processes = args.n_processes)
        jobs = [(job_id, config_path, n_jobs, args.n_steps, args.start_seed)
                for job_id in range(n_jobs)]
        experiments = dict(pool.starmap(helpers_cr.run_experiment_job, jobs))
    else:
        experiments = dict()
        for job_id in range(n_jobs):
            experiments[job_id] = helpers_cr.run_experiment_job(job_id,
                                                             config_path,
                                                             n_jobs,
                                                             args.n_steps,
                                                             args.start_seed)[1]

    # Save results from experiments
    agent_names = list(config.agents.keys())
    results = []
    for job_id in range(n_jobs):
        figures = experiments[job_id][0]
        results.append(experiments[job_id][1])
        agent_name = agent_names[job_id%len(agent_names)]
      
        if figures is not None:
            for fig_id, fig in enumerate(figures):
                for ax in fig.axes:
                    ax.set_title(agent_name + ', job ' + str(job_id))
                fig_path = result_path + '/job_id_' + str(job_id) + \
                           '_fig_' + str(fig_id)
                fig.savefig(fig_path + '.png', bbox_inches='tight', pad_inches = 0)
                if args.result_pdf:
                    fig.savefig(fig_path + '.pdf', bbox_inches='tight', pad_inches = 0)
                if hasattr(fig, 'folium_map'):
                    fig.folium_map.save(fig_path + '.html')
      
    # Copy config file to results folder for reproducability
    config_file_path = './' + config_path.replace('.','/') + '.py'
    shutil.copyfile(config_file_path, result_path + '/' + config_path + '.py')

    #############################################################################
    # Collating data with Pandas
    params_df = config_lib.get_params_df(config)
    df = pd.merge(pd.concat(results), params_df, on='unique_id')
    df.to_csv(result_path + '/results.csv')
    
    # Aggregate data for regret plots
    plt_df_inst = (df.groupby(['agent', 't'])
                .agg({'instant_regret': np.mean})
                .reset_index())
    plt_df_cum = (df.groupby(['agent', 't'])
                .agg({'cum_regret': np.mean})
                .reset_index())      
    
    
    plt_df_inst_best = (plt_df_inst.groupby(['agent', 't'])
                .agg({'instant_regret': np.mean})
                .reset_index())
    plt_df_inst_best['best_instant_regret'] = plt_df_inst_best \
                .groupby(['agent']) \
                .cummin()['instant_regret']
                
    #############################################################################
    # Plotting and analysis (uses plotnine by default)
    gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
    gg.theme_update(figure_size=(12, 8))
    
    p_inst = (gg.ggplot(plt_df_inst)
     + gg.aes('t', 'instant_regret', colour='agent')
     + gg.geom_line()+ gg.geom_line(size=1.25, alpha=0.75)
     + gg.scale_colour_brewer(name='agent', type='qual', palette='Paired')
     + gg.labels.xlab('time period (t)')
     + gg.labels.ylab('instant regret'))
    
    p_cum = (gg.ggplot(plt_df_cum)
     + gg.aes('t', 'cum_regret', colour='agent')
     + gg.geom_line()+ gg.geom_line(size=1.25, alpha=0.75)
     + gg.scale_colour_brewer(name='agent', type='qual', palette='Paired')
     + gg.labels.xlab('time period (t)')
     + gg.labels.ylab('cumulative regret'))
    
    # Save final results plot to results folder
    p_inst.save(result_path + '/results_instant_regret.png')
    p_inst.save(result_path + '/results_instant_regret.eps')
    
    p_cum.save(result_path + '/results_cumulative_regret.png')
    p_cum.save(result_path + '/results_cumulative_regret.eps')
    
    logging.info('Finished experiments, shutting down!')
