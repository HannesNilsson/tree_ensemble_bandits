import importlib
import logging

from base import config_lib

def run_experiment_job(job_id, config_path, n_jobs, n_steps, start_seed):

    logging.info('Seed: ' + str("%02d" % (start_seed,)) + ', job:' + str(job_id + 1) + ' of ' + str(n_jobs))
    
    # Loading in the experiment config file
    config_module = importlib.import_module(config_path)
    config = config_module.get_config()
    
    # Override number of iterations
    if n_steps is not None:
        config = config_lib.Config(config.name, 
                                   config.agents, 
                                   config.environments, 
                                   config.experiments, 
                                   n_steps, 
                                   config.n_seeds)
    
    job_config = config_lib.get_job_config(config, job_id, start_seed=start_seed)
    experiment = job_config['experiment']
    figures = experiment.run_experiment()

    # Delete environment and agent
    del experiment.environment
    del experiment.agent

    return (job_id, (figures, experiment.results))