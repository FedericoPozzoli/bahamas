"""
BAHAMAS Inference Module
This module provides functionality for performing Bayesian inference using various sampling methods.
It includes methods for setting up and running inference, as well as plotting results.
It is designed to work with the BAHAMAS framework for analyzing gravitational wave signals.
It supports both Hamiltonian Monte Carlo (HMC) and nested sampling methods.
It also includes functions for loading configuration files, running inference, and saving results.

Classes:
    Method: Handles different methods of inference.
    BayesianInference: Manages the entire inference process, including loading configurations, running inference, and saving results.
    
Usage: 
    The script can be run from the command line with the following arguments:
    --config: Path to the YAML configuration file.
    --sources: Path to the YAML sources file.
    The script will load the configuration and sources files, run the inference, and save the results.    
"""

from bahamas.method import setting_inference
from bahamas.method import setting_nessai
from bahamas.logger_config import logger

import numpy as np
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
from nessai.flowsampler import FlowSampler
import yaml
import argparse
import matplotlib.pyplot as plt
import corner
import os


# Enable 64-bit precision in NumPyro
numpyro.enable_x64()

def get_last_part(path: str) -> str:
    """
    Extract the last part of a file path.

    Parameters:
    - path (str): The file path.

    Returns:
    - str: The last part of the file path.
    """
    return os.path.basename(path)

def get_first_part(path: str) -> str:
    """
    Extract the first part of a file path.

    Parameters:
    - path (str): The file path.

    Returns:
    - str: The first part of the file path.
    """
    return os.path.dirname(path)


class Method:
    """
    Class to handle the different methods of inference.
    """

    def __init__(self, config, log_like, **kwargs):
        """
        Initialize the Method class.

        Parameters:
        - config (dict): Configuration dictionary.
        - log_like (callable): Log-likelihood function.
        - kwargs (dict): Additional arguments for the inference method.
        """
        self.config = config
        self.log_like = log_like
        self.sampler = self.config['inference']['sampler']
        self.kwargs = kwargs

    def setup(self):
        """
        Set up the inference method based on the sampler specified in the configuration.
        """
        if self.sampler == 'NUTS':

            kernel = NUTS(self.log_like, adapt_mass_matrix=self.config['inference']['adapt_matrix'])
            self.method = MCMC(kernel, **self._get_mcmc_params())
            # Set a seed if provided, else set it to None
            self.seed = self.config['inference'].get('seed', None)
            if self.seed is not None:
                logger.info(f"Chosen seed: {self.seed}")
            else:
                logger.info("No seed provided, will use 100 as default")

        elif self.sampler == 'nested':            
            sampler_opts = self.config['inference']
            # To get checkpointing to work
            if 'flow_config' not in sampler_opts:
                sampler_opts['flow_config'] = None
            if 'checkpointing' not in sampler_opts:
                sampler_opts['checkpointing'] = True
            if 'checkpoint_on_training' not in sampler_opts:
                sampler_opts['checkpoint_on_training'] = True
                
            if 'max_threads' in sampler_opts:
                nthreads = sampler_opts['max_threads']
            
            # Select entries relevant for FlowSampler 
            keys = ['nlive', 'n_pool', 'flow_config', 'checkpointing', 'checkpoint_on_training', 'max_threads']
            sampler_kwargs = {key: self.config['inference'][key] for key in keys if key in sampler_opts}

            # Add beta to likelihood kwargs
            self.kwargs['beta'] = self.config['inference']['beta']
            
            # Filtering
            logger.info(f"Passing the following kwargs: {sampler_kwargs}")
            model = setting_nessai.nessai_model(self.log_like, **self.kwargs)
            name = get_first_part(self.config['inference']['file_post'])
            self.method = FlowSampler(model, resume=False, output=name, **sampler_kwargs, )

   

    def _get_mcmc_params(self):
        """
        Helper function to get MCMC parameters from the configuration.

        Returns:
        - dict: MCMC parameters.
        """
        return {
            'num_warmup': self.config['inference']['warmup'],
            'num_samples': self.config['inference']['samples'],
            'num_chains': self.config['inference']['chains'],
            'chain_method': self.config['inference']['chain_method'],
            'jit_model_args': False
        }
        
    def run(self):
        """
        Run the inference method and return the posterior samples.

        Returns:
        - np.ndarray: Posterior samples.
        """
        self.result = {}
        if self.sampler in ['NUTS']:
            if self.seed is not None:
                logger.info(f"Running HMC with seed: {self.seed}")
                self.method.run(jax.random.PRNGKey(self.seed), **self.kwargs)
            else:
                logger.info("Running HMC with default seed: 100")
                self.method.run(jax.random.PRNGKey(100), **self.kwargs)
            self.posterior = self.method.get_samples()
            self.chain = np.column_stack([self.posterior[key] for key in self.posterior])
            self.result['chain'] = self.chain
            self.plot_corner()
            self.plot_autocorrelation()

            if 'beta' in self.config['inference']:
                self.get_likelihood()
                #add to posterior "log_likelihood"
                self.posterior['log_likelihood'] = self.loglike
                #add to posterior "log_likelihood"
                self.posterior['beta'] = self.config['inference']['beta']
                self.result['chain'] = self.posterior

            return self.result
    
        elif self.sampler == 'nested':
            self.method.run()
            self.posterior = self.method.posterior_samples
            self.result['chain'] = np.column_stack([self.posterior[key] for key in self.posterior.dtype.names])
            return self.result
   
        else:
            logger.error(f"Sampler '{self.sampler}' is not supported.")
            raise ValueError(f"Sampler '{self.sampler}' is not supported.")

    def get_likelihood(self):
        """
        Compute and print the marginalized log-likelihood.
        """
        loglike = numpyro.infer.util.log_likelihood(model=self.log_like, posterior_samples=self.posterior, **self.kwargs)
        self.loglike = loglike.get('log_likelihood')
        
        
    def plot_corner(self):
        """
        Create and save a corner plot of the posterior samples.
        """
        plt.figure()
        corner.corner(self.result['chain'], quiet = True)
        name_ = get_last_part(self.config['inference']['file'])
        name_folder = get_first_part(self.config['inference']['file_post'])
        plt.savefig(f'{name_folder}/corner_{name_}.png')

    def plot_autocorrelation(self):
        """
        Create and save an autocorrelation plot of the posterior samples.
        """
        corr = numpyro.diagnostics.autocorrelation(self.result['chain'], axis=0)
        self.result['autocorr'] = corr
            
        plt.figure()
        for i in corr.T:
            plt.plot(i, alpha=0.3)
        plt.semilogx()
        plt.ylabel('Autocorrelation')
        name_ = get_last_part(self.config['inference']['file'])
        name_folder = get_first_part(self.config['inference']['file_post'])
        plt.savefig(f'{name_folder}/auto_{name_}.png')


class BayesianInference:
    """
    Class to manage the entire Bayesian inference process.
    """

    def __init__(self, config_file, sources_file):
        """
        Initialize the BayesianInference class.

        Parameters:
        - config_file (str): Path to the configuration file.
        - sources_file (str): Path to the sources file.
        """
        self.config_file = config_file
        self.sources_file = sources_file
        self.load_config_files()

    def load_config_files(self):
        """
        Load the configuration and sources files.
        """
        with open(self.sources_file, "r") as file:
            self.sources = yaml.safe_load(file)['sources']
        with open(self.config_file, "r") as file:
            self.config = yaml.safe_load(file)

        self.log_like, self.data, self.freqs, self.response, self.dt, self.t1, self.t2, self.dof, self.gen2 = setting_inference.set(
            config=self.config, sources=self.sources)

    def run_inference(self):
        """
        Run the inference method.

        Returns:
        - np.ndarray: Posterior samples.
        """
        kwargs = {
            'data': self.data, 'freqs': self.freqs, 'response': self.response, 
            'sources': self.sources, 'dt': self.dt, 't1': self.t1, 't2': self.t2, 'dof': self.dof, 'gen2': self.gen2
        }
        method = Method(self.config, self.log_like, **kwargs)
        method.setup()
        return method.run()

    def save_results(self):
        """
        Save the posterior samples to a file.
        """
        # Get seed from config if available
        self.seed = self.config['inference'].get('seed', None)
        if self.seed is not None:
            # Append before the .npz the seed value
            base, ext = os.path.splitext(self.config['inference']["file_post"])
            self.config['inference']["file_post"] = f"{base}.{self.seed}{ext}"
        logger.info(f"Saving results to {self.config['inference']['file_post']}")
        np.savez(self.config['inference']["file_post"], posterior=self.result)

    def run(self):
        """
        Run the entire inference process.
        """
        self.result = self.run_inference()
        self.save_results()

