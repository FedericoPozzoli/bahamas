"""
BAHAMAS Inference Module
This module provides functionality for performing Bayesian inference using various sampling methods.
It includes methods for setting up and running inference, as well as plotting results.
It is designed to work with the BAHAMAS framework for analyzing gravitational wave signals.
It supports Hamiltonian Monte Carlo (HMC), nested sampling, and ERYN transdimensional methods.
"""

from bahamas.method import setting_inference
from bahamas.method import setting_nessai
from bahamas.method import setting_eryn
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
    """Extract the last part of a file path."""
    return os.path.basename(path)

def get_first_part(path: str) -> str:
    """Extract the first part of a file path."""
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
        - log_like (callable): Log-likelihood function (ignored for ERYN).
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

        elif self.sampler == 'nested':            
            sampler_opts = self.config['inference']
            # To get checkpointing to work
            if 'flow_config' not in sampler_opts:
                sampler_opts['flow_config'] = None
            if 'checkpointing' not in sampler_opts:
                sampler_opts['checkpointing'] = True
            if 'checkpoint_on_training' not in sampler_opts:
                sampler_opts['checkpoint_on_training'] = True
            
            # Select entries relevant for FlowSampler 
            keys = ['nlive', 'n_pool', 'flow_config', 'checkpointing', 
                   'checkpoint_on_training', 'max_threads']
            sampler_kwargs = {key: self.config['inference'][key] 
                            for key in keys if key in sampler_opts}

            logger.info(f"Passing the following kwargs: {sampler_kwargs}")
            model = setting_nessai.nessai_model(self.log_like, **self.kwargs)
            name = get_first_part(self.config['inference']['file_post'])
            self.method = FlowSampler(model, resume=False, output=name, **sampler_kwargs)
          
        elif self.sampler == 'eryn':
            # ERYN transdimensional sampler
            logger.info("Setting up ERYN transdimensional sampler")
            
            # Create ERYN inference object
            self.eryn_inference = setting_eryn.ERYNInference(
                self.log_like, config=self.config, **self.kwargs)
            
            # Setup sampler
            self.method = self.eryn_inference.setup_sampler()
        
        else:
            logger.error(f"Sampler '{self.sampler}' is not supported.")
            raise ValueError(f"Sampler '{self.sampler}' is not supported.")

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
        - dict: Results dictionary containing posterior samples and diagnostics.
        """
        self.result = {}
        
        if self.sampler == 'NUTS':
            self.method.run(jax.random.PRNGKey(100), **self.kwargs)
            self.posterior = self.method.get_samples()
            self.chain = np.column_stack([self.posterior[key] for key in self.posterior])
            self.result['chain'] = self.chain
            self.plot_corner()
            self.plot_autocorrelation()

            if 'beta' in self.config['inference']:
                self.get_likelihood()
                self.posterior['log_likelihood'] = self.loglike
                self.posterior['beta'] = self.config['inference']['beta']
                self.result['chain'] = self.posterior

            return self.result
    
        elif self.sampler == 'nested':
            self.method.run()
            self.posterior = self.method.posterior_samples
            self.result['chain'] = np.column_stack([self.posterior[key] 
                                                    for key in self.posterior.dtype.names])
            return self.result
        
        elif self.sampler == 'eryn':
            eryn_results = self.eryn_inference.run()
            self.results = eryn_results
            
            # Store ERYN-specific results
            self.result['eryn_results'] = eryn_results
            self.result['egp_chain'] = eryn_results['egp_chain']
            self.result['scalar_chain'] = eryn_results['scalar_chain']
            self.result['nleaves'] = eryn_results['nleaves']
            self.result['log_like'] = eryn_results['log_like']
            self.result['mean_nodes'] = eryn_results['mean_nodes']
            self.result['mode_nodes'] = eryn_results['mode_nodes']
            
            # Create node distribution histogram
            self.plot_eryn_node_distribution(eryn_results['nleaves'])
            
            # Plot delta parameter traces if available
            if eryn_results['egp_chain'].shape[1] > 0:
                self.plot_eryn_delta_traces(eryn_results['egp_chain'], 
                                           eryn_results['nleaves'])
            
            self.plot_eryn_results()
            self.plot_scalar_only()

            return self.result
   
        else:
            logger.error(f"Sampler '{self.sampler}' is not supported.")
            raise ValueError(f"Sampler '{self.sampler}' is not supported.")

    def get_likelihood(self):
        """
        Compute and print the marginalized log-likelihood.
        """
        loglike = numpyro.infer.util.log_likelihood(
            model=self.log_like, 
            posterior_samples=self.posterior, 
            **self.kwargs
        )
        self.loglike = loglike.get('log_likelihood')
        
    def plot_corner(self):
        """
        Create and save a corner plot of the posterior samples.
        """
        plt.figure()
        corner.corner(self.result['chain'], quiet=True)
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
    
    def plot_eryn_node_distribution(self, nleaves):
        """
        Plot the distribution of number of EGP nodes.
        
        Parameters:
        - nleaves (array): Array of node counts across MCMC samples.
        """
        plt.figure(figsize=(8, 6))
        unique_nodes, counts = np.unique(nleaves, return_counts=True)
        plt.bar(unique_nodes, counts / counts.sum(), alpha=0.7, color='steelblue')
        plt.xlabel('Number of EGP Nodes (K)')
        plt.ylabel('Posterior Probability')
        plt.title('EGP Node Distribution')
        plt.grid(alpha=0.3)
        
        name_ = get_last_part(self.config['inference']['file'])
        name_folder = get_first_part(self.config['inference']['file_post'])
        plt.savefig(f'{name_folder}/eryn_node_distribution_{name_}.png', dpi=150)
        
        logger.info(f"Saved node distribution plot to {name_folder}/eryn_node_distribution_{name_}.png")
    
    def plot_eryn_delta_traces(self, egp_chain, nleaves, max_nodes=5):
        """
        Plot traces of delta parameters for ERYN sampling.
        
        Parameters:
        - egp_chain (array): Chain of EGP delta values [nsteps, max_nodes, ndim].
        - nleaves (array): Number of active nodes at each step.
        - max_nodes (int): Maximum number of node traces to plot.
        """
        nsteps = egp_chain.shape[0]
        max_K = min(max_nodes, egp_chain.shape[1])
        
        fig, axes = plt.subplots(max_K, 1, figsize=(12, 2*max_K), sharex=True)
        if max_K == 1:
            axes = [axes]
        
        for k in range(max_K):
            # Extract delta values for node k, masking inactive nodes
            deltas = egp_chain[:, k, 0].copy()
            # Mask where this node is inactive
            mask = nleaves <= k
            deltas[mask] = np.nan
            
            axes[k].plot(deltas, alpha=0.6, linewidth=0.5)
            axes[k].set_ylabel(f'$\\delta_{k}$')
            axes[k].grid(alpha=0.3)
        
        axes[-1].set_xlabel('MCMC Step')
        plt.tight_layout()
        
        name_ = get_last_part(self.config['inference']['file'])
        name_folder = get_first_part(self.config['inference']['file_post'])
        plt.savefig(f'{name_folder}/eryn_delta_traces_{name_}.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved delta traces plot to {name_folder}/eryn_delta_traces_{name_}.png")


    def plot_eryn_results(self, burnin_fraction=0.3, figsize=(12, 12)):
        """
        Create corner plots for ERYN inference results.
        
        Parameters:
        -----------
        results : dict
            Results dictionary from ERYNInference.run()
        burnin_fraction : float
            Fraction of samples to discard as burn-in (default: 0.3)
        figsize : tuple
            Figure size (default: (12, 12))
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Corner plot figure
        """
        # Extract data
        scalar_chain = self.results['scalar_chain']
        scalar_names = self.results['scalar_names']
        nleaves = self.results['nleaves']
        
        # Apply burn-in
        n_samples = len(scalar_chain)
        burnin_idx = int(n_samples * burnin_fraction)
        
        scalar_chain_burned = scalar_chain[burnin_idx:]
        nleaves_burned = nleaves[burnin_idx:]
        
        # Combine scalar parameters with number of EGP nodes
        samples = np.column_stack([scalar_chain_burned, nleaves_burned])
        labels = scalar_names + ['N_nodes']
        
        #print(f"Creating corner plot with {len(samples)} samples")
        #print(f"Parameters: {labels}")
        #print(f"Number of nodes - mean: {np.mean(nleaves_burned):.2f}, "
        #    f"median: {np.median(nleaves_burned):.0f}, "
        #    f"mode: {int(np.bincount(nleaves_burned.astype(int)).argmax())}")
        
        # Create corner plot
        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            truth_color='red',
            color='blue',
            bins=30,
            smooth=0.9,
            plot_datapoints=True,
            plot_density=True,
            fill_contours=True,
            levels=(0.68, 0.95),
        )

        fig.savefig(f'{get_first_part(self.config["inference"]["file_post"])}/eryn_corner.png', dpi=150)

        return fig


    def plot_scalar_only(self, burnin_fraction=0.3, figsize=(10, 10)):
        """
        Create corner plot for scalar parameters only (without N_nodes).
        
        Parameters:
        -----------
        results : dict
            Results dictionary from ERYNInference.run()
        burnin_fraction : float
            Fraction of samples to discard as burn-in
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Corner plot figure
        """
        scalar_chain = self.results['scalar_chain']
        scalar_names = self.results['scalar_names']
        
        # Apply burn-in
        n_samples = len(scalar_chain)
        burnin_idx = int(n_samples * burnin_fraction)
        scalar_chain_burned = scalar_chain[burnin_idx:]

        
        # Create corner plot
        fig = corner.corner(
            scalar_chain_burned,
            labels=scalar_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            truth_color='red',
            color='blue',
            bins=30,
            smooth=0.9,
            plot_datapoints=True,
            plot_density=True,
            fill_contours=True,
            levels=(0.68, 0.95),
        )
        
        fig.savefig(f'{get_first_part(self.config["inference"]["file_post"])}/eryn_scalar_only_corner.png', dpi=150)
        return fig


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
        - dict: Results dictionary containing posterior samples and diagnostics.
        """
        kwargs = {
            'data': self.data, 
            'freqs': self.freqs, 
            'response': self.response, 
            'sources': self.sources, 
            'dt': self.dt, 
            't1': self.t1, 
            't2': self.t2, 
            'dof': self.dof, 
            'gen2': self.gen2
        }
        method = Method(self.config, self.log_like, **kwargs)
        method.setup()
        return method.run()

    def save_results(self):
        """
        Save the posterior samples to a file.
        """
        np.savez(self.config['inference']["file_post"], **self.result)
        logger.info(f"Results saved to {self.config['inference']['file_post']}")

    def run(self):
        """
        Run the entire inference process.
        """
        self.result = self.run_inference()
        self.save_results()