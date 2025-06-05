"""
This module contains functions for sampling and likelihood computation in a Bayesian analysis framework.
It includes functions for sampling parameters from prior distributions and computing likelihoods using the Whittle and Gamma methods.
It is designed to work with the BAHAMAS framework for analyzing gravitational wave signals.

Functions:
    prior_sample(sources): Samples parameters from prior distributions for multiple sources.
    whittle_lik(data, freqs, response, count, sources, ...): Computes the Whittle likelihood.
    gamma_lik(data, freqs, response, count, sources, ...): Computes the Gamma likelihood.
    beta_scaled_log_likelihood(log_like_fn, beta): Scales a log-likelihood by a beta factor.

Dependencies:
    - JAX
    - NumPyro
    - psd_function (custom module)
"""
from bahamas.psd_strain import psd_function as psd


import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsc
import numpyro
import numpyro.distributions as dist
from jax import lax, random, vmap
from jax.scipy.special import logsumexp


####################################################################################################

# Prior Sampling
def prior_sample(sources):
    """
    Sample parameters from prior distributions for multiple sources.

    Parameters:
    - sources (dict): Dictionary of sources and their parameter configurations.

    Returns:
    - dict: Dictionary of sampled parameters for each source.
    """
    samples = {}
    for source_name, param_list in sources.items():

        source_samples = {
                param["name"]: numpyro.sample(f"{source_name}_{param['name']}", dist.Uniform(*param["bounds"]))
                if param["bounds"] else param["injected"]
                for param in param_list
            }
        samples[source_name] = source_samples
    return samples

# Likelihood Functions
def whittle_lik(data, freqs, response, sources, count, dt, t1, t2, dof, gen2):
    """
    Compute the Whittle likelihood.

    Parameters:
    - data (list): Observed data segments.
    - freqs (list): Frequency grids for each segment.
    - response (list): Response functions for each segment.
    - count (list): Counts for each segment.
    - sources (dict): Source parameters.
    - dt (float): Time step.
    - t1, t2 (list): Start and end times for each segment.
    - dof (list): Degrees of freedom for each segment.

    Returns:
    - None: Adds the log-likelihood factor to the NumPyro model.
    """
    sample = prior_sample(sources)
    log_likelihood = 0
    for j, segment in enumerate(data):
        for i, tdi in enumerate(segment):
            f, n = jnp.array(freqs[j]), dof[j]
            psd_model = psd.model_psd(freqs=f, response=response[j][i], sources=sample, t1=t1[j], t2=t2[j], tdi=i, gen2 = gen2)
            psd_model = (n / dt) * psd_model
            log_likelihood += -0.5 * jnp.sum((jnp.abs(tdi) ** 2) / psd_model) - 0.5 * len(f) * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(jnp.log(psd_model))
    numpyro.factor("log_likelihood", log_likelihood)

def gamma_lik(data, freqs, response, sources, count, dt, t1, t2, dof, gen2):
    """
    Compute the Gamma likelihood.

    Parameters:
    - data (list): Observed data segments.
    - freqs (list): Frequency grids for each segment.
    - response (list): Response functions for each segment.
    - count (list): Counts for each segment.
    - sources (dict): Source parameters.
    - dt (float): Time step.
    - t1, t2 (list): Start and end times for each segment.
    - dof (list): Degrees of freedom for each segment.

    Returns:
    - None: Adds the log-likelihood factor to the NumPyro model.
    """
    sample = prior_sample(sources)
    log_likelihood = 0
    for j, segment in enumerate(data):
        for i, tdi in enumerate(segment):
            f = jnp.array(freqs[j])
            psd_model = psd.model_psd(freqs=f, response=response[j][i], sources=sample, t1=t1[j], t2=t2[j], tdi=i, gen2 = gen2) / count[j]
            log_likelihood += -jnp.sum(jsc.special.gammaln(count[j])) - jnp.sum(count[j] * jnp.log(psd_model)) + jnp.sum((count[j] - 1) * jnp.log(tdi)) - jnp.sum(tdi / psd_model)
    numpyro.factor("log_likelihood", log_likelihood)

def beta_scaled_log_likelihood(log_like_fn, beta):
    """
    Scale a log-likelihood function by a beta factor.

    Parameters:
    - log_like_fn (callable): Log-likelihood function.
    - beta (float): scaling factor.

    Returns:
    - callable: Scaled log-likelihood function.
    """
    def scaled_likelihood(data, freqs, response, sources, count, dt, t1, t2, dof, gen2):
        with numpyro.handlers.scale(scale=beta): 
            return log_like_fn(data, freqs, response, sources, count, dt, t1, t2, dof, gen2)
    return scaled_likelihood
