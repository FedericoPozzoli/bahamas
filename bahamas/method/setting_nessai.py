"""
This module provides likelihood functions and a custom model class for use with the Nessai nested sampling framework.

Functions:
    whittle_lik(sample, data, ...): Computes the Whittle likelihood for a given sample.
    gamma_lik(sample, data, ...): Computes the Gamma likelihood for a given sample.

Classes:
    nessai_model: Custom model class for Nessai that handles likelihood and prior evaluations.

Dependencies:
    - NumPy
    - SciPy
    - Nessai
    - psd_function (custom module)
"""
from bahamas.psd_strain import psd_function as psd

import numpy as np
import scipy as sc
import nessai
import nessai.model



def whittle_lik(sample, data, freqs, response, count, dt, t1, t2, dof,gen2):
    """
    Computes the Whittle likelihood for a given sample.

    Args:
        sample (dict): Sampled parameters.
        data (list): Observed data segments.
        freqs (list): Frequency grids for each segment.
        response (list): Response functions for each segment.
        count (list): Counts for each segment.
        dt (float): Time step.
        t1, t2 (list): Start and end times for each segment.
        dof (list): Degrees of freedom for each segment.
        matrix_egp (array): Matrix for EGP modeling.

    Returns:
        float: Log-likelihood value.
    """
    log_likelihood = 0
    for j, segment in enumerate(data):
        for i, tdi in enumerate(segment):
            f, n = np.array(freqs[j]), dof[j]
            psd_model = psd.model_psd(
                freqs=f, response=response[j][i], sources=sample, t1=t1[j], t2=t2[j], tdi=i, gen2=gen2
            )
            psd_model = (n / dt) * psd_model
            log_likelihood += (
                -0.5 * np.sum((np.abs(tdi) ** 2) / psd_model)
                - 0.5 * len(f) * np.log(2 * np.pi)
                - 0.5 * np.sum(np.log(psd_model))
            )
    return log_likelihood


def gamma_lik(sample, data, freqs, response, count, dt, t1, t2, dof, gen2):
    """
    Computes the Gamma likelihood for a given sample.

    Args:
        sample (dict): Sampled parameters.
        data (list): Observed data segments.
        freqs (list): Frequency grids for each segment.
        response (list): Response functions for each segment.
        count (list): Counts for each segment.
        dt (float): Time step.
        t1, t2 (list): Start and end times for each segment.
        dof (list): Degrees of freedom for each segment.

    Returns:
        float: Log-likelihood value.
    """
    log_likelihood = 0
    for j, segment in enumerate(data):
        for i, tdi in enumerate(segment):
            f = np.array(freqs[j])
            psd_model = psd.model_psd(
                freqs=f, response=response[j][i], sources=sample, t1=t1[j], t2=t2[j], tdi=i, gen2=gen2
            ) / count[j]
            log_likelihood += (
                -np.sum(sc.special.gammaln(count[j]))
                - np.sum(count[j] * np.log(psd_model))
                + np.sum((count[j] - 1) * np.log(tdi))
                - np.sum(tdi / psd_model)
            )
    return log_likelihood


class nessai_model(nessai.model.Model):
    """
    Custom model class for Nessai that handles likelihood and prior evaluations.

    Attributes:
        log_like_func (callable): The likelihood function.
        sources (dict): Source parameters.
        like_kwargs (dict): Additional arguments for the likelihood function.
        names (list): List of parameter names.
        bounds (dict): Dictionary of parameter bounds.
    """

    def __init__(self, log_like_func, **kwargs):
        """
        Initialize the Nessai model.

        Args:
            log_like_func (callable): The likelihood function.
            **kwargs: Additional arguments for the likelihood function.
        """
        self.log_like_func = log_like_func
        self.sources = kwargs.pop('sources', None)
        self.like_kwargs = kwargs

        # Extract parameter names and bounds
        self.names = [item['name'] for category in self.sources.values() for item in category]
        self.bound = [item['bounds'] for category in self.sources.values() for item in category]
        self.bounds = {self.names[i]: self.bound[i] for i in range(len(self.names))}
        self.logprior_volume = -np.sum(
            [np.log(self.bounds[name][1] - self.bounds[name][0]) for name in self.names])

    def log_likelihood(self, livepoint):
        """
        Evaluate the log-likelihood for a given live point.

        Args:
            livepoint (dict or structured array): Live point(s) to evaluate.

        Returns:
            float or np.ndarray: Log-likelihood value(s).
        """
        point = np.array([livepoint[p] for p in self.names])
            
        ll = np.zeros(livepoint.size)
        
        if livepoint.ndim == 0:
            samples = {}
            for source_name, param_list in self.sources.items():
                source_samples = {
                    param["name"]: livepoint[param['name']]
                    for param in param_list
                }
                samples[source_name] = source_samples
            ll = self.log_like_func(sample = samples, **self.like_kwargs)

        else: 
            for i in range(livepoint.size):
                samples = {}
                for source_name, param_list in self.sources.items():
                    source_samples = {
                        param["name"]: livepoint[param['name']]
                        for param in param_list
                    }
                    samples[source_name] = source_samples        
                ll[i] = self.log_like_func(sample = samples, **self.like_kwargs)
                
        return ll

    def log_prior(self, livepoint):
        """
        Evaluate the log-prior for a given live point.

        Args:
            livepoint (dict or structured array): Live point(s) to evaluate.

        Returns:
            float or np.ndarray: Log-prior value(s).
        """
        if not self.in_bounds(livepoint).any():
            return -np.inf  # Discard points outside bounds
        # Create an array with as many elements as the number of livepoints
        log_prior = self.logprior_volume*np.ones(livepoint.size)
        # Evaluate the joint prior
        return  log_prior #np.log(self.in_bounds(livepoint))