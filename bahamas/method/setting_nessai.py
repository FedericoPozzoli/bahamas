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



def whittle_lik(sample, data, freqs, response, dt, t1, t2, dof,gen2):
    """
    Computes the Whittle likelihood for a given sample.

    Args:
        sample (dict): Sampled parameters.
        data (list): Observed data segments.
        freqs (list): Frequency grids for each segment.
        response (list): Response functions for each segment.
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


def gamma_lik(sample, data, freqs, response, dt, t1, t2, dof, gen2):
    """
    Computes the Gamma likelihood for a given sample.

    Args:
        sample (dict): Sampled parameters.
        data (list): Observed data segments.
        freqs (list): Frequency grids for each segment.
        response (list): Response functions for each segment.
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
            #psd_model = psd.model_psd(freqs=f, response=response[j][i], sources=sample, t1=t1[j], t2=t2[j], tdi=i, gen2 = gen2) / dof[j]
            #log_likelihood += (-np.sum(sc.special.gammaln(dof[j]))- np.sum(dof[j] * np.log(psd_model)) + np.sum((dof[j] - 1) * np.log(tdi))- np.sum(tdi / psd_model))
            psd_model = psd.model_psd(freqs=f, response=response[j][i], sources=sample, t1=t1[j], t2=t2[j], tdi=i, gen2 = gen2) #/ dof[j]
            log_likelihood += (-np.sum(sc.special.gammaln(0.5*dof[j]))- np.sum(0.5* dof[j] * np.log(psd_model)) + np.sum((0.5* dof[j] - 1) * np.log(tdi))- np.sum(0.5* dof[j] * tdi / psd_model))
    return log_likelihood


class nessai_model(nessai.model.Model):
    """
    Custom model class for Nessai that handles likelihood and prior evaluations,
    with support for fixed (injected) parameters when no bounds are provided.
    """

    def __init__(self, log_like_func, **kwargs):
        self.log_like_func = log_like_func
        self.sources = kwargs.pop('sources', None)
        self.like_kwargs = kwargs

        # Split parameters into free (with bounds) and fixed (with injected values)
        self.free_params = []
        self.fixed_params = {}
        for category in self.sources.values():
            for param in category:
                if param.get("bounds") is not None:
                    self.free_params.append(param)
                else:
                    self.fixed_params[param["name"]] = param["injected"]

        # Build nessai interface
        self.names = [p["name"] for p in self.free_params]
        self.bounds = {p["name"]: p["bounds"] for p in self.free_params}

        # Prior volume for uniform priors
        self.logprior_volume = -np.sum(
            [np.log(self.bounds[name][1] - self.bounds[name][0]) for name in self.names]
        )

    def log_likelihood(self, livepoint):
        """
        Evaluate the log-likelihood for a given live point.
        """
        ll = np.zeros(livepoint.size) if livepoint.ndim > 0 else 0.0

        def build_samples(lp):
            samples = {}
            for source_name, param_list in self.sources.items():
                source_samples = {}
                for param in param_list:
                    if param.get("bounds") is not None:
                        source_samples[param["name"]] = lp[param["name"]]
                    else:
                        source_samples[param["name"]] = param["injected"]
                samples[source_name] = source_samples
            return samples

        if livepoint.ndim == 0:
            samples = build_samples(livepoint)
            ll = self.log_like_func(sample=samples, **self.like_kwargs)
        else:
            for i in range(livepoint.size):
                samples = build_samples(livepoint[i])
                ll[i] = self.log_like_func(sample=samples, **self.like_kwargs)

        return ll

    def log_prior(self, livepoint):
        """
        Evaluate the log-prior for a given live point.
        """
        if not self.in_bounds(livepoint).any():
            return -np.inf
        return self.logprior_volume * np.ones(livepoint.size)
