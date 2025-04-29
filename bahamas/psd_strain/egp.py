"""
This module contains the implementation of the Expectation value of
Gaussian Process (GP) model for stochastic gravitational wave background (SGWB) analysis. 
"""
import jax
import jax.numpy as jnp
from astropy.cosmology import Planck18

def GP_SGWB_kernel_SE(x1, x2, parameter, eps=None):
    """
    Computes the covariance matrix using the squared exponential kernel.
    Args:
        x1 (array): First input array.
        x2 (array): Second input array.
        parameter (list): List of kernel parameters.
        eps (float, optional): Regularization term. Defaults to None.
    Returns:
        array: Covariance matrix.
    """
    x1 = jnp.asarray(x1)
    x2 = jnp.asarray(x2)
    
    """
    We could include also other types of kernels.
    """
    # Radial basis function
    tau = (x1[:, jnp.newaxis] - x2[jnp.newaxis, :])
    K = jnp.exp(-0.5 * (tau / parameter[0])**2)
   

    # Add regularization term if specified
    if eps is not None:
        assert len(x1) == len(x2), "Covariance matrix is not square"
        K += eps * jnp.eye(len(x1))

    return K


# Constants
Planck18_H0 = Planck18.H0.si.value
H02 = Planck18_H0**2

def egp(freqs, par, matrix, posterior = False):
    """
    Computes the power spectral density (PSD) using the EGP model.
    Args:
        freqs (array): Frequency array.
        par (dict or list): Model parameters.
        matrix (array): Covariance matrix.
        posterior (bool, optional): If True, use posterior parameters. Defaults to False.
    Returns:
        array: Power spectral density.
    """ 
    
    if posterior: 
        amp = par[0]
        slope = par[1]
        value = par[2:]
    
    else:
        amp = par['base_amp']
        slope = par['base_slope']
        value = jnp.array([value for key, value in par.items() if key.startswith('delta_')])
    
    log_mu2 = amp + slope*jnp.log10(freqs / 10**-3)

    

    #matrix = jnp.dot(k_star, inv_cov)  # Shape: (a, j)
    matrix = jnp.array(matrix)
    prod = jnp.sum(matrix * value, axis=1)  # Shape: (a)

    # Using jax.numpy for array operations
    mu21 = log_mu2 + prod
    psd = 10**mu21 * (3*H02) / (4*jnp.pi**2*freqs**3)
    
    return psd 


