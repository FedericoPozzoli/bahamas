"""Module for generating power spectral density (PSD) models for gravitational wave signals.
This module includes various signal models, noise models, and functions to compute the total PSD
from multiple sources. It also provides functions for generating Gaussian processes and averaging data.
The module uses JAX for efficient computation and supports both time and frequency domain analysis.
"""
from bahamas.backend_context import get_backend_components
# Load backend components after backend initialization
jnp, jit, lax = get_backend_components()


# Check if lax is available
if jnp is None:
    #default to jax numpy
    from bahamas.backend_context import initialize_backend
    initialize_backend(use_jax=True)
    jnp, jit, lax = get_backend_components()

if lax is not None:
    import jax
    jax.config.update('jax_enable_x64', True)

from bahamas.psd_response import average_envelope as env
from bahamas.psd_strain import egp

from scipy.signal import welch
from astropy.cosmology import Planck18
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
Planck18_H0 = Planck18.H0.si.value
H02 = Planck18_H0**2
clight = 299792458.0  # Speed of light in m/s
L = 8.3  # Arm length of LISA in seconds
year = 31557600.0  # Seconds in a year

import astropy.constants as const

msun = const.M_sun.si.value
G = const.G.value
Mpc = const.pc.value*1e6

##################################################################################
#SIGNAL MODELS
##################################################################################
@jit
def Omega_extra_foreground(freqs, par):    
    """
    Extra galactic foreground of white dwarf binaries.
    https://arxiv.org/abs/2407.10642
    
    Args:
        freqs (array): Frequency array.
        par (dict): Parameters for the model.
    Returns:
        array: Power spectral density.
    """
    a = par['a']
    b = 4.1875207208364635 #par['b']
    gamma1 = par['gamma1']
    gamma2 = par['gamma2']

    f_b = 7.2e-3

    Omega =  10**a*(freqs/f_b)**gamma1 * (1 + (freqs/f_b)**4.15)**gamma2 * jnp.exp(-10**b*freqs**3)
    S_h = Omega* (3*H02) / (4*jnp.pi**2*freqs**3)
	
    return S_h

@jit
def Omega_gaussian_bump(freqs, par):
    """
    Power spectral density for gravitational waves modeled as a Gaussian bump.

    Args:
        freqs (array): Frequency array.
        par (dict): Parameters for the model.
    Returns:
        array: Power spectral density.
    """
    A = par['Amp']
    fp = par['fp']
    sigma = par['sigma']

    # Compute the Omega formula
    Omega = 10**A * jnp.exp(-1 / (2 * sigma**2) * jnp.log(freqs / 10**fp)**2)
    S_h = Omega * (3 * H02) / (4 * jnp.pi**2 * freqs**3)

    return S_h

@jit
def Omega_phase_transition(freqs, par):
    """
    Power spectral density for gravitational waves from phase transitions.

    Args:
        freqs (array): Frequency array.
        par (dict): Parameters for the model.
    Returns:
        array: Power spectral density.
    """
    Amp = par['Amp']
    fp = par['fp']
    n = par['n']

    # Compute the Omega formula
    Omega = 10**Amp * (freqs / 10**fp)**3 * (7 / (4 + 3 * (freqs / 10**fp)**2))**n
    S_h = Omega * (3 * H02) / (4 * jnp.pi**2 * freqs**3)

    return S_h

@jit
def Omega_pl(freqs, par):
    """"
    Power-law model for gravitational wave signal.

    Args:
        freqs (array): Frequency array.
        par (dict): Parameters for the model.
        
    Returns:
        array: Power spectral density.
    """

    Amp = par['Amp']
    slope = par['slope']

    f0 = 10**-2.5
    Omega = 10**Amp*(freqs/ f0)**slope
    S_h = Omega * (3*H02)/ (4*jnp.pi**2*freqs**3)

    
    return S_h



@jit
def Omega_DF(freqs, par):
    """
    Dynamical friction model for gravitational wave signal with a bump.
    Args:
        freqs (array): Frequency array.
        par (dict or array): Parameters for the model.
        posterior (bool): If True, uses a different parameterization.
    Returns:
        array: Power spectral density.
    """
    A_m = -13.564594467605877


    Avac = par['Avac']
    rho = par['rho']

    numerator = 10**Avac * freqs**(2./3)
    denominator = (1 - 10**rho * 10**A_m * freqs**(-11./3)*jnp.log(freqs)) 
    log_f_peak = -3.4059 + 0.2623 * rho 
    amplitude = 1.2034 + (-0.0262 * rho)
    sigma = 0.2217
    Gauss = amplitude * jnp.exp(-((jnp.log10(freqs) - log_f_peak)**2) / (2 * sigma**2))
    Gauss_correction = 1 + Gauss
    omega =  numerator / denominator / Gauss_correction

    S_h = omega * (3*H02) / (4*jnp.pi**2*freqs**3) 
    return S_h
