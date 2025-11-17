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


#--------------------new for ecc------------------
@jit
def _compute_fp_jnp(e, f):
    term = (e**(12.0/19.0)) / (1.0 - e**2) * (1.0 + 121.0/304.0 * e**2)**(870.0/2299.0)
    return (1293.0/181.0) * (term**1.5) * f

@jit
def _hc_squared_circular_jnp(fobs, z0, M0):
    prefactor = (4 * G * (jnp.pi * G)**(2/3)) / (3 * jnp.pi * clight**2)
    normalization_factor = Mpc**3
    return prefactor * M0**(5/3) * (1 + z0)**(-1/3) * fobs**(-4/3) / normalization_factor

@jit
def _h_c_fit_jnp(f_obs):
    x0 = 5.531e+00
    q = 2.942e+00
    Al = 6.946e+00
    xl = 2.774e-01
    pl = -6.217e-01
    Ah = 4.118e+00
    xh = 4.483e+00
    ph = 8.503e-01

    z = 0.02
    M0 = 4.16 * 1e8 * msun
    x = f_obs/1e-9
    low = (x / x0)**q / (1.0 + (x / x0)**q)
    bumpL = 1.0 + Al * (x / xl) ** pl * jnp.exp(-x / xl)
    bumpH = 1.0 + Ah * (x / xh) ** ph * jnp.exp(-x / xh)
    S_shape = low * bumpL* bumpH
    hc_circ = _hc_squared_circular_jnp(f_obs, z, M0) 
    return jnp.sqrt(hc_circ * S_shape)

@jit
def _hc_general_jnp(fobs, et, ft, e0_generate=0.9, f0_generate=1e-10):
    fp0 = _compute_fp_jnp(e0_generate, f0_generate)
    fpt = _compute_fp_jnp(et, ft)

    f_rescaled = fobs * (fp0 / fpt)
    hc0 = _h_c_fit_jnp(f_rescaled)
    return hc0 * (fpt / fp0)**(-2.0/3.0)

@jit
def _hc_squared_integrated_jnp(fobs, z0, M0, e0, fp0):
    M_0 = 4.16e8 * msun
    z_0 = 0.02
    hc_fit_sq = _hc_general_jnp(fobs, e0, fp0)**2
    amp_factor = (M0 / M_0)**(5.0/3.0) * ((1.0 + z0) / (1.0 + z_0))**(-1.0/3.0)
    return hc_fit_sq * amp_factor

@jit
def _hc_piecewise(fobs, z0, M0, e0, fp0):
    """
    Piecewise characteristic strain squared:
      - if e0 == 0:   use circular model
      - if e0 != 0:   use integrated eccentric model
    """
    if lax is None:
        return (_hc_squared_circular_jnp if float(e0) == 0.0
                else _hc_squared_integrated_jnp)(fobs, z0, M0, e0, fp0)

    e0j  = jnp.asarray(e0)

    def branch_circular(_):
        return _hc_squared_circular_jnp(fobs, z0, M0)

    def branch_eccentric(_):
        return _hc_squared_integrated_jnp(fobs, z0, M0, e0j, fp0)

    return lax.cond(e0j == 0.0, branch_circular, branch_eccentric, operand=None)

@jit
def _omega_gwb_jnp(fobs, z0, M0, e0, fp0 = 1e-4):
    hc2 = _hc_piecewise(fobs, z0, M0, e0, fp0)
    return (2.0 * jnp.pi**2 * fobs**2 * hc2) / (3.0 * H02)

@jit
def _omega_gwb_with_pop_jnp(fobs, Avac, e0, fp0 = 1e-4):
    M0 = 4.16e8 * msun
    z0 = 0.02
    omega_GW_scaled_value =  Avac 
    return _omega_gwb_jnp(fobs, z0, M0, e0, fp0) * omega_GW_scaled_value

@jit
def Omega_ecc_pop(freqs, par):
    """
    Eccentric binary population model parameterized by (e0, log10f0),
    returning S_h(f).

    Parameters:
        freqs (array): Frequency array in Hz.
        par (dict): Dictionary containing:
            - log10Avac : log10 of amplitude scaling factor
            - e0        : initial eccentricity (defined at f0)
            - log10f0   : (optional) log10 of orbital frequency f0 in Hz
                          (e.g., f0 = 1e-4 corresponds to log10f0 = -4)

    Returns:
        array: Power spectral density S_h(f).
    """
    # Extract parameters
    log10Avac = par['Amp']
    Avac = 10.0 ** log10Avac
    e0 = par['e0']

    # Compute Omega_gw using the population model
    Omega = _omega_gwb_with_pop_jnp(freqs, Avac, e0)

    # Convert Omega -> S_h
    S_h = Omega * (3.0 * H02) / (4.0 * jnp.pi**2 * freqs**3)
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