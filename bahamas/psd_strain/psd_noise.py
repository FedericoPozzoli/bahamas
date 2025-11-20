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

from bahamas.psd_response.orbits import SpacecraftOrbit


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
year = 31557600.0  # Seconds in a year

@jit
def acc_noise(f, par):
    A = par['A']
    return (A*1e-15)**2 * (1 + (4e-4/f)**2) / (2*jnp.pi*f*clight)**2

@jit
def oms_noise(f, par):
    P = par['P']
    fs, fknee, fmin = 0.5, 2e-3, 1/year
    z = jnp.exp(1j*2*jnp.pi*f/fs)
    mod = jnp.abs(2*jnp.pi*fmin / (1 - jnp.exp(-2*jnp.pi*fmin/fs)*z**-1))
    return (P*1e-12)**2 * ((2*jnp.pi*fknee**2/clight)**2 * mod**2/(fs*fmin)**2 + 
                           (fs/clight)**2 * jnp.sin(2*jnp.pi*f/fs)**2)

#####################################################################################
# ARMLENGTH COMPUTATION FROM ORBITS
#####################################################################################

def compute_armlengths_from_orbit(orbit_obj, t):
    """
    Compute all six armlengths from spacecraft positions at time t.
    
    Args:
        orbit_obj: SpacecraftOrbit instance
        t: Time in seconds
        
    Returns:
        Dictionary with armlengths in millions of km: 
        {'L12', 'L21', 'L23', 'L32', 'L31', 'L13'}
    """
    # Get spacecraft positions [light-seconds]
    positions = np.zeros((len(t), 3, 3))

    for i, t in enumerate(t):
        positions[i,:,:] = orbit_obj.SC_positions(t)
    
    # Compute distances between spacecraft
    L12 = np.linalg.norm(positions[:,1,:] - positions[:,0,:], axis=1)
    L21 = L12  # Symmetric
    # L13: SC1 -> SC3
    L13 = np.linalg.norm(positions[:,2,:] - positions[:,0,:], axis=1)
    L31 = L13  # Symmetric
    # L23: SC2 -> SC3
    L23 = np.linalg.norm(positions[:,2,:] - positions[:,1,:], axis=1)
    L32 = L23  # Symmetric
    
    return {
        'L12': L12, 'L21': L21,
        'L23': L23, 'L32': L32,
        'L31': L31, 'L13': L13
    }

def compute_avg_armlengths_from_orbit(orbit_obj, t):
    """
    Compute average armlengths from orbit at time t.
    
    Args:
        orbit_obj: SpacecraftOrbit instance
        t: Time in seconds
        
    Returns:
        Tuple of (L_bar_12, L_bar_23, L_bar_31) in millions of km
    """
    armlengths = compute_armlengths_from_orbit(orbit_obj, t)
    
    L_bar_12 = (armlengths['L12'] + armlengths['L21']) / 2.0
    L_bar_23 = (armlengths['L23'] + armlengths['L32']) / 2.0
    L_bar_31 = (armlengths['L31'] + armlengths['L13']) / 2.0
    

    return jnp.mean(L_bar_12), jnp.mean(L_bar_23), jnp.mean(L_bar_31)


#####################################################################################
# EQUAL ARMLENGTH TRANSFER FUNCTIONS (Original Implementation)
#####################################################################################


@jit
def transfer_functions_acc(f, L, tdi=0, gen2=False):
    w = 2*jnp.pi*f*L
    Cxx = 8*jnp.sin(w)**2
    tr_acc_AE = Cxx * 4*(1 + 2*jnp.cos(w))
    tr_acc_T = 4*Cxx * (1 - jnp.cos(w))**2

    if lax is not None:
        tr_acc = lax.cond(
            tdi == 2,
            lambda _: tr_acc_T,
            lambda _: tr_acc_AE,
            operand=None
        )
    else:
        tr_acc = tr_acc_T if tdi == 2 else tr_acc_AE

    if lax is not None:
        tr_acc = lax.cond(gen2, lambda s: s * 4 * jnp.sin(2 * w)**2, lambda s: s, tr_acc)
    else:
        tr_acc = tr_acc * 4 * jnp.sin(2 * w)**2 if gen2 else tr_acc

    return tr_acc

@jit
def transfer_functions_oms(f, L, tdi=0, gen2=False):
    w = 2*jnp.pi*f*L
    Cxx = 8*jnp.sin(w)**2
    tr_oms_AE = Cxx * (2 + jnp.cos(w))
    tr_oms_T = 2*Cxx * (1 - jnp.cos(w))

    if lax is not None:
        tr_oms = lax.cond(
            tdi == 2,
            lambda _: tr_oms_T,
            lambda _: tr_oms_AE,
            operand=None
        )
    else:
        tr_oms = tr_oms_T if tdi == 2 else tr_oms_AE

    if lax is not None:
        tr_oms = lax.cond(gen2, lambda s: s * 4 * jnp.sin(2 * w)**2, lambda s: s, tr_oms)
    else:
        tr_oms = tr_oms * 4 * jnp.sin(2 * w)**2 if gen2 else tr_oms    

    return tr_oms


#####################################################################################
# PSD FUNCTIONS (Original equal-arm versions)
#####################################################################################

@jit
def psd_AE(f, L, A=2.4, P=7.9):
    # Accelerometer noise
    Sacc = (A*1e-15)**2 * (1 + (4e-4/f)**2) / (2*jnp.pi*f*clight)**2
    
    # OMS noise
    fs, fknee, fmin = 0.5, 2e-3, 1/year
    z = jnp.exp(1j*2*jnp.pi*f/fs)
    mod = jnp.abs(2*jnp.pi*fmin / (1 - jnp.exp(-2*jnp.pi*fmin/fs)*z**-1))
    Soms = (P*1e-12)**2 * ((2*jnp.pi*fknee**2/clight)**2 * mod**2/(fs*fmin)**2 + 
                           (fs/clight)**2 * jnp.sin(2*jnp.pi*f/fs)**2)
    
    # Transfer functions
    w = 2*jnp.pi*f*L
    Cxx = 8*jnp.sin(w)**2
    tr_acc = Cxx * 4*(1 + 2*jnp.cos(w))
    tr_oms = Cxx * (2 + jnp.cos(w))
    
    return Sacc*tr_acc + Soms*tr_oms

@jit
def psd_T(f, L, A=2.4, P=7.9):
    # Accelerometer noise
    Sacc = (A*1e-15)**2 * (1 + (4e-4/f)**2) / (2*jnp.pi*f*clight)**2
    
    # OMS noise
    fs, fknee, fmin = 0.5, 2e-3, 1/year
    z = jnp.exp(1j*2*jnp.pi*f/fs)
    mod = jnp.abs(2*jnp.pi*fmin / (1 - jnp.exp(-2*jnp.pi*fmin/fs)*z**-1))
    Soms = (P*1e-12)**2 * ((2*jnp.pi*fknee**2/clight)**2 * mod**2/(fs*fmin)**2 + 
                           (fs/clight)**2 * jnp.sin(2*jnp.pi*f/fs)**2)
    
    # Transfer functions
    w = 2*jnp.pi*f*L
    Cxx = 8*jnp.sin(w)**2
    tr_acc = 4*Cxx * (1 - jnp.cos(w))**2
    tr_oms = 2*Cxx * (1 - jnp.cos(w))
    
    return Sacc*tr_acc + Soms*tr_oms


@jit 
def noise(freq, par, tdi, gen2=False, L = 8.3):
    """
    Computes the noise power spectral density (PSD) using component functions.

    This function calculates the noise PSD for LISA by combining contributions from
    test mass noise and the optical metrology system. It supports both standard and TDI2
    (Time-Delay Interferometry 2) configurations.

    Args:
        freq (array): Frequency array in Hz.
        par (dict): Dictionary containing the parameters:
            - 'A' (float): Amplitude of the test mass noise.
            - 'P' (float): Amplitude of the optical metrology system noise.
            - 'Tobs' (float): Observation time in seconds.
        tdi (int): Time-delay interferometry channel (0 = A, 1 = E, 2 = T).
        gen2 (bool): If True, applies TDI2 factor to the PSD.
        L (float): Arm length in million km (default is 8.3 million km).

    Returns:
        array: Noise power spectral density (PSD) as a function of frequency.
    """
    A = par['A']
    P = par['P']

    # Select the appropriate noise based on TDI channel
    if lax is not None:
        s_n = lax.cond(
        tdi == 2,
        lambda _: psd_T(freq, L, A, P),
        lambda _: psd_AE(freq, L, A, P),
        operand=None
        )  
    else:
        s_n = psd_T(freq, L, A, P) if tdi == 2 else psd_AE(freq, L, A, P)
    
    # Apply TDI2 factor if specified    
    omega = 2.0 * jnp.pi * freq
    

    x = omega * L
    factor_tdi2 = 4 * jnp.sin(2 * x)**2
    if lax is not None:
        # Use lax.cond for JAX compatibility
        s_n = lax.cond(gen2, lambda s: s * factor_tdi2, lambda s: s, s_n)
    else:
        # Fallback for non-JAX environments
        s_n = s_n * factor_tdi2 if gen2 else s_n
    return s_n




