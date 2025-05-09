"""Module for generating power spectral density (PSD) models for gravitational wave signals.
This module includes various signal models, noise models, and functions to compute the total PSD
from multiple sources. It also provides functions for generating Gaussian processes and averaging data.
The module uses JAX for efficient computation and supports both time and frequency domain analysis.
"""
from bahamas.backend_context import get_backend_components
# Load backend components after backend initialization
jnp, jit, lax = get_backend_components()

print(f"np is from: {jnp.__name__}")
print(f"jit is from: {jit}")
# Check if lax is available
if lax is not None:
    print(f"lax is from: {lax}")
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

##################################################################################
#SIGNAL MODELS
##################################################################################
@jit
def Omega_vacuum(freqs, par):
    """"
    Vacuum model for gravitational wave signal.
    Args:
        freqs (array): Frequency array.
        par (dict): Parameters for the model.
        
    Returns:
        array: Power spectral density.
    """

    Avac = par['Avac']
    slope = par['slope']

    f0 = 1 
    Omega = 10**Avac*(freqs/ f0)**slope
    S_h = Omega * (3*H02)/ (4*jnp.pi**2*freqs**3)

    return S_h


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
    b = par['b']
	
    gamma1 = 0.741
    f_b = 7.2e-3
    gamma2 = -0.255
    gamma3 = 3 

    Omega =  10**a*(freqs/f_b)**gamma1 * (1 + (freqs/f_b)**4.15)**gamma2 * jnp.exp(-10**b*freqs**gamma3)
    S_h = Omega* (3*H02) / (4*jnp.pi**2*freqs**3)
	
    return S_h

@jit
def galactic_foreground(freqs, par, injected = False):
    """
    Galactic foreground of white dwarf binaries.
    https://arxiv.org/abs/2103.14598
    Args:
        freqs (array): Frequency array.
        par (dict): Parameters for the model.
        injected (bool): If True, uses injected parameters.
    Returns:
        array: Power spectral density.
    """     

    if 'fknee' not in par:
        Tobs = par['Tobs'] / year
        
        
        alpha = par['alpha']
        amp = par['amp']
        a1 = par['a1']
        b1 = par['b1']
        ak = par['ak']
        bk = par['bk']
        fr2 = par['fr2']
        
        fr1 = a1 * jnp.log10(Tobs) + b1
        fknee = ak * jnp.log10(Tobs) + bk
    else:
        alpha = par['alpha']
        amp = par['amp']
        fknee = par['fknee']
        fr1 = par['fr1']
        fr2 = par['fr2']
        
    
    res = 10**amp*jnp.exp(-(freqs/10**fr1)**alpha) *\
        (freqs**(-7./3.))*0.5*(1.0 + jnp.tanh(-(freqs-10**fknee)/10**fr2))
    
    fstar = 1 / L
    x = 2*jnp.pi * (freqs / fstar) * jnp.sin(2*jnp.pi * (freqs / fstar))
    response = 4 * x**2 
    psd = res*response
    return psd


@jit
def galactic_foreground_time(freqs, par, injected=False, t1=0, t2=0, tdi=0):
    """
    Galactic foreground model for gravitational wave signal with time dependence.
    https://arxiv.org/abs/2410.08274, https://arxiv.org/abs/2410.08263
    Args:
        freqs (array): Frequency array.
        injected (bool): If True, uses injected parameters.
        par (dict): Parameters for the model.
        t1 (float): Start time for the envelope.
        t2 (float): End time for the envelope.
        tdi (int): TDI channel (0 = A, 1 = E).
    Returns:
        array: Power spectral density.
    """
        
    if 'fknee' not in par:
        Tobs = par['Tobs'] / year
        
        alpha = par['alpha']
        amp = par['amp']
        a1 = par['a1']
        b1 = par['b1']
        ak = par['ak']
        bk = par['bk']
        fr2 = par['fr2']
        lat = par['lat']
        long = par['long']
        psi = par['psi']
        s1 = par['s1']
        s2 = par['s2']
        
        fr1 = a1 * jnp.log10(Tobs) + b1
        fknee = ak * jnp.log10(Tobs) + bk
    else:
        alpha = par['alpha']
        amp = par['amp']
        fknee = par['fknee']
        fr1 = par['fr1']
        fr2 = par['fr2']
        lat = par['lat']
        long = par['long']
        psi = par['psi']
        s1 = par['s1']
        s2 = par['s2']

    amp_time = env.average_envelopes_gaussian(
        lat, long, s1, s2, psi, t1, t2, 
        LISA_Orbital_Freq=1 / year, alpha0=0., beta0=0., tdi=tdi
    )

    res = (
        amp_time * 10**amp * jnp.exp(-(freqs / 10**fr1)**alpha) *
        (freqs**(-7./3.)) * 0.5 * (1.0 + jnp.tanh(-(freqs - 10**fknee) / 10**fr2))
    )

    fstar = 1 / L
    x = 2 * jnp.pi * (freqs / fstar) * jnp.sin(2 * jnp.pi * (freqs / fstar))
    
    psd = res * x**2
    return psd

##################################################################################NOISE
@jit
def Stm(f, A):
    """
    Test mass noise model. True value of A is 3.
    Args:
        f (array): Frequency array.
        A (float): Amplitude parameter.
    Returns:
        array: Power spectral density.
    """
    
    return A**2*10**(-30)*(1 + (0.4*10**-3/f)**2)*(1 + (f/(8*10**-3))**4)*(1/(2*jnp.pi*f*clight)**2) 

@jit
def Soms(f, P):
    """
    Optical metrology noise model. True value of P is 15.
    Args:
        f (array): Frequency array.
        P (float): Amplitude parameter.
    Returns:
        array: Power spectral density.
    """
 
    return P**2*10**(-24)*(1 + (2*10**-3/f)**4)*(2*jnp.pi*f/clight)**2

@jit
def noise(freqs, par):
    """
    Instrumental noise model in AE channel.
    https://arxiv.org/abs/2211.02539
    Args:
        freqs (array): Frequency array.
        par (dict): Parameters for the model.
        
    Returns:
        array: Power spectral density.
    """

    A = par['A']
    P = par['P']

    psd  = 8*jnp.sin(2*jnp.pi*freqs*L)**2*(Soms(freqs, P)*(jnp.cos(2*jnp.pi*freqs*L) + 2) 
                                          + 2*(3 + 2*jnp.cos(2*jnp.pi*freqs*L) + jnp.cos(4*jnp.pi*freqs*L))*Stm(freqs, A))
    
    return psd


##################################################################################

def model_psd(freqs, sources, response, injected=False, tdi=0, **kwargs):
    """
    Computes the total power spectral density (PSD) by combining multiple sources.

    Args:
        freqs (array): Frequency array.
        sources (dict): Dictionary of sources with their parameters.
        response (float): Response factor to scale the PSD.
        injected (bool, optional): If True, uses injected parameters for sources. Defaults to False.
        tdi (int, optional): Time-delay interferometry channel (0 = A, 1 = E). Defaults to 0.
        **kwargs: Additional arguments for specific sources (e.g., `matrix_egp`, `t1`, `t2`).

    Returns:
        array: Total PSD if `injected` is False.
        tuple: Total PSD and list of true PSDs for each source if `injected` is True.
    """
    if injected:
        # Convert injected parameters into a dictionary format
        dict1 = {}
        for source_name, param_list in sources.items():
            source = {param["name"]: param["injected"] for param in param_list}
            dict1[source_name] = source
        sources = dict1

    # Initialize PSD arrays
    psd = jnp.zeros(len(freqs))
    true_psd = []

    # Loop through each source and compute its contribution to the PSD
    for source_name in sources.keys():

        if source_name == 'egp':
            matrix = kwargs.get('matrix_egp')
            psd += response * egp.egp(freqs, sources[source_name], matrix)

        elif source_name == 'extra_DWD':
            psd += response * Omega_extra_foreground(freqs, sources[source_name])
            if injected:
                true_psd.append(response * Omega_extra_foreground(freqs, sources[source_name]))

        elif source_name == 'galactic_DWD':
            psd += galactic_foreground(freqs, sources[source_name], injected=injected)
            if injected:
                true_psd.append(galactic_foreground(freqs, sources[source_name], injected=injected))

        elif source_name == 'galactic_DWD_time':
            t1 = kwargs.get('t1')
            t2 = kwargs.get('t2')
            psd += galactic_foreground_time(freqs, sources[source_name], t1=t1, t2=t2, tdi=tdi, injected=injected)
            if injected:
                true_psd.append(galactic_foreground_time(freqs, sources[source_name], t1=t1, t2=t2, tdi=tdi, injected=injected))

        elif source_name == 'instr_noise':
            psd += noise(freqs, sources[source_name])
            if injected:
                true_psd.append(noise(freqs, sources[source_name]))

    # Return the total PSD and optionally the true PSDs for each source
    if injected:
        return psd, true_psd
    else:
        return psd