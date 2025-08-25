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

from scipy.signal import welch
from astropy.cosmology import Planck18
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
print(lax)
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

    f0 = 1e-3
    Omega = 10**Amp*(freqs/ f0)**slope
    S_h = Omega * (3*H02)/ (4*jnp.pi**2*freqs**3)

    
    return S_h


@jit
def galactic_foreground(freqs, par, injected = False, gen2=False):
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
        
    
    Sh = 10**amp*jnp.exp(-(freqs/10**fr1)**alpha) *\
        (freqs**(-7./3.))*0.5*(1.0 + jnp.tanh(-(freqs-10**fknee)/10**fr2))
    
    # LISA arm length and angular frequency
    omega = 2.0 * jnp.pi * freqs
    x = omega * L
    tr = (x) ** 2 * jnp.sin(x)**2
    Sh *= tr

    factor_tdi2 = 4 * jnp.sin(2 * x)**2

    if lax is not None:
        # Use lax.cond for JAX compatibility
        Sh = lax.cond(gen2, lambda s: s * factor_tdi2, lambda s: s, Sh)
    else:
        # Fallback for non-JAX environments
        Sh = Sh * factor_tdi2 if gen2 else Sh

    return Sh



@jit
def galactic_foreground_time(freqs, par, injected=False, t1=0, t2=0, tdi=0, gen2=False):
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

    Sh = (
        amp_time * 10**amp * jnp.exp(-(freqs / 10**fr1)**alpha) *
        (freqs**(-7./3.)) * 0.5 * (1.0 + jnp.tanh(-(freqs - 10**fknee) / 10**fr2))
    )

    # LISA arm length and angular frequency
    omega = 2.0 * jnp.pi * freqs
    x = omega * L
    tr = (x) ** 2 * jnp.sin(x)**2
    Sh *= tr

    factor_tdi2 = 4 * jnp.sin(2 * x)**2

    if lax is not None:
        # Use lax.cond for JAX compatibility
        Sh = lax.cond(gen2, lambda s: s * factor_tdi2, lambda s: s, Sh)
    else:
        # Fallback for non-JAX environments
        Sh = Sh * factor_tdi2 if gen2 else Sh

    return Sh


#@jit
#def bulge_time(freqs, par, injected=False, t1=0, t2=0, tdi=0, gen2=False):
#    """
#    Galactic foreground model for gravitational wave signal with time dependence.
#    https://arxiv.org/abs/2410.08274, https://arxiv.org/abs/2410.08263
#
#    Args:
#        freqs (array): Frequency array.
#        injected (bool): If True, uses injected parameters.
#        par (dict): Parameters for the model.
#        t1 (float): Start time for the envelope.
#        t2 (float): End time for the envelope.
#        tdi (int): TDI channel (0 = A, 1 = E).
#    Returns:
#        array: Power spectral density.
#    """
#        
#    if 'fknee' not in par:
#        Tobs = par['Tobs'] / year
#        
#        alpha = par['alpha']
#        amp = par['amp']
#        a1 = par['a1']
#        b1 = par['b1']
#        ak = par['ak']
#        bk = par['bk']
#        fr2 = par['fr2']
#        lat = par['lat']
#        long = par['long']
#        psi = par['psi']
#        s1 = par['s1']
#        s2 = par['s2']
#        
#        fr1 = a1 * jnp.log10(Tobs) + b1
#        fknee = ak * jnp.log10(Tobs) + bk
#    else:
#        alpha = par['alpha']
#        amp = par['amp']
#        fknee = par['fknee']
#        fr1 = par['fr1']
#        fr2 = par['fr2']
#        lat = par['lat']
#        long = par['long']
#        psi = par['psi']
#        s1 = par['s1']
#        s2 = par['s2']
#
#    amp_time = env.average_envelopes_gaussian(
#        lat, long, s1, s2, psi, t1, t2, 
#        LISA_Orbital_Freq=1 / year, alpha0=0., beta0=0., tdi=tdi
#    )
#
#    Sh = (
#        amp_time * 10**amp * jnp.exp(-(freqs / 10**fr1)**alpha) *
#        (freqs**(-7./3.)) * 0.5 * (1.0 + jnp.tanh(-(freqs - 10**fknee) / 10**fr2))
#    )
#
#    # LISA arm length and angular frequency
#    omega = 2.0 * jnp.pi * freqs
#    x = omega * L
#    tr = (x) ** 2 * jnp.sin(x)**2
#    Sh *= tr
#
#    factor_tdi2 = 4 * jnp.sin(2 * x)**2
#
#    if lax is not None:
#        # Use lax.cond for JAX compatibility
#        Sh = lax.cond(gen2, lambda s: s * factor_tdi2, lambda s: s, Sh)
#    else:
#        # Fallback for non-JAX environments
#        Sh = Sh * factor_tdi2 if gen2 else Sh
#
#    return Sh


#@jit
#def disk_time(freqs, par, injected=False, t1=0, t2=0, tdi=0, gen2=False):
#    """
#    Galactic foreground model for gravitational wave signal with time dependence.
#    https://arxiv.org/abs/2410.08274, https://arxiv.org/abs/2410.08263
#
#    Args:
#        freqs (array): Frequency array.
#        injected (bool): If True, uses injected parameters.
#        par (dict): Parameters for the model.
#        t1 (float): Start time for the envelope.
#        t2 (float): End time for the envelope.
#        tdi (int): TDI channel (0 = A, 1 = E).
#    Returns:
#        array: Power spectral density.
#    """
#        
#    if 'fknee' not in par:
#        Tobs = par['Tobs'] / year
#        
#        alpha = par['alpha']
#        amp = par['amp']
#        a1 = par['a1']
#        b1 = par['b1']
#        ak = par['ak']
#        bk = par['bk']
#        fr2 = par['fr2']
#        lat = par['lat']
#        long = par['long']
#        psi = par['psi']
#        s1 = par['s1']
#        s2 = par['s2']
#        
#        fr1 = a1 * jnp.log10(Tobs) + b1
#        fknee = ak * jnp.log10(Tobs) + bk
#    else:
#        alpha = par['alpha']
#        amp = par['amp']
#        fknee = par['fknee']
#        fr1 = par['fr1']
#        fr2 = par['fr2']
#        lat = par['lat']
#        long = par['long']
#        psi = par['psi']
#        s1 = par['s1']
#        s2 = par['s2']
#
#    amp_time = env.average_envelopes_gaussian(
#        lat, long, s1, s2, psi, t1, t2, 
#        LISA_Orbital_Freq=1 / year, alpha0=0., beta0=0., tdi=tdi
#    )
#
#    Sh = (
#        amp_time * 10**amp * jnp.exp(-(freqs / 10**fr1)**alpha) *
#        (freqs**(-7./3.)) * 0.5 * (1.0 + jnp.tanh(-(freqs - 10**fknee) / 10**fr2))
#    )
#
#    # LISA arm length and angular frequency
#    omega = 2.0 * jnp.pi * freqs
#    x = omega * L
#    tr = (x) ** 2 * jnp.sin(x)**2
#    Sh *= tr
#
#    factor_tdi2 = 4 * jnp.sin(2 * x)**2
#
#    if lax is not None:
#        # Use lax.cond for JAX compatibility
#        Sh = lax.cond(gen2, lambda s: s * factor_tdi2, lambda s: s, Sh)
#    else:
#        # Fallback for non-JAX environments
#        Sh = Sh * factor_tdi2 if gen2 else Sh
#
#    return Sh




##################################################################################NOISE
@jit
def noise(freq, par, tdi, gen2=False):
    """
    Computes the noise power spectral density (PSD).

    This function calculates the noise PSD for LISA,
    including contributions from test mass noise and the optical metrology system.
    It supports both standard and TDI2 (Time-Delay Interferometry 2) configurations.

    Args:
        freq (array): Frequency array in Hz.
        par (dict): Dictionary containing the parameters:
            - 'A' (float): Amplitude of the test mass noise.
            - 'P' (float): Amplitude of the optical metrology system noise.
        tdi2 (bool, optional): If True, applies the TDI2 factor. Defaults to False.

    Returns:
        array: Noise power spectral density (PSD) as a function of frequency.
    """
    A = par['A']
    P = par['P']

    # Test mass noise
    sa_a = (A * 10**-15)**2 * (1.0 + (0.4e-3 / freq)**2) * (1.0 + (freq / 8e-3)**4)  # in acceleration
    sa_d = sa_a * (2.0 * jnp.pi * freq)**(-4.0)  # in displacement
    sa_nu = sa_d * (2.0 * jnp.pi * freq / clight)**2  # in relative frequency units
    s_pm = sa_nu

    # Optical Metrology System (OMS)
    relax = 1  # Relaxation factor (can be modified if needed)
    psd_oms_d = (P * 10**-12)**2 * relax  # in displacement
    s_oms_nu = psd_oms_d * (2.0 * jnp.pi * freq / clight)**2  # in relative frequency units
    s_op = s_oms_nu

    # LISA arm length and angular frequency
    omega = 2.0 * jnp.pi * freq
    
    # Compute the noise PSD
    x = omega * L
    
    sn_a = 8.0 * jnp.sin(x)**2 * (
        2.0 * s_pm * (3.0 + 2.0 * jnp.cos(x) + jnp.cos(2 * x)) +
        s_op * (2.0 + jnp.cos(x))
    )

    sn_t = (16.0 * s_op * (1.0 - jnp.cos(x)) * jnp.sin(x)**2 + 
                    128.0 * s_pm * jnp.sin(x)**2 * jnp.sin(0.5*x)**4
    )

    # Select the appropriate noise based on TDI channel
    if lax is not None:
        s_n = lax.cond(
        tdi == 2,
        lambda _: sn_t,
        lambda _: sn_a,
        operand=None
        )  
    else:
        s_n = sn_t if tdi == 2 else sn_a

    # Apply TDI2 factor if specified
    factor_tdi2 = 4 * jnp.sin(2 * x)**2

    if lax is not None:
        # Use lax.cond for JAX compatibility
        s_n = lax.cond(gen2, lambda s: s * factor_tdi2, lambda s: s, s_n)
    else:
        # Fallback for non-JAX environments
        s_n = s_n * factor_tdi2 if gen2 else s_n

    return s_n


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
        **kwargs: Additional arguments for specific sources (e.g., `t1`, `t2`).

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

        if source_name == 'cosmic_string':
            psd += response * Omega_pl(freqs, sources[source_name])
            if injected:
                true_psd.append(response * Omega_pl(freqs, sources[source_name]))

        elif source_name == 'extra_DWD':
            psd += response * Omega_extra_foreground(freqs, sources[source_name])
            if injected:
                true_psd.append(response * Omega_extra_foreground(freqs, sources[source_name]))

        elif source_name == 'galactic_DWD':
            psd += galactic_foreground(freqs, sources[source_name], injected=injected, gen2=kwargs.get('gen2'))
            if injected:
                true_psd.append(galactic_foreground(freqs, sources[source_name], injected=injected, gen2=kwargs.get('gen2')))

        elif source_name == 'galactic_DWD_time':
            t1 = kwargs.get('t1')
            t2 = kwargs.get('t2')
            psd += galactic_foreground_time(freqs, sources[source_name], t1=t1, t2=t2, tdi=tdi, injected=injected, gen2=kwargs.get('gen2'))
            if injected:
                true_psd.append(galactic_foreground_time(freqs, sources[source_name], t1=t1, t2=t2, tdi=tdi, injected=injected, gen2=kwargs.get('gen2')))

        elif source_name == 'gaussian_bump':
            psd += response * Omega_gaussian_bump(freqs, sources[source_name])
            if injected:
                true_psd.append(response * Omega_gaussian_bump(freqs, sources[source_name]))

        elif source_name == 'instr_noise':
            psd += noise(freqs, sources[source_name], tdi = tdi, gen2=kwargs.get('gen2'))
            if injected:
                true_psd.append(noise(freqs, sources[source_name], tdi = tdi, gen2=kwargs.get('gen2')))

        elif source_name == 'phase_transition':
            psd += response * Omega_phase_transition(freqs, sources[source_name])
            if injected:
                true_psd.append(response * Omega_phase_transition(freqs, sources[source_name]))

        elif source_name == 'power_law':
            psd += response * Omega_pl(freqs, sources[source_name])
            if injected:
                true_psd.append(response * Omega_pl(freqs, sources[source_name]))

        #elif source_name == 'bulge_time':
        #    t1 = kwargs.get('t1')
        #    t2 = kwargs.get('t2')
        #    psd += bulge_time(freqs, sources[source_name], t1=t1, t2=t2, tdi=tdi, injected=injected, gen2=kwargs.get('gen2'))
        #    if injected:
        #        true_psd.append(disk_time(freqs, sources[source_name], t1=t1, t2=t2, tdi=tdi, injected=injected, gen2=kwargs.get('gen2')))

        #elif source_name == 'disk_time':
        #    t1 = kwargs.get('t1')
        #    t2 = kwargs.get('t2')
        #    psd += disk_time(freqs, sources[source_name], t1=t1, t2=t2, tdi=tdi, injected=injected, gen2=kwargs.get('gen2'))
        #    if injected:
        #        true_psd.append(disk_time(freqs, sources[source_name], t1=t1, t2=t2, tdi=tdi, injected=injected, gen2=kwargs.get('gen2')))


    # Return the total PSD and optionally the true PSDs for each source
    if injected:
        return psd, true_psd
    else:
        return psd