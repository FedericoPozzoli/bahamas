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
from bahamas.psd_strain import psd_signal as signal
from bahamas.psd_strain import psd_noise as noise
from bahamas.psd_strain import psd_galaxy as gal

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
    true_psd = {}

    # Loop through each source and compute its contribution to the PSD
    for source_name in sources.keys():
        
        # Check if this source has EGP parameters
        # EGP parameters are indicated by keys like 'delta_0', 'delta_1', etc.
        source_params = sources[source_name]
        has_egp = any(key.startswith('delta_') for key in source_params.keys())
        
        if not injected:
            # For non-injected case, check if any parameter has egp flag
            param_list = sources.get(source_name, [])
            if isinstance(param_list, list):
                apply_egp = any(param.get("egp", False) for param in param_list)
            else:
                apply_egp = False
        else:
            apply_egp = False

        if source_name == 'cosmic_string':
            base = signal.Omega_pl(freqs, source_params)
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += response * base
            if injected:
                true_psd['cosmic_string'] = response * base
        
        elif source_name == 'dynamical_friction':
            base = signal.Omega_DF(freqs, source_params)
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += response * base
            if injected:
                true_psd['dynamical_friction'] = response * base

        elif source_name == 'ecc_pop':
            base = signal.Omega_ecc_pop(freqs, source_params)
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += response * base
            if injected:
                true_psd['ecc_pop'] = response * base

        elif source_name == 'extra_DWD':
            base = signal.Omega_extra_foreground(freqs, source_params)
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += response * base
            if injected:
                true_psd['extra_DWD'] = response * base

        elif source_name == 'galactic_DWD':
            base = gal.galactic_foreground(freqs, source_params, injected=injected, gen2=kwargs.get('gen2'))
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += base
            if injected:
                true_psd['galactic_DWD'] = base

        elif source_name == 'galactic_DWD_num':
            t1 = kwargs.get('t1')
            t2 = kwargs.get('t2')

            base = gal.galactic_foreground_num(freqs, source_params, t1=t1, t2=t2, tdi=tdi, injected=injected, gen2=kwargs.get('gen2'))
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += base
            if injected:
                true_psd['galactic_DWD_num'] = base


        elif source_name == 'galactic_DWD_time':
            t1 = kwargs.get('t1')
            t2 = kwargs.get('t2')
            base = gal.galactic_foreground_time(freqs, source_params, t1=t1, t2=t2, tdi=tdi, injected=injected, gen2=kwargs.get('gen2'))
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += base
            if injected:
                true_psd['galactic_DWD_time'] = base

        elif source_name == 'gaussian_bump':
            base = signal.Omega_gaussian_bump(freqs, source_params)
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += response * base
            if injected:
                true_psd['gaussian_bump'] = response * base

        elif source_name == 'instr_noise':
            psd += noise.noise(freqs, source_params, tdi = tdi, gen2=kwargs.get('gen2'))
            if injected:
                true_psd['instr_noise'] = noise.noise(freqs, source_params, tdi = tdi, gen2=kwargs.get('gen2'))

        elif source_name == 'phase_transition':
            base = signal.Omega_phase_transition(freqs, source_params)
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += response * base
            if injected:
                true_psd['phase_transition'] = response * base

        elif source_name == 'power_law':
            base = signal.Omega_pl(freqs, source_params)
            if has_egp:
                base = egp.egp(freqs, base, source_params)
            psd += response * base
            if injected:
                true_psd['power_law'] = response * base

        elif source_name == 'acc_noise':
            base = noise.acc_noise(freqs, source_params)
            psd += base * noise.transfer_functions_acc(freqs, L, tdi, gen2=kwargs.get('gen2'))

        elif source_name == 'oms_noise':
            base = noise.oms_noise(freqs, source_params)
            psd += base * noise.transfer_functions_oms(freqs, L, tdi, gen2=kwargs.get('gen2'))
   
    # Return the total PSD and optionally the true PSDs for each source
    if injected:
        return psd, true_psd
    else:
        return psd