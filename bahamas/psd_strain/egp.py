"""
This module implements a Gaussian Process (GP) based method for estimating the power spectral density (PSD)
of a signal. It includes functions for computing the covariance matrix using a squared exponential kernel
and for estimating the PSD using the GP framework.
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


def GP_SGWB_kernel_SE(x1, x2, lam, eps=None):
    # Convert inputs to jax arrays
    x1 = jnp.asarray(x1)
    x2 = jnp.asarray(x2)
    
    """
    We could include also other types of kernels.
    """
    # Radial basis function
    tau = (x1[:, jnp.newaxis] - x2[jnp.newaxis, :])
    K = jnp.exp(-0.5 * (tau / lam)**2)
   

    # Add regularization term if specified
    if eps is not None:
        assert len(x1) == len(x2), "Covariance matrix is not square"
        K += eps * jnp.eye(len(x1))

    return K


def egp(freqs, base_psd, par, posterior = False):
    """
    Apply Gaussian Process corrections to a base power spectral density (PSD) estimate.
    
    Args:
        freqs (array-like): Frequencies at which to evaluate the PSD.
        base_psd (array-like): Base PSD values (not in log space).
        par (dict): Dictionary containing hyperparameters for the Gaussian Process.
                   Must include delta_i parameters and fmin/fmax.
        posterior (bool): If True, use posterior mean for log_psd (currently unused).
    
    Returns:
        psd (array-like): Corrected power spectral density at the given frequencies.
    """
    
    fmin = par['fmin']
    fmax = par['fmax']
    
    # Length scale for the GP kernel
    lam = (jnp.log10(fmax) - jnp.log10(fmin)) / 10
    
    # Extract delta values from parameter dictionary
    value = jnp.array([value for key, value in par.items() if key.startswith('delta_')])
    
    # Number of nodes
    n_nodes = len(value)
    
    if n_nodes == 0:
        # No EGP correction, return base PSD
        return base_psd
    
    # Create node positions uniformly in log-frequency space
    nodes = jnp.logspace(jnp.log10(freqs[0]), jnp.log10(freqs[-1]), n_nodes)
    
    # Compute covariance matrix at nodes
    cov = GP_SGWB_kernel_SE(jnp.log10(nodes), jnp.log10(nodes), lam=lam, eps=1.0e-6)
    inv_cov = jnp.linalg.inv(cov)
    
    # Compute cross-covariance between frequencies and nodes
    k_star = GP_SGWB_kernel_SE(jnp.log10(freqs), jnp.log10(nodes), lam=lam)
    
    # GP regression matrix
    matrix = jnp.dot(k_star, inv_cov)
    
    # GP correction in log space
    log_correction = jnp.sum(matrix * value, axis=1)  # Shape: (nfreqs,)
    
    # Apply correction: multiply base PSD by exponential of log correction
    # This is equivalent to: psd = base_psd * 10^(log_correction)
    log_base_psd = jnp.log10(base_psd)
    corrected_log_psd = log_base_psd + log_correction
    psd = 10**corrected_log_psd
    
    return psd
