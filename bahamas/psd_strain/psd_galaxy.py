"""Galactic foreground models for gravitational wave signals.
    This module provides functions to compute the power spectral density (PSD) of the galactic foreground
    of white dwarf binaries, including time-dependent models.
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
from bahamas.psd_response import average_envelope_num as env_num

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


def precompute_grids(N_lambda=180, N_beta=90):
    """Precompute sky grids and a few derived arrays (kept small)."""
    R_sun = 8.12  # kpc
    z_sun = 0.03  # kpc
    N_D = None    # not used anymore for full 3D heavy grid

    lams = jnp.linspace(0, 2 * jnp.pi, N_lambda)
    betas = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, N_beta)
    Lam_grid, Beta_grid = jnp.meshgrid(lams, betas, indexing='ij')  # shape (N_lambda, N_beta)

    Lam_flat = Lam_grid.ravel()
    Beta_flat = Beta_grid.ravel()

    Eclat = Beta_flat
    Eclon = Lam_flat
    cos_eclat = jnp.cos(Eclat)

    return {
        'N_lambda': N_lambda,
        'N_beta': N_beta,
        'grid_shape': (N_lambda, N_beta),
        'Lam_flat': Lam_flat,
        'Beta_flat': Beta_flat,
        'Eclat': Eclat,
        'Eclon': Eclon,
        'cos_eclat': cos_eclat,
    }


import numpy as np
def precompute_los_quadrature(PRECOMPUTED, D_max=20.0, N_quad=16, R_sun=8.12, z_sun=0.03):
    """
    Precompute Gauss-Legendre nodes & weights mapped to [0,D_max] and the geometry
    (R(D), z(D)) for each quadrature node and each sky pixel.
    """
    # 1) Gauss-Legendre nodes & weights on [-1,1]
    nodes, weights = np.polynomial.legendre.leggauss(N_quad)  # numpy arrays
    D_nodes = 0.5 * (nodes + 1.0) * D_max
    quad_weights = 0.5 * D_max * weights

    # 2) sky coordinates flattened (ecliptic l,b)
    Lam_flat = PRECOMPUTED['Lam_flat']
    Beta_flat = PRECOMPUTED['Beta_flat']
    # convert to galactic-like coords using your env_num transform (vmap)
    lam_gal, beta_gal = env_num.transform_sky_coordinates_vmap(Lam_flat, Beta_flat)  # shape (N_sky,)

    # broadcast to (N_quad, N_sky)
    D_mat = jnp.broadcast_to(jnp.asarray(D_nodes)[:, None], (len(D_nodes), lam_gal.shape[0]))
    lam_mat = jnp.broadcast_to(lam_gal[None, :], (len(D_nodes), lam_gal.shape[0]))
    beta_mat = jnp.broadcast_to(beta_gal[None, :], (len(D_nodes), beta_gal.shape[0]))

    # compute galactic cartesian coords per node/pixel
    x = D_mat * jnp.sin(lam_mat) * jnp.cos(beta_mat)
    y = R_sun - D_mat * jnp.cos(lam_mat) * jnp.cos(beta_mat)
    z = z_sun - D_mat * jnp.sin(beta_mat)

    R_vals = jnp.sqrt(x**2 + y**2)   # shape (N_quad, N_sky)
    z_vals = z                       # shape (N_quad, N_sky)

    # store into PRECOMPUTED (convert quad_weights to jnp)
    PRECOMPUTED['D_nodes'] = jnp.asarray(D_nodes)
    PRECOMPUTED['quad_weights'] = jnp.asarray(quad_weights)
    PRECOMPUTED['R_vals_quad'] = R_vals
    PRECOMPUTED['z_vals_quad'] = z_vals
    PRECOMPUTED['N_quad'] = len(D_nodes)
    PRECOMPUTED['lam_gal_flat'] = lam_gal
    PRECOMPUTED['beta_gal_flat'] = beta_gal

    return PRECOMPUTED

@jit
def compute_weight_auto_components(par):
    """
    Compute weights automatically detecting bulge/disk from parameters.
    Uses the precomputed quadrature structure for line-of-sight integration.
    """
    
    # New behavior - separate bulge/disk detection
    # Extract disk parameters
    R_d = par.get('R_d', 0.0)
    z_d = par.get('z_d', 0.0)
    
    # Extract bulge parameters
    R_cut = par.get('R_cut', 0.0)
    alpha_bulge = par.get('alpha_bulge', 0.0)

    
    # Compute components separately
    weight_disk = jnp.where((R_d > 0) & (z_d > 0),
                           compute_disk_weight_quadrature(R_d, z_d),
                           0.0)
    
    weight_bulge = jnp.where((R_cut > 0),
                            compute_bulge_weight_quadrature(R_cut, alpha_bulge),
                            0.0)
    
    return weight_disk + weight_bulge


@jit
def compute_disk_weight_quadrature(R_d, z_d):
    """
    Compute disk component weight using quadrature integration.
    McMillan 2011 disk: rho = rho_WD_sun * exp(-R/R_d) * exp(-|z|/z_d)
    """
    return compute_sigma_and_weights_disk(
        PRECOMPUTED['R_vals_quad'],
        PRECOMPUTED['z_vals_quad'], 
        PRECOMPUTED['quad_weights'],
        PRECOMPUTED['cos_eclat'],
        R_d, z_d
    )


@jit
def compute_bulge_weight_quadrature(R_cut, alpha_bulge):
    """
    Compute bulge component weight using quadrature integration.
    McMillan 2011 bulge: rho = rho_WD_sun * exp(-R/R_cut^2) / (1 + rprime/r0)^alpha_bulge
    where rprime = sqrt(R^2 + (z/q)^2)
    """
    return compute_sigma_and_weights_bulge(
        PRECOMPUTED['R_vals_quad'],
        PRECOMPUTED['z_vals_quad'],
        PRECOMPUTED['quad_weights'], 
        PRECOMPUTED['cos_eclat'],
        R_cut, alpha_bulge
    )


@jit
def compute_sigma_and_weights_disk(quad_R_vals, quad_z_vals, quad_w, cos_eclat, R_d, z_d):
    """
    Quadrature integration for disk density profile.
    Same structure as original but with McMillan 2011 disk density.
    """
    # McMillan 2011 disk density
    expR = jnp.exp(-quad_R_vals / R_d)           # (N_quad, N_sky)
    expZ = jnp.exp(-jnp.abs(quad_z_vals) / z_d)  # (N_quad, N_sky)
    density =  expR * expZ           # (N_quad, N_sky)
    
    # Weight integrand by quadrature weights
    integrand = quad_w[:, None] * density        # (N_quad, N_sky)
    
    # Sum over quadrature nodes -> Sigma per sky pixel
    Sigma = jnp.sum(integrand, axis=0)           # (N_sky,)
    
    # Solid-angle weighting and normalization
    numer = Sigma * cos_eclat
    denom = jnp.sum(numer)
    denom = jnp.where(denom == 0.0, 1.0, denom)
    weight = numer / denom
    
    return weight


@jit
def compute_sigma_and_weights_bulge(quad_R_vals, quad_z_vals, quad_w, cos_eclat, 
                                   R_cut, alpha_bulge):
    """
    Quadrature integration for bulge density profile.
    McMillan 2011 bulge density with triaxial structure.
    """
    # McMillan 2011 bulge density
    rprime = jnp.sqrt(quad_R_vals**2 + (quad_z_vals / 0.5)**2)  # (N_quad, N_sky)
    exp_term = jnp.exp(-quad_R_vals / R_cut**2)               # (N_quad, N_sky)
    power_term = (1 + rprime / 0.075)**alpha_bulge               # (N_quad, N_sky)
    density =  exp_term / power_term              # (N_quad, N_sky)
    
    # Weight integrand by quadrature weights
    integrand = quad_w[:, None] * density                     # (N_quad, N_sky)
    
    # Sum over quadrature nodes -> Sigma per sky pixel
    Sigma = jnp.sum(integrand, axis=0)                        # (N_sky,)
    
    # Solid-angle weighting and normalization  
    numer = Sigma * cos_eclat
    denom = jnp.sum(numer)
    denom = jnp.where(denom == 0.0, 1.0, denom)
    weight = numer / denom
    
    return weight


# compute once (choose N_quad small e.g. 16)
PRECOMPUTED = precompute_grids(N_lambda=180, N_beta=90)
PRECOMPUTED = precompute_los_quadrature(PRECOMPUTED, D_max=20.0, N_quad=16)

@jit
def galactic_foreground_num(freqs, par, injected=False, t1=0, t2=0, tdi=0, gen2=False):
    """
    PSD for galactic foreground using quadrature-based LOS integration for weights.
    Automatically detects which galactic components to include based on provided parameters.
    
    freqs: 1D array
    par: dict with parameters. Component selection based on parameter presence:
         - Disk: requires 'R_d' and 'z_d' 
         - Bulge: requires 'R_cut', 'r0', 'alpha_bulge', 'q'
         - Set any parameter to 0 to disable that component
         - Legacy support: 'rb', 'zd' behaves as before (both components)
    """
    # Parse frequency/amplitude parameters (keep your original handling)
    if 'fknee' not in par:
        Tobs = par['Tobs'] / year
        alpha = par['alpha']
        amp = par['amp']
        a1 = par['a1']; b1 = par['b1']
        ak = par['ak']; bk = par['bk']; fr2 = par['fr2']
        fr1 = a1 * jnp.log10(Tobs) + b1
        fknee = ak * jnp.log10(Tobs) + bk
    else:
        alpha = par['alpha']; amp = par['amp']
        fknee = par['fknee']; fr1 = par['fr1']; fr2 = par['fr2']
    
    # Automatic component detection and weight computation
    weight = compute_weight_auto_components(par)
    
    # compute time-dependent amplitude using pre-computed sky grids
    # weight shape should be (N_sky,); Eclat, Eclon shape (N_sky,)
    amp_time = env_num.compute_average_envelope_time(
        EclipticLatitude=PRECOMPUTED['Eclat'],
        EclipticLongitude=PRECOMPUTED['Eclon'],
        weight=weight,
        LISA_Orbital_Freq=1 / year,
        t1=t1,
        t2=t2,
        tdi=tdi,
    )

    # spectral shape
    Sh = (
        amp_time * 10**amp * jnp.exp(-(freqs / 10**fr1)**alpha) *
        (freqs**(-7.0/3.0)) * 0.5 * (1.0 + jnp.tanh(-(freqs - 10**fknee) / 10**fr2))
    )

    # apply LISA transfer-like factors (use your original definition)
    omega = 2.0 * jnp.pi * freqs
    x = omega * L
    tr = (x) ** 2 * jnp.sin(x)**2
    Sh = Sh * tr
    factor_tdi2 = 4 * jnp.sin(2 * x)**2

    if hasattr(lax, 'cond'):
        Sh = lax.cond(gen2, lambda s: s * factor_tdi2, lambda s: s, Sh)
    else:
        Sh = Sh * factor_tdi2 if gen2 else Sh

    return Sh

def plot_sky_distribution(sources, folder):
    """Plot the sky distribution of the Galactic foreground model."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    from mpl_toolkits.mplot3d import proj3d

    # Convert injected parameters into a dictionary format
    dict1 = {}
    for source_name, param_list in sources.items():
        source = {param["name"]: param["injected"] for param in param_list}
        dict1[source_name] = source
    sources = dict1['galactic_DWD_num']

    weight = compute_weight_auto_components(sources)

    lams = jnp.linspace(0, 2 * jnp.pi, PRECOMPUTED['N_lambda'])
    betas = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, PRECOMPUTED['N_beta'])
    Lam_grid, Beta_grid = jnp.meshgrid(lams, betas, indexing='ij')
    # Reshape weight to 2D grid
    weight = weight.reshape((PRECOMPUTED['N_lambda'], PRECOMPUTED['N_beta']))
    

    lat = jnp.deg2rad(-5.6)  # Galactic latitude of the Sun in radians
    lon = jnp.deg2rad(266.4)  # Galactic longitude of the Sun in radians
   
    # Plot the sky distribution
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(Lam_grid, Beta_grid, weight, shading='auto', cmap='inferno')
    plt.colorbar(label='Weight')
    plt.xlabel('Ecliptic Longitude (radians)')
    plt.ylabel('Ecliptic Latitude (radians)')
    plt.title('Sky Distribution of Galactic Foreground Model')
    plt.scatter(lon, lat, color='white', marker='*', label='Sun Position')
    plt.savefig(f"{folder}/galactic_foreground_sky_distribution.png")
    plt.close()



