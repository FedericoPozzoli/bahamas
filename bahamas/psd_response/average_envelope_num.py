from bahamas.backend_context import get_backend_components
# Load backend components after backend initialization
jnp, jit, lax = get_backend_components()

if jnp is None:
    import numpy as jnp

if lax is not None:
    import jax
    jax.config.update('jax_enable_x64', True)
    vmap = jax.vmap
else:
    # Define a numpy-compatible vmap substitute
    def numpy_vmap(func, in_axes):
        """NumPy-compatible substitute for JAX's vmap."""
        def vectorized_func(*args):
            # Handle the most common case where all axes are 0 (vectorize over first dimension)
            if all(ax == 0 for ax in in_axes if ax is not None):
                # Dynamically determine the signature based on the number of arguments
                n_inputs = len(args)
                
                # Try to call the function once to determine output structure
                try:
                    # Use a small subset to test the function
                    test_args = []
                    for i, arg in enumerate(args):
                        if in_axes[i] is not None and hasattr(arg, '__len__') and len(arg) > 0:
                            # Take first element if it's an array
                            test_args.append(arg[0] if hasattr(arg[0], '__len__') else arg[0])
                        else:
                            test_args.append(arg)
                    
                    test_result = func(*test_args)
                    
                    # Determine if single or multiple outputs
                    if isinstance(test_result, tuple):
                        n_outputs = len(test_result)
                        signature = '(),' * n_inputs + '->' + '(),' * n_outputs
                        signature = signature.rstrip(',')  # Remove trailing comma
                    else:
                        signature = '(),' * n_inputs + '->()'
                        signature = signature.rstrip(',')  # Remove trailing comma
                    
                    return jnp.vectorize(func, signature=signature)(*args)
                    
                except Exception:
                    # Fallback to manual implementation if signature detection fails
                    pass
            
            # Manual implementation for complex cases or when signature detection fails
            shapes = [arg.shape for i, arg in enumerate(args) if in_axes[i] is not None]
            if shapes:
                n_elements = shapes[0][0]  # Assuming first dimension
                results = []
                for i in range(n_elements):
                    element_args = []
                    for j, arg in enumerate(args):
                        if in_axes[j] is not None:
                            element_args.append(arg[i])
                        else:
                            element_args.append(arg)
                    results.append(func(*element_args))
                
                # Stack results
                if results and isinstance(results[0], tuple):  # Multiple outputs
                    return tuple(jnp.array([r[k] for r in results]) for k in range(len(results[0])))
                else:  # Single output
                    return jnp.array(results)
            else:
                return func(*args)
        return vectorized_func
    
    vmap = numpy_vmap

# =============================================================================

# Define the rotation matrix as a JAX array
rotationMatrix = jnp.array([[-5.48755604e-02, -9.93821362e-01, -9.64768044e-02],
                           [ 4.94109428e-01, -1.10990888e-01,  8.62285855e-01],
                           [-8.67666149e-01, -3.51679084e-04,  4.97147192e-01]])

# =============================================================================
# COORDINATE TRANSFORMATION FUNCTIONS
# =============================================================================

@jit
def transform_cartesian_coordinates(x, y, z):
    """
    Rotate Cartesian coordinates from one reference system to another.
    """
    coords = jnp.array([x, y, z])
    rotated_coords = jnp.dot(rotationMatrix, coords)
    return rotated_coords[0], rotated_coords[1], rotated_coords[2]

@jit
def spherical_to_cartesian(r, phi, theta):
    """
    Convert spherical to Cartesian coordinates (astronomical convention).
    """
    ctheta = jnp.cos(theta)
    x = r * jnp.cos(phi) * ctheta
    y = r * jnp.sin(phi) * ctheta
    z = r * jnp.sin(theta)
    return x, y, z

@jit
def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian to spherical coordinates (astronomical convention).
    """
    rCylSq = x * x + y * y
    r = jnp.sqrt(rCylSq + z * z)
    
    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi < 0.0, phi + 2 * jnp.pi, phi)
    
    theta = jnp.where(r == 0.0, jnp.nan, jnp.arctan2(z, jnp.sqrt(rCylSq)))
    phi = jnp.where(r == 0.0, jnp.nan, phi)
    
    return r, phi, theta

@jit
def transform_sky_coordinates(phi, theta):
    """
    Convert sky coordinates from one reference system to another.
    """
    r = jnp.ones_like(phi)
    x, y, z = spherical_to_cartesian(r, phi, theta)
    xrot, yrot, zrot = transform_cartesian_coordinates(x, y, z)
    r, phirot, thetarot = cartesian_to_spherical(xrot, yrot, zrot)
    return phirot, thetarot

@jit
def density_distribution(R, z, R_d=2.5, z_d=0.3):
    """
    Compute the density distribution in the Galactic disk.
    """
    rho = jnp.exp(-R / R_d) * jnp.exp(-jnp.abs(z) / z_d)
    return rho

# =============================================================================
# LISA ENVELOPE FUNCTIONS
# =============================================================================

@jit
def envelopes_time_average(EclipticLatitude, EclipticLongitude, LISA_Orbital_Freq, t1, t2, tdi, alpha0=0, beta0=0):
    """
    Returns the envelopes of the A and E signals for a source located at the given sky position,
    averaged over inclination and polarization.
    """
    cthN = jnp.sin(EclipticLatitude)
    sthN = jnp.sqrt(1. - cthN * cthN)
    sthN2 = sthN * sthN
    sthN3 = sthN * sthN2
    sthN4 = sthN * sthN3
    
    T = t2 - t1
    fac = 2. * jnp.pi * LISA_Orbital_Freq
    phit1 = fac * t1
    phit2 = fac * t2
    a = EclipticLongitude - alpha0
    FourphiNbar = 4. * ((EclipticLongitude - alpha0) + jnp.pi / 12.)
    root3 = jnp.sqrt(3.)
    overall_fact = 1. / 640.
    
    # A^2 + E^2
    Sum = 5904. + 2736. * sthN2 - 666. * sthN4
    Sum += root3 * cthN * sthN * (-7488. - 720. * sthN2) * (jnp.sin(a - phit1) - jnp.sin(a - phit2)) / (T * fac)
    Sum += sthN2 * (6480. - 648. * sthN2) * (jnp.sin(2. * (a - phit1)) - jnp.sin(2. * (a - phit2))) / (2. * T * fac)
    Sum += -432. * root3 * cthN * sthN3 * (jnp.sin(3. * (a - phit1)) - jnp.sin(3. * (a - phit2))) / (3. * T * fac)
    Sum += 162. * sthN4 * (jnp.sin(4. * (a - phit1)) - jnp.sin(4. * (a - phit2))) / (4. * T * fac)
    Sum *= overall_fact
    
    # A^2 - E^2
    Diff = (-1296. + 6480. * sthN2 - 5670. * sthN4) * (jnp.sin(4. * (a - FourphiNbar - phit1)) - jnp.sin(4. * (a - FourphiNbar - phit2))) / (4. * T * fac)
    Diff += root3 * cthN * sthN * (864. - 1512. * sthN2) \
        * (-3. * (jnp.sin(3*a - FourphiNbar - 3*phit1) - jnp.sin(3*a - FourphiNbar - 3*phit2)) / (3*T*fac) + (jnp.sin(5*a - FourphiNbar - 5*phit1) - jnp.sin(5*a - FourphiNbar - 5*phit2)) / (5*T*fac))
    Diff += sthN2 * (-648. + 756. * sthN2) \
        * (9. * (jnp.sin(2*a - FourphiNbar - 2*phit1) - jnp.sin(2*a - FourphiNbar - 2*phit2)) / (2*T*fac) + (jnp.sin(6*a - FourphiNbar - 6*phit1) - jnp.sin(6*a - FourphiNbar - 6*phit2)) / (6*T*fac))
    Diff += 72. * root3 \
        * cthN \
        * sthN3 \
        * (-27. * (jnp.sin(a - FourphiNbar - phit1) - jnp.sin(a - FourphiNbar - phit2)) / (T*fac) + (jnp.sin(7*a - FourphiNbar - 7*phit1) - jnp.sin(7*a - FourphiNbar - 7*phit2)) / (7*T*fac))
    Diff += -9. * sthN4 * (81. * jnp.cos(FourphiNbar) + (jnp.sin(8*a - FourphiNbar - 8*phit1) - jnp.sin(8*a - FourphiNbar - 8*phit2)) / (8*T*fac))
    Diff *= overall_fact
    
    A0 = 0.5 * (jnp.abs(Sum + Diff))
    E0 = 0.5 * (jnp.abs(Sum - Diff))
    
    sDect = jnp.sin(2. * (beta0 - alpha0))
    cDect = jnp.cos(2. * (beta0 - alpha0))

    A = jnp.abs(cDect * A0 - sDect * E0)  
    E = jnp.abs(sDect * A0 + cDect * E0) 
    

    if lax is not None:
        mod = lax.cond(
        tdi == 0,
        lambda _: A,
        lambda _: E,
        operand=None
        )  
    else:
        mod = A if tdi == 0 else E   
         
    return mod

@jit
def compute_average_envelope_time(EclipticLatitude, EclipticLongitude, weight, LISA_Orbital_Freq, t1, t2, tdi, alpha0=0., beta0=0.):
    """
    Compute time-averaged envelope functions with weights.
    """
    def single_time_envelope(t1i, t2i):
        mod = envelopes_time_average(
            EclipticLatitude, EclipticLongitude,
            LISA_Orbital_Freq, t1i, t2i,
            alpha0=alpha0, beta0=beta0, tdi=tdi
        )
        mod_weighted = jnp.sum(mod * weight)
        
        return mod_weighted
    
    return single_time_envelope(t1, t2)

# =============================================================================
# VECTORIZED VERSIONS
# =============================================================================

# Create vectorized versions that work with both JAX and NumPy
transform_cartesian_coordinates_vmap = vmap(transform_cartesian_coordinates, in_axes=(0, 0, 0))
spherical_to_cartesian_vmap = vmap(spherical_to_cartesian, in_axes=(0, 0, 0))
cartesian_to_spherical_vmap = vmap(cartesian_to_spherical, in_axes=(0, 0, 0))
transform_sky_coordinates_vmap = vmap(transform_sky_coordinates, in_axes=(0, 0))
density_distribution_vmap = vmap(density_distribution, in_axes=(0, 0, None, None))