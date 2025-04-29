"""
This module contains the function envelopes_gaussian, which computes the envelopes of the A and E signals
for sources centered at a given sky position, averaged over inclination and polarization, with a Gaussian distribution
with given standard deviations along two principal axes rotated with respect to the latitude and longitude.
The function takes as input the ecliptic latitude and longitude, the standard deviations along the two principal axes,
the rotation angle between the ecliptic longitude/latitude and the principal axes, the orbital frequency of LISA barycenter,
the time, the initial phase of LISA barycenter, the initial rotation of satellites in the detector plane, and a flag for TDI.
It returns the envelopes of the A and E signals.
"""

import jax.numpy as jnp
from jax import lax
import jax
jax.config.update('jax_enable_x64', True)

def envelopes_gaussian(EclipticLatitude, EclipticLongitude, Sigma1, Sigma2, Psi, t1, t2, LISA_Orbital_Freq, alpha0 = 0., beta0 = 0., tdi = 0):
    """
    Returns the envelopes of the A and E signals for sources centered at the given sky position,
    averaged over inclination and polarization, with some gaussian distribution with given
    standard deviations along two principal axes rotated with respect to the latitude and longitude

    :param SinEclipticLatitude: Sine sky position param
    :param EclipticLongitude: Sky position param
    :param Sigma1: Standard deviation along the first principal axis
    :param Sigma2: Standard deviation along the second principal axis
    :param Psi: Rotation angle between ecliptic longitude/latitude and principal axes
    :param LISA_Orbital_Freq: orbital frequency of LISA barycenter (1 / year)
    :param t: time
    :param alpha0: initial phase of LISA barycenter
    :param beta0: initial rotation of satellites in detector plane
    :return: A(t), E(t)
    """

    SigmaSqSum = Sigma1  +  Sigma2
    SigmaSqDiff = Sigma1  - Sigma2 

    cosPsi = Psi #cos 2 * Psi
    sinPsi = jnp.sqrt(1 - cosPsi**2)
    SigmaSqCos = SigmaSqDiff * cosPsi#jnp.cos(2. * Psi)
    SigmaSqSin = SigmaSqDiff * sinPsi#jnp.sin(2. * Psi)
    SigmaSqPlus = SigmaSqSum + SigmaSqCos

    fact1b0 = jnp.exp(-0.25 * SigmaSqPlus)
    fact3b = jnp.exp(-2. * SigmaSqPlus)
    fact5b = jnp.exp(-6. * SigmaSqPlus)

    coshHalfSigmaSqSin = jnp.cosh(0.5 * SigmaSqSin)
    sinhHalfSigmaSqSin = jnp.sinh(0.5 * SigmaSqSin)

    coshSigmaSqSin = coshHalfSigmaSqSin * coshHalfSigmaSqSin + sinhHalfSigmaSqSin * sinhHalfSigmaSqSin
    sinhSigmaSqSin = 2. * coshHalfSigmaSqSin * sinhHalfSigmaSqSin

    cosh3HalvesSigmaSqSin = coshHalfSigmaSqSin * coshSigmaSqSin + sinhHalfSigmaSqSin * sinhSigmaSqSin
    sinh3HalvesSigmaSqSin = coshHalfSigmaSqSin * sinhSigmaSqSin + sinhHalfSigmaSqSin * coshSigmaSqSin

    cosh5HalvesSigmaSqSin = cosh3HalvesSigmaSqSin * coshSigmaSqSin + sinh3HalvesSigmaSqSin * sinhSigmaSqSin
    sinh5HalvesSigmaSqSin = cosh3HalvesSigmaSqSin * sinhSigmaSqSin + sinh3HalvesSigmaSqSin * coshSigmaSqSin

    cosh3SigmaSqSin = cosh3HalvesSigmaSqSin * cosh3HalvesSigmaSqSin + sinh3HalvesSigmaSqSin * sinh3HalvesSigmaSqSin
    sinh3SigmaSqSin = 2. * cosh3HalvesSigmaSqSin * sinh3HalvesSigmaSqSin

    cosh5SigmaSqSin = cosh5HalvesSigmaSqSin * cosh5HalvesSigmaSqSin + sinh5HalvesSigmaSqSin * sinh5HalvesSigmaSqSin
    sinh5SigmaSqSin = 2. * cosh5HalvesSigmaSqSin * sinh5HalvesSigmaSqSin

    cosh9HalvesSigmaSqSin = cosh3HalvesSigmaSqSin * cosh3SigmaSqSin + sinh3HalvesSigmaSqSin * sinh3SigmaSqSin
    sinh9HalvesSigmaSqSin = cosh3HalvesSigmaSqSin * sinh3SigmaSqSin + sinh3HalvesSigmaSqSin * cosh3SigmaSqSin

    cosh15HalvesSigmaSqSin = cosh5HalvesSigmaSqSin * cosh5SigmaSqSin + sinh5HalvesSigmaSqSin * sinh5SigmaSqSin
    sinh15HalvesSigmaSqSin = cosh5HalvesSigmaSqSin * sinh5SigmaSqSin + sinh5HalvesSigmaSqSin * cosh5SigmaSqSin

    cosh2SigmaSqSin = coshSigmaSqSin * coshSigmaSqSin + sinhSigmaSqSin * sinhSigmaSqSin
    sinh2SigmaSqSin = 2. * coshSigmaSqSin * sinhSigmaSqSin

    cosh6SigmaSqSin = cosh3SigmaSqSin * cosh3SigmaSqSin + sinh3SigmaSqSin * sinh3SigmaSqSin
    sinh6SigmaSqSin = 2. * cosh3SigmaSqSin * sinh3SigmaSqSin

    cosh10SigmaSqSin = cosh5SigmaSqSin * cosh5SigmaSqSin + sinh5SigmaSqSin * sinh5SigmaSqSin
    sinh10SigmaSqSin = 2. * cosh5SigmaSqSin * sinh5SigmaSqSin

    root3 = jnp.sqrt(3.)

    T = t2 - t1
    fac = 2. * jnp.pi * LISA_Orbital_Freq
    phit1 = fac * t1
    phit2 = fac * t2
    a =  (EclipticLongitude - alpha0)

    FourphiMbar = 4. * ((EclipticLongitude - alpha0) + jnp.pi / 12.)

    sbM = EclipticLatitude #Sin Ecliptic Latitude
    cbM = jnp.sqrt(1 - sbM**2)
    s2bM = 2. * sbM * cbM
    c2bM = cbM * cbM - sbM * sbM

    s3bM = s2bM * cbM + c2bM * sbM
    c3bM = c2bM * cbM - s2bM * sbM

    s5bM = s2bM * c3bM + c2bM * s3bM
    c5bM = c2bM * c3bM - s2bM * s3bM

    sDphiL = (jnp.cos(a - phit1) - jnp.cos(a - phit2)) / (T*fac)
    cDphiL = (jnp.sin(a - phit1) - jnp.sin(a - phit2)) / (T*fac)

    s2DphiL = (jnp.cos(2*a - 2*phit1) - jnp.cos(2*a - 2*phit2)) / (2*fac*T)
    c2DphiL = (jnp.sin(2*a - 2*phit1) - jnp.sin(2*a - 2*phit2)) / (2*fac*T)

    s3DphiL = (jnp.cos(3*a - 3*phit1) - jnp.cos(3*a - 3*phit2)) / (3*fac*T)
    c3DphiL = (jnp.sin(3*a - 3*phit1) - jnp.sin(3*a - 3*phit2)) / (3*fac*T)

    s4DphiL = (jnp.cos(4*a - 4*phit1) - jnp.cos(4*a - 4*phit2)) / (4*fac*T)
    c4DphiL = (jnp.sin(4*a - 4*phit1) - jnp.sin(4*a - 4*phit2)) / (4*fac*T)

    s4phiMbar = jnp.sin(FourphiMbar)
    c4phiMbar = jnp.cos(FourphiMbar)

    sDphiL_4phiMbar = (jnp.cos(a - FourphiMbar - phit1) - jnp.cos(a - FourphiMbar - phit2)) / (T*fac)
    cDphiL_4phiMbar = (jnp.sin(a - FourphiMbar - phit1) - jnp.sin(a - FourphiMbar - phit2)) / (T*fac)

    s2DphiL_4phiMbar = (jnp.cos(2*a - FourphiMbar - 2*phit1) - jnp.cos(2*a - FourphiMbar - 2*phit2)) / (2*T*fac)
    c2DphiL_4phiMbar = (jnp.sin(2*a - FourphiMbar - 2*phit1) - jnp.sin(2*a - FourphiMbar - 2*phit2)) / (2*T*fac)

    s3DphiL_4phiMbar = (jnp.cos(3*a - FourphiMbar - 3*phit1) - jnp.cos(3*a - FourphiMbar - 3*phit2)) / (3*T*fac)
    c3DphiL_4phiMbar = (jnp.sin(3*a - FourphiMbar - 3*phit1) - jnp.sin(3*a - FourphiMbar - 3*phit2)) / (3*T*fac)

    s4DphiL_4phiMbar = (jnp.cos(4*a - FourphiMbar - 4*phit1) - jnp.cos(4*a - FourphiMbar - 4*phit2)) / (4*T*fac)
    c4DphiL_4phiMbar = (jnp.sin(4*a - FourphiMbar - 4*phit1) - jnp.sin(4*a - FourphiMbar - 4*phit2)) / (4*T*fac)

    s5DphiL_4phiMbar = (jnp.cos(5*a - FourphiMbar - 5*phit1) - jnp.cos(5*a - FourphiMbar - 5*phit2)) / (5*T*fac)
    c5DphiL_4phiMbar = (jnp.sin(5*a - FourphiMbar - 5*phit1) - jnp.sin(5*a - FourphiMbar - 5*phit2)) / (5*T*fac)

    s6DphiL_4phiMbar = (jnp.cos(6*a - FourphiMbar - 6*phit1) - jnp.cos(6*a - FourphiMbar - 6*phit2)) / (6*T*fac)
    c6DphiL_4phiMbar = (jnp.sin(6*a - FourphiMbar - 6*phit1) - jnp.sin(6*a - FourphiMbar - 6*phit2)) / (6*T*fac)

    s7DphiL_4phiMbar = (jnp.cos(7*a - FourphiMbar - 7*phit1) - jnp.cos(7*a - FourphiMbar - 7*phit2)) / (7*T*fac)
    c7DphiL_4phiMbar = (jnp.sin(7*a - FourphiMbar - 7*phit1) - jnp.sin(7*a - FourphiMbar - 7*phit2)) / (7*T*fac)

    s8DphiL_4phiMbar = (jnp.cos(8*a - FourphiMbar - 8*phit1) - jnp.cos(8*a - FourphiMbar - 8*phit2)) / (8*T*fac)
    c8DphiL_4phiMbar = (jnp.sin(8*a - FourphiMbar - 8*phit1) - jnp.sin(8*a - FourphiMbar - 8*phit2)) / (8*T*fac)

    fact1DphiL = root3 * jnp.exp(-0.5 * SigmaSqSum)
    fact2DphiL = jnp.exp(-1.25 * SigmaSqSum + 0.75 * SigmaSqCos)
    fact3DphiL = root3 * jnp.exp(-2.5 * SigmaSqSum + 2. * SigmaSqCos)
    fact4DphiL = jnp.exp(-4.25 * SigmaSqSum + 3.75 * SigmaSqCos)

    overall_factor = 1. / 10240.

    # A^2 + E^2
    Sum = fact1b0 * (120636. * cbM + 7614. * fact3b * c3bM - 666. * fact5b * c5bM)
    
    
    Sum += fact1DphiL * (-31392. * (coshHalfSigmaSqSin * cDphiL * sbM + sinhHalfSigmaSqSin * sDphiL * cbM)
                         - 32112. * (cosh3HalvesSigmaSqSin * cDphiL * s3bM + sinh3HalvesSigmaSqSin * sDphiL * c3bM)
                         - 720. * (cosh5HalvesSigmaSqSin * cDphiL * s5bM + sinh5HalvesSigmaSqSin * sDphiL * c5bM))


    Sum += fact2DphiL * (71280. * (coshSigmaSqSin * c2DphiL * cbM - sinhSigmaSqSin * s2DphiL * sbM)
                         + 22680. * (cosh3SigmaSqSin * c2DphiL * c3bM - sinh3SigmaSqSin * s2DphiL * s3bM)
                         - 648. * (cosh5SigmaSqSin * c2DphiL * c5bM - sinh5SigmaSqSin * s2DphiL * s5bM))


    Sum += fact3DphiL * (-864. * (cosh3HalvesSigmaSqSin * c3DphiL * sbM + cosh3HalvesSigmaSqSin * s3DphiL * cbM)
                         - 1296. * (cosh9HalvesSigmaSqSin * c3DphiL * s3bM + sinh9HalvesSigmaSqSin * s3DphiL * c3bM)
                         - 432. * (cosh15HalvesSigmaSqSin * c3DphiL * s5bM + sinh15HalvesSigmaSqSin * s3DphiL * c5bM))

    
    Sum += fact4DphiL * (1620. * (cosh2SigmaSqSin * c4DphiL * cbM - sinh2SigmaSqSin * s4DphiL * sbM)
                         + 810. * (cosh6SigmaSqSin * c4DphiL * c3bM - sinh6SigmaSqSin * s4DphiL * s3bM)
                         + 162. * (cosh10SigmaSqSin * c4DphiL * c5bM - sinh10SigmaSqSin * s4DphiL * s5bM))

    
    Sum *= overall_factor

    # A^2 - E^2
    Diff = c4DphiL_4phiMbar * fact1b0 * (324. * cbM - 2430. * fact3b * c3bM - 5670. * fact5b * c5bM)

    Diff += fact1DphiL * (-1296. * (coshHalfSigmaSqSin * c3DphiL_4phiMbar * sbM - sinhHalfSigmaSqSin * s3DphiL_4phiMbar * cbM)
                          + 3240. * (cosh3HalvesSigmaSqSin * c3DphiL_4phiMbar * s3bM - sinh3HalvesSigmaSqSin * s3DphiL_4phiMbar * c3bM)
                          + 4536. * (cosh5HalvesSigmaSqSin * c3DphiL_4phiMbar * s5bM - sinh5HalvesSigmaSqSin * s3DphiL_4phiMbar * c5bM))

    Diff += fact1DphiL * (432. * (coshHalfSigmaSqSin * c5DphiL_4phiMbar * sbM + sinhHalfSigmaSqSin * s5DphiL_4phiMbar * cbM)
                          - 1080. * (cosh3HalvesSigmaSqSin * c5DphiL_4phiMbar * s3bM + sinh3HalvesSigmaSqSin * s5DphiL_4phiMbar * c3bM)
                          - 1512. * (cosh5HalvesSigmaSqSin * c5DphiL_4phiMbar * s5bM + sinh5HalvesSigmaSqSin * s5DphiL_4phiMbar * c5bM))

    Diff += fact2DphiL * (-1944. * (coshSigmaSqSin * c2DphiL_4phiMbar * cbM + sinhSigmaSqSin * s2DphiL_4phiMbar * sbM)
                          + 10692. * (cosh3SigmaSqSin * c2DphiL_4phiMbar * c3bM + sinh3SigmaSqSin * s2DphiL_4phiMbar * s3bM)
                          + 6804. * (cosh5SigmaSqSin * c2DphiL_4phiMbar * c5bM + sinh5SigmaSqSin * s2DphiL_4phiMbar * s5bM))

    Diff += fact2DphiL * (-216. * (coshSigmaSqSin * c6DphiL_4phiMbar * cbM - sinhSigmaSqSin * s6DphiL_4phiMbar * sbM)
                          + 1188. * (cosh3SigmaSqSin * c6DphiL_4phiMbar * c3bM - sinh3SigmaSqSin * s6DphiL_4phiMbar * s3bM)
                          + 756. * (cosh5SigmaSqSin * c6DphiL_4phiMbar * c5bM - sinh5SigmaSqSin * s6DphiL_4phiMbar * s5bM))

    Diff += fact3DphiL * (-3888. * (cosh3HalvesSigmaSqSin * cDphiL_4phiMbar * sbM - sinh3HalvesSigmaSqSin * sDphiL_4phiMbar * cbM)
                          - 5832. * (cosh9HalvesSigmaSqSin * cDphiL_4phiMbar * s3bM - sinh9HalvesSigmaSqSin * sDphiL_4phiMbar * c3bM)
                          - 1944. * (cosh15HalvesSigmaSqSin * cDphiL_4phiMbar * s5bM - sinh15HalvesSigmaSqSin * sDphiL_4phiMbar * c5bM))

    Diff += fact3DphiL * (144. * (cosh3HalvesSigmaSqSin * c7DphiL_4phiMbar * sbM + sinh3HalvesSigmaSqSin * s7DphiL_4phiMbar * cbM)
                          + 216. * (cosh9HalvesSigmaSqSin * c7DphiL_4phiMbar * s3bM + sinh9HalvesSigmaSqSin * s7DphiL_4phiMbar * c3bM)
                          + 72. * (cosh15HalvesSigmaSqSin * c7DphiL_4phiMbar * s5bM + sinh15HalvesSigmaSqSin * s7DphiL_4phiMbar * c5bM))

    Diff += fact4DphiL * (-7290. * (cosh2SigmaSqSin * c4phiMbar * cbM + sinh2SigmaSqSin * s4phiMbar * sbM)
                          - 3645. * (cosh6SigmaSqSin * c4phiMbar * c3bM + sinh6SigmaSqSin * s4phiMbar * s3bM)
                          - 729. * (cosh10SigmaSqSin * c4phiMbar * c5bM + sinh10SigmaSqSin * s4phiMbar * s5bM))

    Diff += fact4DphiL * (-90. * (cosh2SigmaSqSin * c8DphiL_4phiMbar * cbM - sinh2SigmaSqSin * s8DphiL_4phiMbar * sbM)
                          - 45. * (cosh6SigmaSqSin * c8DphiL_4phiMbar * c3bM - sinh6SigmaSqSin * s8DphiL_4phiMbar * s3bM)
                          - 9. * (cosh10SigmaSqSin * c8DphiL_4phiMbar * c5bM - sinh10SigmaSqSin * s8DphiL_4phiMbar * s5bM))

    Diff *= overall_factor
    A0 = (0.5 * jnp.abs(Sum + Diff))
    E0 = (0.5 * jnp.abs(Sum - Diff))

    sDect = jnp.sin(2. * (beta0 - alpha0))
    cDect = jnp.cos(2. * (beta0 - alpha0))

    A = jnp.abs(cDect * A0 - sDect * E0)  
    E = jnp.abs(sDect * A0 + cDect * E0) 
    mod = lax.cond(
    tdi == 0,
    lambda _: A,
    lambda _: E,
    operand=None
    )      
    return mod












