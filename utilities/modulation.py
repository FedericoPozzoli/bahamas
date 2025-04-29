"""
This module contains the function to compute the envelopes of the A and E signals as functions of time
for sources centered at a given sky position, averaged over inclination and polarization.
It uses a Gaussian distribution with specified standard deviations along two principal axes
rotated with respect to the latitude and longitude.
The function `envelopes_gaussian` computes the envelopes based on the provided parameters.
"""
import numpy as np
import pathlib
import matplotlib.pylab as pl


def envelopes_gaussian(EclipticLatitude, EclipticLongitude, Sigma1, Sigma2, Psi, LISA_Orbital_Freq, t, alpha0 = 0., beta0 = 0.):
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

    SigmaSqSum = Sigma1 + Sigma2
    SigmaSqDiff = Sigma1 - Sigma2

    sin2Psi = Psi
    cos2Psi = np.sqrt(1. - sin2Psi * sin2Psi)
    SigmaSqCos = SigmaSqDiff * cos2Psi
    SigmaSqSin = SigmaSqDiff * sin2Psi
    SigmaSqPlus = SigmaSqSum + SigmaSqCos

    fact1b0 = np.exp(-0.25 * SigmaSqPlus)
    fact3b = np.exp(-2. * SigmaSqPlus)
    fact5b = np.exp(-6. * SigmaSqPlus)

    coshHalfSigmaSqSin = np.cosh(0.5 * SigmaSqSin)
    sinhHalfSigmaSqSin = np.sinh(0.5 * SigmaSqSin)

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

    root3 = np.sqrt(3.)

    phiL = 2. * np.pi * LISA_Orbital_Freq * t
    DphiL = phiL - (EclipticLongitude - alpha0)

    FourphiMbar = 4. * ((EclipticLongitude - alpha0) + np.pi / 12.)

    sbM = EclipticLatitude #sin Ecliptic Latitude
    cbM = np.sqrt(1. - sbM * sbM)
    s2bM = 2. * sbM * cbM
    c2bM = cbM * cbM - sbM * sbM

    s3bM = s2bM * cbM + c2bM * sbM
    c3bM = c2bM * cbM - s2bM * sbM

    s5bM = s2bM * c3bM + c2bM * s3bM
    c5bM = c2bM * c3bM - s2bM * s3bM

    sDphiL = np.sin(DphiL)
    cDphiL = np.cos(DphiL)

    s2DphiL = 2. * sDphiL * cDphiL
    c2DphiL = cDphiL * cDphiL - sDphiL * sDphiL

    s3DphiL = s2DphiL * cDphiL + c2DphiL * sDphiL
    c3DphiL = c2DphiL * cDphiL - s2DphiL * sDphiL

    s4DphiL = s3DphiL * cDphiL + c3DphiL * sDphiL
    c4DphiL = c3DphiL * cDphiL - s3DphiL * sDphiL

    s4phiMbar = np.sin(FourphiMbar)
    c4phiMbar = np.cos(FourphiMbar)

    sDphiL_4phiMbar = sDphiL * c4phiMbar + cDphiL * s4phiMbar
    cDphiL_4phiMbar = cDphiL * c4phiMbar - sDphiL * s4phiMbar

    s2DphiL_4phiMbar = sDphiL * cDphiL_4phiMbar + cDphiL * sDphiL_4phiMbar
    c2DphiL_4phiMbar = cDphiL * cDphiL_4phiMbar - sDphiL * sDphiL_4phiMbar

    s3DphiL_4phiMbar = sDphiL * c2DphiL_4phiMbar + cDphiL * s2DphiL_4phiMbar
    c3DphiL_4phiMbar = cDphiL * c2DphiL_4phiMbar - sDphiL * s2DphiL_4phiMbar

    s4DphiL_4phiMbar = sDphiL * c3DphiL_4phiMbar + cDphiL * s3DphiL_4phiMbar
    c4DphiL_4phiMbar = cDphiL * c3DphiL_4phiMbar - sDphiL * s3DphiL_4phiMbar

    s5DphiL_4phiMbar = sDphiL * c4DphiL_4phiMbar + cDphiL * s4DphiL_4phiMbar
    c5DphiL_4phiMbar = cDphiL * c4DphiL_4phiMbar - sDphiL * s4DphiL_4phiMbar

    s6DphiL_4phiMbar = sDphiL * c5DphiL_4phiMbar + cDphiL * s5DphiL_4phiMbar
    c6DphiL_4phiMbar = cDphiL * c5DphiL_4phiMbar - sDphiL * s5DphiL_4phiMbar

    s7DphiL_4phiMbar = sDphiL * c6DphiL_4phiMbar + cDphiL * s6DphiL_4phiMbar
    c7DphiL_4phiMbar = cDphiL * c6DphiL_4phiMbar - sDphiL * s6DphiL_4phiMbar

    s8DphiL_4phiMbar = sDphiL * c7DphiL_4phiMbar + cDphiL * s7DphiL_4phiMbar
    c8DphiL_4phiMbar = cDphiL * c7DphiL_4phiMbar - sDphiL * s7DphiL_4phiMbar

    fact1DphiL = root3 * np.exp(-0.5 * SigmaSqSum)
    fact2DphiL = np.exp(-1.25 * SigmaSqSum + 0.75 * SigmaSqCos)
    fact3DphiL = root3 * np.exp(-2.5 * SigmaSqSum + 2. * SigmaSqCos)
    fact4DphiL = np.exp(-4.25 * SigmaSqSum + 3.75 * SigmaSqCos)

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
    A0 = np.sqrt(0.5 * np.abs(Sum + Diff))
    E0 = np.sqrt(0.5 * np.abs(Sum - Diff))

    sDect = np.sin(2. * (beta0 - alpha0))
    cDect = np.cos(2. * (beta0 - alpha0))

    A = np.abs(cDect * A0 - sDect * E0)
    E = np.abs(sDect * A0 + cDect * E0)

    return A, E












