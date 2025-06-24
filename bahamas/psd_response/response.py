"""
This module provides functions and classes to compute the response of LISA (Laser Interferometer Space Antenna) 
to gravitational waves, including TDI (Time-Delay Interferometry) responses and simplified response functions.

Functions:
    cartesian_to_spherical(vec): Converts a 3D Cartesian vector to spherical coordinates.
    sinc(SC, freqs, t, khat, link): Computes the sinc function for LISA's single-arm response.
    phase(SC, freqs, t, khat, link): Computes the phase response for a given link.
    antenna_pattern(SC, t, khat, link, polarization): Computes the antenna pattern for a given polarization.
    get_response(freq, tdi): Computes the response for a specific TDI channel (A, E, or T).
    raa(freqs, L): Simplified response function for isotropic, stationary SGWB (A channel).
    rtt(freqs, L): Simplified response function for isotropic, stationary SGWB (T channel).

Classes:
    SGWBResponseStationary: Computes the stationary response of LISA to an isotropic SGWB.
    Mtdi: Handles TDI matrix computations for different generations (1 or 2).
    TDImatrix: Combines TDI matrices with response functions to compute the full response matrix.

Dependencies:
    - NumPy
    - Healpy
    - SciPy
    - orbits (custom module)
"""
from bahamas.psd_response import orbits as orb

import numpy as np
import healpy as hp
import itertools as itt
from scipy.interpolate import CubicSpline
      

# For now, let's define this two.
# They are the wrapped version of Antoine's code, which is in C++.

import math

def cartesian_to_spherical(vec):
    """Converter from 3D cartesian coordinates to 2D spherical coordinates (assuming unitary radius)

    Args:
        vec (float): 3D vector in cartesian coordinates

    Returns:
        numpy.array: 2D vector in spherical coordinates
    """
    
    x = vec[0]
    y = vec[1]
    z = vec[2]
    theta = math.atan2(math.sqrt(x**2 + y**2), z)
    phi = math.atan2(y, x)
    return theta, phi

def sinc(SC, freqs, t, khat, link):
    """
    Sinc function for LISA single arm response 
    given link, frequency, time, and unit vector in the direction of the source

    Args:
        SC (SpacecraftOrbit): Object containing the spacecraft orbit description
        freqs (numpy.array): Frequencies at which the links unit vectors are evaluated
        t (float): Time at which the response is calculated
        khat (numpy.array): Unit vector in the direction of the source in cartesian coordinates
        link (int): Link number (1,2,3,-1,-2,-3) 

    Returns:
        numpy.array: response as a function of frequency
    """
    SC = orb.SpacecraftOrbit()
    rhatl, L = SC.link_versor(t,link)
    return np.sinc(np.pi*L*freqs*(1-np.dot(khat,rhatl)))

def phase(SC, freqs, t, khat, link):
    """
    Computes the phase response for a given link.

    Args:
        SC (SpacecraftOrbit): Object containing the spacecraft orbit description.
        freqs (array-like): Frequencies at which the response is evaluated.
        t (float): Time at which the response is calculated.
        khat (array-like): Unit vector in the direction of the source.
        link (int): Link number (1, 2, 3, -1, -2, -3).
        equal_arm (float, optional): Length of the arm if not using the spacecraft orbit. Defaults to False.

    Returns:
        array-like: Phase response as a function of frequency.
    """
    ind = [0,1,2,2,1,0]

    pos = SC.SC_positions(t)
    
    if link > 0:
        xs = int(ind[link])
        xr = int(ind[link+1])    
    if link < 0:
        xs = int(ind[-link+1])
        xr = int(ind[-link])
    __, L = SC.link_versor(t,link)
    

    return np.exp(-1j*np.pi*freqs*(L + np.dot(khat,(pos[xs] + pos[xr])))) 
    
def fract_freq(SC, freqs, t, khat, link):
    """
    Computes the fractional frequency factor for a given link.
    Args:
        SC (SpacecraftOrbit): Object containing the spacecraft orbit description.
        freqs (array-like): Frequencies at which the response is evaluated.
        t (float): Time at which the response is calculated.
        khat (array-like): Unit vector in the direction of the source.
        link (int): Link number (1, 2, 3, -1, -2, -3).
    Returns:
        array-like: Fractional frequency factor as a function of frequency.
    """
    ind = [0,1,2,2,1,0]

    pos = SC.SC_positions(t)
    
    if link > 0:
        xs = int(ind[link])
        xr = int(ind[link+1])    
    if link < 0:
        xs = int(ind[-link+1])
        xr = int(ind[-link])
    __, L = SC.link_versor(t,link)

    return 2 * np.pi * freqs * 1j * L

def antenna_pattern(SC, t, khat, link, polarization):
    """
    Computes the antenna pattern for a given polarization.

    Args:
        SC (SpacecraftOrbit): Object containing the spacecraft orbit description.
        t (float): Time at which the response is calculated.
        khat (array-like): Unit vector in the direction of the source.
        link (int): Link number (1, 2, 3, -1, -2, -3).
        polarization (str): Polarization type ('plus' or 'cross').

    Returns:
        float: Antenna pattern value.
    """
    rhatl, L = SC.link_versor(t,link)

    th,ph = cartesian_to_spherical(khat) #covnert cartesian component of khat in spherical 
    uhat = np.array([np.cos(th)*np.cos(ph),np.cos(th)*np.sin(ph),-np.sin(th)])
    vhat = np.array([np.sin(ph),-np.cos(ph),0])
    if polarization == "plus":
        return np.dot(uhat,rhatl)**2 - np.dot(vhat,rhatl)**2 
    if polarization == "cross":
        return 2*np.dot(uhat,rhatl)*np.dot(vhat,rhatl) 

class SGWBResponseStationary():
    """
        Initialize the SGWBResponseStationary class.

        Args:
            freqs (array-like): Frequency array.
            t (float): Time at which the response is calculated.
            nside (int): Healpy nside parameter for pixelization.
            nest (bool): Whether to use nested pixel ordering.
            equal_arm (float, optional): Length of the arm if not using the spacecraft orbit. Defaults to False.
    """
    def __init__(self, freqs, t=21037939, nside=4, nest=False): #
        self.nside = nside
        self.npixels = 12 * self.nside**2
        self.nest = nest
        # Calculate total solid angle divided by number of pixels
        self.pixelsize =  4*np.pi / hp.nside2npix(nside)
        self.t = t
        self.khats = self.get_khats()
     
        self.linkvector = np.array([1,2,3,-1,-2,-3])
        self.freqs = freqs
        self.shape = (6,6,len(freqs))
        self.SC = orb.SpacecraftOrbit()
    
    
    def get_khats(self):
        """
        Generates all khats in Cartesian coordinates based on Healpy pixelization.

        Returns:
            np.ndarray: Array of khats in Cartesian coordinates.
        """
        return - np.array(hp.pixelfunc.pix2vec(self.nside, np.arange(self.npixels), nest=self.nest)) # This is the khat array vector

    def Response(self, polarization='plus'): 
        """
        Computes the response matrix for a given polarization.

        Args:
            polarization (str): Polarization type ('plus' or 'cross').

        Returns:
            np.ndarray: Response matrix.
        """
        BigArray=np.zeros(self.shape, dtype=complex)
        if polarization != 'plus' and polarization != 'cross':
            raise ValueError("Polarization must be either 'plus' or 'cross'")
        for (i,j) in itt.product(range(6), range(6)):
            if i<=j:
                l = self.linkvector[i] 
                lp  = self.linkvector[j]
                BigArray[i,j,:] = self.GbarGbarConj(l,lp,polarization)
                if i != j:
                    BigArray[j,i,:] = np.conjugate(BigArray[i,j,:])

        return BigArray
        
        
    def GbarGbarConj(self, l,lp,p):
        """
        Computes the conjugate of the GbarGbar response.

        Args:
            l (int): Link index.
            lp (int): Link index for the second arm.
            polarization (str): Polarization type ('plus' or 'cross').

        Returns:
            np.ndarray: Conjugate response array.
        """
        GbarGbarConjK = np.zeros_like(self.freqs,dtype=complex)

        for k in (self.khats.T):
            ### Easy to parallelize using multiprocessing.map.
            y = self.pixelsize*((0.5*antenna_pattern(self.SC, self.t,k,l, p)*
                               phase(self.SC, self.freqs, self.t, k,l)*
                               sinc(self.SC, self.freqs, self.t, k,l) *
                                fract_freq(self.SC, self.freqs, self.t, k,l)
                               )*
                               np.conjugate(
                                   (0.5*antenna_pattern(self.SC, self.t,k,lp, p)*
                                    phase(self.SC,self.freqs, self.t, k,lp)*
                                    sinc(self.SC, self.freqs, self.t, k,lp)) *
                                    fract_freq(self.SC, self.freqs, self.t, k,lp)
                                    ) / (4 * np.pi)
            )

            GbarGbarConjK += y                          
        return GbarGbarConjK
  

class Mtdi():
    """
    Handles TDI matrix computations for different generations (1 or 2).
    """
    def __init__(self,freq,gen2=1, L_map=None):
        """
        Initialize the Mtdi class.

        Args:
            freq (array-like): Frequency array.
            gen (int): Generation of TDI (1 or 2).
        """
        self.freq = freq
        self.gen2 = gen2
        self.shape = (3, 6, len(freq))
        self.L_map = L_map or {
            '12': 8.3, '23': 8.3, '31': 8.3,
            '13': 8.3, '32': 8.3, '21': 8.3
        }
        self.link_indices = ['12', '23', '31', '13', '32', '21']
        


    def delay(self, f, *link_labels):
        """Compute compound delay operator D_{i...}(f) = exp(-2πif (L1 + L2 + ...)).
        Args:
            f (float): Frequency.
            link_labels (list): List of link labels.
        Returns:
            complex: Compound delay operator.
        """
        total_L = sum(self.L_map[lbl] for lbl in link_labels)
        return np.exp(-2j * np.pi * f * total_L)

    def get_M_TDI1(self, f):
        """Build the 3x6 M_TDI(f) matrix for Michelson observables (X, Y, Z).
        Args:
            f (float): Frequency.
        Returns:
            np.ndarray: TDI matrix.
        """
        # Initialize the TDI matrix
        M = np.zeros((3, 6), dtype=complex)

        # Map: link -> column index in y vector
        idx = {l: i for i, l in enumerate(self.link_indices)}

        # ---- X observable (1st row) ----
        M[0][4] += 1
        M[0][1] += self.delay(f, '13')
        M[0][2] += self.delay(f, '13', '31')
        M[0][5] += self.delay(f, '13', '31', '12')

        M[0][2] -= 1
        M[0][5] -= self.delay(f, '12')
        M[0][4] -= self.delay(f, '12', '21')
        M[0][1] -= self.delay(f, '12', '21', '13')

        # ---- Y observable (2nd row) ---- (cycle 1→2→3)
        M[1][5] += 1
        M[1][2] += self.delay(f, '21')
        M[1][0] += self.delay(f, '21', '12')
        M[1][3] += self.delay(f, '21', '12', '23')

        M[1][0] -= 1
        M[1][3] -= self.delay(f, '23')
        M[1][5] -= self.delay(f, '23', '32')
        M[1][2] -= self.delay(f, '23', '32', '21')

        # ---- Z observable (3rd row) ---- (cycle 2→3→1)
        M[2][3] += 1
        M[2][0] += self.delay(f, '32')
        M[2][1] += self.delay(f, '32', '23')
        M[2][4] += self.delay(f, '32', '23', '31')

        M[2][1] -= 1
        M[2][4] -= self.delay(f, '31')
        M[2][3] -= self.delay(f, '31', '13')
        M[2][0] -= self.delay(f, '31', '13', '32')

        return M

    def get_M_TDI2(self, f):
        """
        Build the 3x6 M_TDI(f) matrix for second-gen2eration Michelson observables (X2, Y2, Z2).

        Args:
            f (float): Frequency.
        
        Returns:
            np.ndarray: Second-gen2eration TDI matrix.
        """
        # First-gen2eration Michelson matrix
        M1 = self.get_M_TDI1(f)

        # Apply second-gen2 delay factors
        X_factor = 1 - self.delay(f, '31', '31', '12', '12')
        Y_factor = 1 - self.delay(f, '12', '12', '23', '23')
        Z_factor = 1 - self.delay(f, '23', '23', '31', '31')

        # Initialize second-gen2 matrix
        M2 = np.zeros_like(M1, dtype=complex)
        M2[0] = X_factor * M1[0]
        M2[1] = Y_factor * M1[1]
        M2[2] = Z_factor * M1[2]

        return M2
    
    
    def Mxyz(self, f, gen2=False):
        """
        Returns the TDI matrix (X, Y, Z) for the specified gen2eration.

        Args:
            f (float): Frequency in Hz.
            gen2 (int): gen2eration of TDI (1 or 2).

        Returns:
            np.ndarray: A 3x6 complex matrix mapping link measurements to TDI variables.
        """
        if gen2 == False:
            return self.get_M_TDI1(f)
        else:
            return self.get_M_TDI2(f)

    def Maet(self, f, gen2=None):
        """
        Converts the TDI matrix from XYZ to AET coordinates.

        Args:
            f (float): Frequency in Hz.
            gen2 (int, optional): TDI gen2eration. If None, uses self.gen2.

        Returns:
            np.ndarray: 3x6 complex matrix for AET channels.
        """
        gen2eration = gen2 if gen2 is not None else self.gen2
        Mxyz = self.Mxyz(f, gen2=gen2eration)
        M = np.empty((3, 6), dtype=complex)
        
        # A = (Z - X) / sqrt(2)
        M[0] = (Mxyz[2] - Mxyz[0]) / np.sqrt(2)
        
        # E = (X - 2Y + Z) / sqrt(6)
        M[1] = (Mxyz[0] - 2 * Mxyz[1] + Mxyz[2]) / np.sqrt(6)
        
        # T = (X + Y + Z) / sqrt(3)
        M[2] = (Mxyz[0] + Mxyz[1] + Mxyz[2]) / np.sqrt(3)
        
        return M
    
        
    def Mwrap(self,type = 'AET'):
        """
        Computes the TDI transfer matrix for all frequencies.

        Args:
            type (str): TDI type ('AET' or 'XYZ').

        Returns:
            np.ndarray: A (3, 6, Nf) complex array where Nf = len(self.freq),
                        representing the TDI matrix at all frequencies.
        """
        M = np.empty(self.shape,complex) 
        for i,f in enumerate(self.freq):
            if type == 'AET':
                M[:,:,i] = self.Maet(f)
            if type == 'XYZ':
                M[:,:,i] = self.Mxyz(f,gen2=self.gen2)

        return M
    

    

class TDImatrix():
    """
    Combines TDI matrices with response functions to compute the full response matrix.
    """
    def __init__(self,freq,gen2=1,nside=4,type='AET', equal_arm=False):
        self.freq = freq
        self.gen2 = gen2
        self.type = type
        self.nside = nside
        myR = SGWBResponseStationary(freqs=self.freq,nside=self.nside)
        Rp, Rc = myR.Response(polarization='plus'), myR.Response(polarization='cross') 
        self.R = Rp + Rc

        if equal_arm is not False:
            self.L_map = {
                '12': 8.3, '23': 8.3, '31': 8.3,
                '13': 8.3, '32': 8.3, '21': 8.3
            }
        else:
            link = np.array([1, 2, 3, -1, -2, -3])
            ind = ['32', '13', '21', '23', '31', '12'] 
            SC = orb.SpacecraftOrbit()
            self.L_map = {ind[i]: SC.link_versor(t = 0, link = link[i])[1] for i in range(len(link))}
         
        mtM = Mtdi(freq=self.freq, gen2=self.gen2)
        self.MTDI = mtM.Mwrap(type=self.type)
    
    def matrix(self):
    
        return (np.einsum('ijk,jlk,klx->ixk',self.MTDI,self.R,np.conjugate(self.MTDI.T))) 

def get_response(freq,  gen2=False, equal_arm=False, cross_term = False, tdi = 'AE'):
    """
    Computes the response for a specific TDI channel (A, E, or T).

    Args:
        freq (array-like): Frequency array.
        tdi (str): TDI channel ('A', 'E', or 'T').
        gen2 (int): TDI gen2eration (1 or 2).
        equal_arm (float, optional): Length of the arm if not using the spacecraft orbit. Defaults to False.
        cross_term (bool, optional): Whether to include cross terms in the response. Defaults to False.
        tdi (str): TDI type ('AE' or 'AET').
        
    Returns:
        array-like: Response for the specified TDI channel.
    """
    if freq[0] == 0:
        freq[0] = freq[1]
    log_f = np.linspace(np.min(np.log10(freq[0])), np.max(np.log10(freq[-1])), 1000)

    TDI = TDImatrix(freq=10**log_f, gen2=gen2, type='AET', equal_arm= equal_arm)
    rm = TDI.matrix()

    if gen2 == False or equal_arm is not False:
        cross_term = False
    
    if tdi == 'AET':
        ntdi = 3
        
    elif tdi == 'AE':
        ntdi = 2
        
    else:
        raise ValueError("TDI must be 'AE' or 'AET'")

    list_resp = []
    if cross_term:
        for i in range(ntdi):
            resp = np.abs(rm[i][i])
            spline = CubicSpline(log_f, np.log10(resp), bc_type='natural')
            response = spline(np.log10(freq))
            list_resp.append(10**response)
            for j in range(i+1, ntdi):
                resp = np.abs(rm[i][j])
                spline = CubicSpline(log_f, np.log10(resp), bc_type='natural')
                response = spline(np.log10(freq))
                list_resp.append(10**response)

    else:
        for i in range(ntdi):
            resp = np.abs(rm[i][i])
            spline = CubicSpline(log_f, np.log10(resp), bc_type='natural')
            response = spline(np.log10(freq))
            list_resp.append(10**response)

    return np.array(list_resp)

