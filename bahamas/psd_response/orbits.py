"""
This module contains the class SpacecraftOrbit, which is used to compute the positions of spacecraft in a LISA-like formation.
The class is initialized with the following parameters:
- L_init: arm length (default: 8.3)
- phi_init: initial phase (default: 0)
- lambda_init: initial angle (default: 0)
- R_init: distance from the Sun (default: 1 AU)
- orbitFreq_init: orbital frequency (default: 1.99106488756578923121e-7)
- ecc_order_init: order of the eccentricity expansion (default: 2)
- The class has methods to compute the positions of the spacecraft at a given time and to compute the versor of the links between spacecraft.
"""

import numpy as np

twoPiThird = 2.09439510239319549231
meanArmlength = 8.3#8.33910237995380123939
twoPiPerYear= 1.99106488756578923121e-7
year = 2*np.pi/twoPiPerYear
astroUnit= 499.00478383615641191347
sqrt_3 = 1.73205080756887729353

class SpacecraftOrbit():
    """
    Class to compute the positions of spacecraft in a LISA-like formation.
    The class is initialized with the following parameters:
    - L_init: arm length (default: 8.3)
    - phi_init: initial phase (default: 0)
    - lambda_init: initial angle (default: 0)
    - R_init: distance from the Sun (default: 1 AU)
    - orbitFreq_init: orbital frequency (default: 1.99106488756578923121e-7)
    - ecc_order_init: order of the eccentricity expansion (default: 2)
    """
    def __init__(self, L_init=meanArmlength, phi_init= 0., 
                 lambda_init=0., R_init=astroUnit, 
                 orbitFreq_init=twoPiPerYear, 
                 ecc_order_init=2):
        self.L = L_init
        self.one_over_L = 1./self.L
        self.phi = phi_init
        self.lam = lambda_init
        self.R = R_init
        self.orbitFreq = orbitFreq_init
        self.orbitLinearVelocity = self.R*self.orbitFreq
        self.orbitLinearAccel = self.orbitFreq*self.orbitLinearVelocity
        self.e = self.L/(2.*sqrt_3*self.R)
        self.ecc_order = ecc_order_init
        self.compute_factors()

    def compute_factors(self):
        """
        Precompute the factors for the spacecraft positions.
        """
        indices = np.arange(3)
        b = indices*twoPiThird + self.lam
        # sin-cosine
        z = np.tan(0.5*b)
        z2 = z**2
        fact = 1./(1. + z2)
        self.sb = 2.*z*fact
        self.cb = (1. - z2)*fact
        self.sb2 = self.sb**2
            
    def SC_positions(self, t):
        """
        Compute the positions of the spacecraft at a given time.
        Args:
            t (float): Time in seconds.
        Returns:
            np.ndarray: Positions of the spacecraft in a 3x3 array.
        """
        # Arrange spacecrafts in a 3x3 grid, rows are spacecrafts, columns are x,y,z
        self.SC = np.zeros((3,3))
        alpha = t*self.orbitFreq + self.phi        
        # sin-cosine
        z = np.tan(0.5*alpha)
        z2 = z*z
        fact = 1./(1. + z2)
        sa = 2.*z*fact
        ca = (1. - z2)*fact
        sa2 = sa*sa
        ca2 = ca*ca
        saca = sa*ca
        c2a = ca2 - sa2
  
        # precompute the spacecraft-specific factors
        if self.ecc_order >= 2:
            fact_x3 = -0.25*ca*(4. - 3.*c2a)
            fact_x4 = -0.5*sa*(1. - 3.*c2a)
            fact_x5 = -fact_x3
            fact_x6 = -1.25*ca
      
            fact_y3 = 0.25*sa*(4. + 3.*c2a)
            fact_y4 = -0.5*ca*(1. + 3.*c2a)
            fact_y5 = -fact_y3
            fact_y6 = -1.25*sa
      
            fact_z1 = -0.5*c2a
            fact_z2 = -2.*saca
            fact_z3 = -fact_z1
            fact_z4 = 1.5
        if self.ecc_order >= 1:
            fact_x1 = 0.5*(c2a - 3.)
            fact_x2 = saca
            fact_y1 = saca
            fact_y2 = -0.5*(c2a + 3.)

        # compute the spacecraft positions
        if self.ecc_order >=  2:
            self.SC[:,0] += (self.cb*(fact_x3*self.cb + fact_x4*self.sb) + fact_x5*self.sb2 + fact_x6)
            self.SC[:,1] += (self.cb*(fact_y3*self.cb + fact_y4*self.sb) + fact_y5*self.sb2 + fact_y6)
            self.SC[:,2] += (self.cb*(fact_z1*self.cb + fact_z2*self.sb) + fact_z3*self.sb2 + fact_z4)

            self.SC[:,0] *= self.e
            self.SC[:,1] *= self.e
            self.SC[:,2] *= self.e
            
        if self.ecc_order >= 1:

            self.SC[:,0] += fact_x1*self.cb + fact_x2*self.sb
            self.SC[:,1] += fact_y1*self.cb + fact_y2*self.sb
            self.SC[:,2] += -ca*self.cb - sa*self.sb
                
            self.SC *= self.e
                
        if self.ecc_order >= 0:
            self.SC[:,0] += ca
            self.SC[:,1] += sa
                
            self.SC[:,0] *= self.R
            self.SC[:,1] *= self.R
            self.SC[:,2] *= sqrt_3*self.R
        return self.SC
    

    def link_versor(self,t=0,link=1):
        """
        Compute the versor of the link between two spacecraft.
        Args:
            t (float): Time in seconds.
            link (int): Link number (-1, -2, -3 for opposite direction).
        Returns:
            tuple: (versor, distance)
        """
        # link = 1,2,3 for the three links, -1,-2,-3 for the opposite direction
        orbits = self.SC_positions(t)
        if link == 1:
            rhat = (orbits[1]-orbits[2])

        if link == 2:
            rhat = (orbits[2]-orbits[0])

        if link == 3:
            rhat = (orbits[0]-orbits[1])
        
        if link == -1:
            rhat = -(orbits[1]-orbits[2])

        if link == -2:
            rhat = -(orbits[2]-orbits[0])

        if link == -3:
            rhat = -(orbits[0]-orbits[1])

        return rhat/np.linalg.norm(rhat), np.linalg.norm(rhat)
    

if __name__ == "__main__":
    """
    Example usage of the SpacecraftOrbit class.
    """
    # Creates the object
    mySC = SpacecraftOrbit()
    
    # Prepare time and array to store
    times = np.linspace(0,year,100)
    orbits = np.zeros((len(times), 3, 3))

    for i, t in enumerate(times):
        orbits[i,:,:] = mySC.SC_positions(t)

    