"""
Bayesian computation of the Power Law Sensitivity (PLS) using JAX.

EXPLANATION: Setting Alpha Parameters for Inverse Gamma Priors
================================================================

This script computes Bayes factors for gravitational wave detection by marginalizing
over the Power Spectral Density (PSD) using inverse gamma priors.

WHAT WE'RE COMPUTING:
--------------------
For each point in our parameter grid (amp[l], slope[l]), we compute:

    log BF(θ_l) = log [p(data|H1, θ_l) / p(data|H0)]

where we have MARGINALIZED OVER THE PSD S analytically:
    - H1: p(data|H1,θ_l) = ∫ p(data|S) p(S|θ_l,H1) dS  where S = S_gw(θ_l) + S_noise
    - H0: p(data|H0)     = ∫ p(data|S) p(S|H0) dS      where S = S_noise

The inverse gamma prior on S allows us to compute these integrals in closed form
using conjugacy with the chi-squared likelihood.


THE ROLE OF ALPHA:
------------------
The inverse gamma prior p(S|θ) = InvGamma(α, β) has:
    - Mean:     E[S] = β/(α-1)
    - Variance: Var[S] = β²/[(α-1)²(α-2)]
    - CV:       CV² = 1/(α-2)

We set β such that the prior is centered on the predicted PSD:
    - For H0: β₀ = (α₀-1) × S_noise
    - For H1: β₁ = (α₁-1) × [S_gw(θ_l) + S_noise]

The parameter α controls HOW TIGHTLY the prior concentrates around this mean:
    - α → 2:  very weak prior (high uncertainty, CV → ∞)
    - α = 3:  CV = 1 (100% coefficient of variation)
    - α = 7:  CV ≈ 0.45 (moderate uncertainty)
    - α >> 10: very tight prior (low uncertainty)


WHAT SHOULD ALPHA REPRESENT?
-----------------------------
The key question: What uncertainty are we encoding in p(S)?

CORRECT INTERPRETATION: Calibration/Systematic Uncertainty
    Alpha should represent uncertainty in our PSD PREDICTION due to:
    - Detector noise parameter uncertainty (A, P)
    - Calibration errors
    - Systematic effects
    - Model uncertainty in the noise curve
    
    This uncertainty is INDEPENDENT of which signal parameters (amp, slope) we're
    testing. At each grid point θ_l, we're asking: "Given that the signal has 
    parameters θ_l, how uncertain am I about the total PSD S_total(θ_l)?"
    
    The answer depends on NOISE uncertainty, not on which signal we're testing!

INCORRECT INTERPRETATION: Signal Parameter Uncertainty
    Alpha should NOT represent uncertainty about which signal parameters are correct.
    That uncertainty is handled by:
    1. Evaluating BF across the entire grid
    2. Combining BF with a prior p(θ) to get posterior p(θ|data)
    
    If we set alpha using variance across the entire parameter grid, we get:
    - Huge variance (signals span 10^-20 to 10^-8)
    - α ≈ 2 (very weak prior)
    - Numerical instabilities and exploding Bayes factors


"""
from bahamas.psd_strain import psd_function as psd
from bahamas.psd_strain import psd_signal as signal
from bahamas.psd_strain import psd_noise as noise
from bahamas.psd_response import response as resp
from bahamas.logger_config import logger

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
from numpyro.distributions import Wishart
from jax import lax
from jax import jit, vmap
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from skimage import measure
from functools import partial
import time


# Enable 64-bit precision in JAX
jax.config.update('jax_enable_x64', True)


L = 8.3  # Arm length of LISA in seconds
c = 299792458 # Speed of light in m/s
H02 = (69.8e3 / (3.08e22))**2  # Hubble constant squared



@partial(jit, static_argnums=(5,))
def compute_bayes_factor(freq, response, psd_noise_base, amp, slope, n_samp, key, dof_used, alpha_arr, batch_size=10):
    """Compute the Bayes factor for given parameters using gamma distributions for independent PSDs."""
    
    def compute_single_bf(l):
        # Extract parameters for this index
        par_l = {'Amp': amp[l], 'slope': slope[l]}
        psd_gw = signal.Omega_pl(freq, par_l) * response  # Shape: (3, n_freq)
        psd_noise = psd_noise_base  # Shape: (3, n_freq)
        psd_total = psd_gw + psd_noise  # Shape: (3, n_freq)
        
        # Initialize Bayes factor accumulator
        bf_accum = 0.0
        k = dof_used / 2
        
        # Process samples in batches
        for batch_start in range(0, n_samp, batch_size):
            batch_end = min(batch_start + batch_size, n_samp)
            batch_size_actual = batch_end - batch_start
            
            # Generate subkeys for the batch
            subkeys = random.split(key, batch_size_actual)
            
            def compute_freq_bf(i, subkey):
                # Sample power spectrum estimate ~ scaled-chi2 with dof_used
                y = jnp.array([
                    random.gamma(k_sub, a=k, shape=()) * 2.0 * psd_total[ch, i] / dof_used
                    for ch, k_sub in enumerate(random.split(subkey, 3))
                ])
                
                # Extract alpha values for this frequency
                alpha = alpha_arr[i]  # Scalar

                
                # Compute beta parameters
                beta_1 = psd_total[:, i] * (alpha - 1)  # Shape: (3,)
                beta_0 = psd_noise[:, i] * (alpha- 1)  # Shape: (3,)
                
                # --- Sum over channels ---
                # Gamma function ratio term (per channel)
                norm_term = jnp.sum(
                    jsp.special.gammaln(alpha + k) - jsp.special.gammaln(alpha) -
                    jsp.special.gammaln(alpha + k) + jsp.special.gammaln(alpha)
                )
                
                # Prior term (per channel)
                model_term = jnp.sum(
                    alpha * jnp.log(beta_1) - alpha * jnp.log(beta_0)
                )
                
                # Data term (per channel)
                data_term = jnp.sum(
                    -(alpha + k) * jnp.log(beta_1 + k * y) +
                    (alpha + k) * jnp.log(beta_0 + k * y)
                )
                
                return norm_term + model_term + data_term
            
            # Compute Bayes factor for the batch
            bf_batch = vmap(lambda subkey: vmap(lambda i: compute_freq_bf(i, subkey))(jnp.arange(len(freq))))(subkeys)
            
            # Sum over frequencies and average over the batch
            bf_accum += jnp.sum(bf_batch, axis=1).sum()
        
        return bf_accum / n_samp
    
    # Compute Bayes factor for all parameter combinations
    BF_tot = vmap(compute_single_bf)(jnp.arange(len(amp)))
    
    return BF_tot / jnp.log(10)

class BahamasBPLS:
    """Class to compute Bayesian Power Law Sensitivity (PLS) using JAX."""
    
    def __init__(self, n_samp=1000, alpha=7, dof=1000, dof_used=None, Tobs=1):
        self.n_samp = int(n_samp)  # Ensure it's a Python int, not JAX array
        self.dof = int(dof)
        self.dof_used = int(dof_used) if dof_used is not None else int(dof)
        year = 31557600.0  # Seconds in a year
        self.Tobs = Tobs * year  # Convert years to seconds
        
        # Frequency setup
        fmin, fmax = 1e-4, 0.029
        self.df = dof / self.Tobs  # Frequency step size
        self.freq = jnp.arange(fmin, fmax, self.df)  # Frequency array
        
        # Compute noise PSD for AET channels
        noise_par = {'A': 2.4, 'P': 7.9}
        psdA = noise.noise(self.freq, noise_par, tdi=0, gen2=False)
        psdE = noise.noise(self.freq, noise_par, tdi=1, gen2=False)
        psdT = noise.noise(self.freq, noise_par, tdi=2, gen2=False)
        self.psd_noise_base = jnp.array([psdA, psdE, psdT])  # Shape: (3, n_freq)
        
        # Compute response for AET channels
        self.response = resp.get_response(self.freq, gen2=False, tdi='AET', equal_arm=True, cross_term=False)
        self.key = random.PRNGKey(1000)
        
        # Prior grid setup
        self.amp = np.random.uniform(-20, -8, 20000)
        self.slope = np.random.uniform(-6, 6, 20000)
        self.dict = []
        for i in range(len(self.amp)):
            self.dict.append({'Amp': self.amp[i], 'slope': self.slope[i]})
        
        # Setup alpha values
        if alpha is not None:
            self.alpha = float(alpha) * jnp.ones_like(self.freq)
        else:
            # Sample noise parameters for uncertainty estimation
           
            A_sampler = np.random.uniform(1, 5, len(self.amp))
            P_sampler = np.random.uniform(5, 10, len(self.amp))
            
            # Compute alpha_0 (noise only)
            alpha = self.map_params_to_inv_gamma(
                tdi=0,
                key=self.key,
                A_samples=A_sampler,
                P_samples=P_sampler
            )
            self.alpha = alpha
            
            # Plot comparison
            fig = plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.hist(alpha, label='Noise only', bins=20, alpha=0.7)
            plt.hist(alpha, label='Noise + GW', bins=20, alpha=0.7)
            plt.xlabel('alpha')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.semilogx(self.freq, alpha, label='Noise only')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('alpha')
            plt.legend()
            plt.tight_layout()
            plt.savefig('alpha_prior_comparison.png', dpi=300)
            logger.info("Saved alpha prior comparison plot to 'alpha_prior_comparison.png'")
    
    def compute_psd_samples(self, tdi, A_samples, P_samples, Amp_samples=None, slope_samples=None):
        """
        Compute PSD samples from parameter samples.
        
        Args:
            tdi: TDI channel index (0=A, 1=E, 2=T)
            A_samples: Noise parameter A samples
            P_samples: Noise parameter P samples
            Amp_samples: GW amplitude samples (optional)
            slope_samples: GW spectral slope samples (optional)
        
        Returns:
            psd_samples: array of shape (n_samples, n_freq)
        """
        n_samples = len(A_samples)
        psd_samples = []
        
        for i in range(n_samples):
            # Noise PSD
            noise_par = {'A': A_samples[i], 'P': P_samples[i]}
            psd_noise = noise.noise(self.freq, noise_par, tdi=tdi, gen2=False)
            
            # Add signal if provided
            if Amp_samples is not None and slope_samples is not None:
                # Sample randomly from the GW parameter arrays
                idx = np.random.randint(0, len(Amp_samples))
                gw_par = {'Amp': Amp_samples[idx], 'slope': slope_samples[idx]}
                psd_gw = signal.Omega_pl(self.freq, gw_par) * self.response[tdi]
                psd_total = psd_gw + psd_noise
            else:
                psd_total = psd_noise
            
            psd_samples.append(psd_total)
        
        return jnp.array(psd_samples)
    
    def map_params_to_inv_gamma(self, tdi, key, 
                                A_samples, P_samples):
        """
        Map priors on physical parameters to frequency-dependent inverse-Gamma priors for PSD.
        """

        # Compute PSD samples - FIXED VERSION
        def compute_psd_sample(a, p, amp=0., slope=0.):
            par_noise = {'A': a, 'P': p}
            S_n = noise.noise(self.freq, par_noise, tdi, gen2=False)

            return S_n

        compute_vmap = vmap(compute_psd_sample, in_axes=(0,0,None,None))
        PSD_samples = compute_vmap(A_samples, P_samples, None, None)

        # Compute mean and variance per frequency
        mu = jnp.mean(PSD_samples, axis=0)
        sigma2 = jnp.var(PSD_samples, axis=0, ddof=1)

        # Map to inverse-Gamma parameters
        # CV² = σ²/μ², so α = 2 + 1/CV² = 2 + μ²/σ²
        alpha = 2.0 + mu**2 / (sigma2 )  
        # Clip to reasonable range
        alpha = jnp.clip(alpha, 2.5, 50.0)

        return alpha


    
    def run(self):
        """Run the Bayesian PLS computation."""
        logger.info("Starting Bayesian PLS computation...")
        time_start = time.time()
        
        self.BF_tot = compute_bayes_factor(
            self.freq,
            self.response,
            self.psd_noise_base,
            self.amp,  # Pass amp array
            self.slope,  # Pass slope array
            self.n_samp,  # This is static
            self.key,  # This is NOT static (it's a JAX array)
            self.dof_used,  # This is static
            self.alpha,  
        )
        
        time_end = time.time()
        logger.info(f"Bayesian PLS computation completed in {time_end - time_start:.2f} seconds.")
        logger.info(self.BF_tot)
        return self.BF_tot


    def snr2(self, Sh, Sn):
        """Compute the squared Signal-to-Noise Ratio (SNR^2)."""
        snr_squared = 4 * jnp.sum((Sh / Sn)**2) * self.Tobs * self.df

        return snr_squared
    
    def snr(self, par):
        """Compute the Signal-to-Noise Ratio (SNR)."""
        psd_gw = signal.Omega_pl(self.freq, par) * self.response
        psd_noise = self.psd_noise_base

        sn = 0
        for i in range(3):
            psd_gw_i = psd_gw[i, :]
            psd_noise_i = psd_noise[i, :]
            sn += self.snr2(psd_gw_i, psd_noise_i)
        return jnp.sqrt(sn)

    def plot_results(self):
        import matplotlib.colors as mcolors
        logger.info("Plotting results...")


        #snr_values = np.zeros_like(self.amp)
        
        #for i in range(snr_values.shape[0]):
        #    par = {'Amp': self.amp[i], 'slope': self.slope[i]}
        #    snr_values[i] = self.snr(par)

        #fig, ax = plt.subplots(figsize=(5,4))
        #scat = ax.scatter(self.slope, self.amp, c = np.log10(snr_values.flatten()), vmin = np.log10(10), vmax = np.log10(3000), rasterized = True, marker = '.', cmap = 'PuBu')
        #cbar = fig.colorbar(scat,
        #            ax=ax, #orientation='horizontal',
        #            label="$\\rm{SNR}$")

        #cbar.set_ticks(np.log10(np.array([  10,   25,   70,  170,
        #        450, 1000 , 3000])), labels=np.array([  10,   25,   70,  170,
        #        450, 1000 , 3000]))
        #ax.set_xlabel('Slope (gamma)')
        #ax.set_ylabel('Amplitude (log10 Omega)')
        #ax.set_title('SNR Contour Plot')
        #plt.savefig('snr_contour_plot.png', dpi=300)


        #grid_bf = griddata((self.slope, self.amp), self.BF_tot, (slope_grid, amp_grid), method='cubic')
        
        
        fig, ax = plt.subplots(figsize=(5,4))
        scat = ax.scatter(self.slope, self.amp, c = self.BF_tot.flatten(), vmin = -5, vmax = 5, rasterized = True, marker = '.', cmap = 'RdBu_r')
        cbar = fig.colorbar(scat,
                    ax=ax, #orientation='horizontal',
                    label="Bayes Factor (log10)")
        ax.set_xlabel('Slope (gamma)')
        ax.set_ylabel('Amplitude (log10 Omega)')
        ax.set_title('Bayes Factor Contour Plot')
        fig.savefig('bayes_factor_contour_plot.png', dpi=300)



        
   




