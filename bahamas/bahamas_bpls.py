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


RECOMMENDED USAGE:
-----------------
# Standard usage (noise uncertainty only - RECOMMENDED)
bpls = BahamasBPLS(n_samp=1000, alpha=None, alpha_0=None)
BF = bpls.run()
bpls.plot_results()

# With signal model uncertainty (advanced)
bpls = BahamasBPLS(n_samp=1000, alpha=None, alpha_0=None,
                   include_signal_uncertainty=True,
                   signal_model_uncertainty=0.15)
BF = bpls.run()

# Fixed alpha (fastest, for testing)
bpls = BahamasBPLS(n_samp=1000, alpha=7.0, alpha_0=7.0)
BF = bpls.run()
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
from jax import jit, vmap
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import time


# Enable 64-bit precision in JAX
jax.config.update('jax_enable_x64', True)


L = 8.3  # Arm length of LISA in seconds
c = 299792458  # Speed of light in m/s
H02 = (69.8e3 / (3.08e22))**2  # Hubble constant squared


@partial(jit, static_argnums=(5,))
def compute_bayes_factor(freq, response, psd_noise_base, amp, slope, n_samp, key, 
                        dof_used, alpha_arr, alpha_0_arr, batch_size=10):
    """
    Compute the Bayes factor for given parameters using inverse gamma priors on PSD.
    
    Args:
        freq: Frequency array (n_freq,)
        response: Response function (3, n_freq)
        psd_noise_base: Baseline noise PSD (3, n_freq)
        amp: Amplitude array for grid (n_grid,)
        slope: Slope array for grid (n_grid,)
        n_samp: Number of samples (static)
        key: JAX random key
        dof_used: Degrees of freedom (static)
        alpha_arr: Shape parameter for H1 (n_freq,)
        alpha_0_arr: Shape parameter for H0 (n_freq,)
        batch_size: Batch size for processing (static)
    
    Returns:
        BF_tot: Log10 Bayes factors for each grid point (n_grid,)
    """
    
    def compute_single_bf(l):
        """Compute BF for a single parameter point."""
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
                """Compute BF contribution for a single frequency."""
                # Sample power spectrum estimate ~ scaled-chi2 with dof_used
                # y ~ chi2(dof_used) * S_total / dof_used
                y = jnp.array([
                    random.gamma(k_sub, a=k, shape=()) * 2.0 * psd_total[ch, i] / dof_used
                    for ch, k_sub in enumerate(random.split(subkey, 3))
                ])
                
                # Get alpha values for this frequency
                alpha_1 = alpha_arr[i]      # H1: signal + noise
                alpha_0 = alpha_0_arr[i]    # H0: noise only
                
                # Compute beta parameters
                # beta = (alpha - 1) * mean_PSD
                beta_1 = psd_total[:, i] #* (alpha_1 - 1)  # Shape: (3,) for 3 channels
                beta_0 = psd_noise[:, i] #* (alpha_0 - 1)  # Shape: (3,)
                
                # Sum over 3 channels (A, E, T)
                
                # Gamma function ratio term
                norm_term = 0
                #jnp.sum(
                #    jsp.special.gammaln(alpha_1 + k) - jsp.special.gammaln(alpha_1) -
                #    jsp.special.gammaln(alpha_0 + k) + jsp.special.gammaln(alpha_0)
                #)
                
                # Prior term (log ratio of prior normalizations)
                model_term = jnp.sum(
                    alpha_1 * jnp.log(beta_1) - alpha_0 * jnp.log(beta_0)
                )
                
                # Data term (log ratio of posterior normalizations)
                data_term = jnp.sum(
                    -(alpha_1 + k) * jnp.log(beta_1 + k * y) +
                    (alpha_0 + k) * jnp.log(beta_0 + k * y)
                )
                
                return norm_term + model_term + data_term
            
            # Compute Bayes factor for the batch
            # Outer vmap: over batch of samples
            # Inner vmap: over frequencies
            bf_batch = vmap(
                lambda subkey: vmap(lambda i: compute_freq_bf(i, subkey))(jnp.arange(len(freq)))
            )(subkeys)
            
            # Sum over frequencies and average over the batch
            bf_accum += jnp.sum(bf_batch, axis=1).sum()
        
        # Average over samples
        return bf_accum / n_samp
    
    # Compute Bayes factor for all parameter combinations
    BF_tot = vmap(compute_single_bf)(jnp.arange(len(amp)))
    
    # Convert to log10
    return BF_tot / jnp.log(10)


class BahamasBPLS:
    """Class to compute Bayesian Power Law Sensitivity (PLS) using JAX."""
    
    def __init__(self, n_samp=1000, alpha=None, alpha_0=None, 
                 dof=1000, dof_used=None, Tobs=1, 
                 n_grid_points=20000, 
                 A_range=(0, 5), P_range=(5, 10),
                 amp_range=(-15, -8), slope_range=(-5, 5)):
        """
        Initialize Bayesian PLS computation.
        
        Args:
            n_samp: Number of samples for Monte Carlo integration
            alpha: Fixed alpha for H1, or None for adaptive (based on noise uncertainty)
            alpha_0: Fixed alpha for H0, or None for adaptive (based on noise uncertainty)
            include_signal_uncertainty: If True, make alpha depend on signal strength
            signal_model_uncertainty: Fractional uncertainty in GW signal model (e.g., 0.1 = 10%)
            dof: Degrees of freedom for chi-squared distribution
            dof_used: Effective degrees of freedom used (defaults to dof)
            Tobs: Observation time in years
            n_grid_points: Number of points in parameter grid
            amp_range: Range for amplitude parameter (log10)
            slope_range: Range for slope parameter
        """
        self.n_samp = int(n_samp)
        self.dof = int(dof)
        self.dof_used = int(dof_used) if dof_used is not None else int(dof)

        
        year = 31557600.0  # Seconds in a year
        self.Tobs = Tobs * year  # Convert years to seconds
        
        logger.info("=" * 70)
        logger.info("INITIALIZING BAHAMAS BAYESIAN PLS")
        logger.info("=" * 70)
        logger.info(f"Observation time: {Tobs} years ({self.Tobs:.2e} seconds)")
        logger.info(f"Degrees of freedom: {self.dof} (using: {self.dof_used})")
        logger.info(f"Monte Carlo samples: {self.n_samp}")
        
        # Frequency setup
        fmin, fmax = 1e-4, 0.029
        self.df = self.dof / self.Tobs  # Frequency step size
        self.freq = jnp.arange(fmin, fmax, self.df)  # Frequency array
        logger.info(f"Frequency range: [{fmin:.2e}, {fmax:.2e}] Hz")
        logger.info(f"Frequency bins: {len(self.freq)} (df = {self.df:.2e} Hz)")
        
        # Compute noise PSD for AET channels
        noise_par = {'A': 2.4, 'P': 7.9}
        psdA = noise.noise(self.freq, noise_par, tdi=0, gen2=False)
        psdE = noise.noise(self.freq, noise_par, tdi=1, gen2=False)
        psdT = noise.noise(self.freq, noise_par, tdi=2, gen2=False)
        self.psd_noise_base = jnp.array([psdA, psdE, psdT])  # Shape: (3, n_freq)


        # Compute response for AET channels
        self.response = resp.get_response(self.freq, gen2=False, tdi='AET', 
                                         equal_arm=True, cross_term=False)
        logger.info(f"Response function computed for AET channels")
        
        self.key = random.PRNGKey(1000)
        
        # Prior grid setup
        logger.info(f"Setting up parameter grid: {n_grid_points} points")
        logger.info(f"  Amplitude range: {amp_range}")
        logger.info(f"  Slope range: {slope_range}")
        self.amp = np.random.uniform(amp_range[0], amp_range[1], n_grid_points)
        self.slope = np.random.uniform(slope_range[0], slope_range[1], n_grid_points)
        
        # Setup alpha values
        if alpha is not None and alpha_0 is not None:
            # Fixed alpha values
            self.alpha = float(alpha) * jnp.ones_like(self.freq)
            self.alpha_0 = float(alpha_0) * jnp.ones_like(self.freq)
            self.noise_samples = None
            logger.info(f"Using FIXED alpha values:")
            logger.info(f"  alpha (H1) = {alpha}")
            logger.info(f"  alpha_0 (H0) = {alpha_0}")
            
        else:
            # Adaptive alpha based on noise parameter uncertainty
            logger.info("Computing ADAPTIVE alpha from noise parameter uncertainty...")
            logger.info("  This represents calibration/systematic uncertainty")
            
            A_samples = np.random.uniform(A_range[0], A_range[1], self.n_samp)
            P_samples = np.random.uniform(P_range[0], P_range[1], self.n_samp)
            
            # Compute alpha from noise uncertainty (for first TDI channel)
            alpha_computed = self.map_params_to_inv_gamma(
                tdi=0,
                key=self.key,
                A_samples=A_samples,
                P_samples=P_samples
            )
            
            # Store noise samples for computing parameter-dependent alpha if needed
            self.noise_samples = {
                'A': jnp.array(A_samples),
                'P': jnp.array(P_samples)
            }
            
            # Use same alpha for both H0 and H1
            # (since at fixed signal params, uncertainty is dominated by noise)
            self.alpha = alpha_computed
            self.alpha_0 = alpha_computed
            logger.info("  Mode: NOISE UNCERTAINTY ONLY (recommended)")
            logger.info(f"  Using same alpha for H0 and H1")
          
            # Plot alpha as function of frequency
            self._plot_alpha_vs_frequency(alpha_computed)
        
        logger.info("=" * 70)
        logger.info("INITIALIZATION COMPLETE")
        logger.info("=" * 70)
    
    def _plot_alpha_vs_frequency(self, alpha_computed):
        """Plot alpha as a function of frequency."""
        fig = plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.semilogx(self.freq, alpha_computed, 'b-', linewidth=2, label='Computed α')
        plt.axhline(y=2., color='r', linestyle='--', alpha=0.5, label='Minimum (α=2.)')
        #plt.axhline(y=7.0, color='g', linestyle='--', alpha=0.5, label='Typical (α=7)')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('$\\nu$')
        plt.title('Inverse Gamma Shape Parameter vs Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(alpha_computed, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=jnp.mean(alpha_computed), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean = {jnp.mean(alpha_computed):.2f}')
        plt.axvline(x=jnp.median(alpha_computed), color='g', linestyle='--', 
                   linewidth=2, label=f'Median = {jnp.median(alpha_computed):.2f}')
        plt.xlabel('$\\nu$')
        plt.ylabel('Count')
        plt.title('Distribution of Alpha Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('alpha_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("  Saved alpha analysis plot to 'alpha_analysis.png'")
        plt.close()
    
    
    def map_params_to_inv_gamma(self, tdi, key, A_samples, P_samples):
        """
        Map noise parameter uncertainty to inverse-gamma shape parameter alpha.
        
        This computes the uncertainty in the PSD due to uncertainty in the
        noise parameters (A, P), which represents calibration/systematic errors.
        
        Args:
            tdi: TDI channel index (0=A, 1=E, 2=T)
            key: JAX random key (unused but kept for API compatibility)
            A_samples: Noise parameter A samples
            P_samples: Noise parameter P samples
        
        Returns:
            alpha_arr: Shape parameter for each frequency bin
        """
        
        # Compute PSD samples from noise parameter samples
        def compute_psd_sample(a, p):
            par_noise = {'A': a, 'P': p}
            return noise.noise(self.freq, par_noise, tdi, gen2=False)
        
        # Vectorize over samples
        compute_vmap = vmap(compute_psd_sample, in_axes=(0, 0))
        PSD_samples = compute_vmap(A_samples, P_samples)  # Shape: (n_samples, n_freq)
        
        # Compute mean and variance per frequency
        mu = jnp.mean(PSD_samples, axis=0)
        sigma2 = jnp.var(PSD_samples, axis=0, ddof=1)
        
        # Map to inverse-gamma parameters
        # For inverse gamma: CV² = σ²/μ² = 1/(α-2)
        # Therefore: α = 2 + 1/CV² = 2 + μ²/σ²
        alpha = 2.0 + mu**2 / (sigma2 )  # Add small constant for stability
        
        # Clip to reasonable range
        alpha = jnp.clip(alpha, 2.1, 50.0)
        
        return alpha
    
    def run(self):
        """Run the Bayesian PLS computation."""
        logger.info("=" * 70)
        logger.info("STARTING BAYES FACTOR COMPUTATION")
        logger.info("=" * 70)
        
        time_start = time.time()
        
        # Use fixed alpha for both H0 and H1
        logger.info("Computing with FIXED alpha (same for all grid points)...")
        self.BF_tot = compute_bayes_factor(
            self.freq, self.response, self.psd_noise_base,
            self.amp, self.slope, self.n_samp, self.key, self.dof_used,
            self.alpha, self.alpha_0
        )
        
        time_end = time.time()
        
        logger.info("=" * 70)
        logger.info("COMPUTATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Elapsed time: {time_end - time_start:.2f} seconds")
        logger.info(f"Time per grid point: {(time_end - time_start)/len(self.amp)*1000:.2f} ms")
        
        return self.BF_tot
    
    def snr2(self, Sh, Sn):
        """Compute the squared Signal-to-Noise Ratio (SNR^2)."""
        snr_squared = 4 * jnp.sum((Sh / Sn)**2) * self.Tobs * self.df
        return snr_squared
    
    def snr(self, par):
        """Compute the Signal-to-Noise Ratio (SNR) for given parameters."""
        psd_gw = signal.Omega_pl(self.freq, par) * self.response
        psd_noise = self.psd_noise_base
        
        sn = 0
        for i in range(3):  # Sum over 3 TDI channels
            psd_gw_i = psd_gw[i, :]
            psd_noise_i = psd_noise[i, :]
            sn += self.snr2(psd_gw_i, psd_noise_i)
        
        return jnp.sqrt(sn)
    
    def plot_results(self, save_dir='.'):
        """
        Plot the Bayes factor results.
        
        Args:
            save_dir: Directory to save plots
        """
        import matplotlib.colors as mcolors
        
        logger.info("=" * 70)
        logger.info("GENERATING PLOTS")
        logger.info("=" * 70)
        
        # Plot 1: Bayes Factor
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter = ax.scatter(self.slope, self.amp, c=self.BF_tot.flatten(), 
                           vmin=-1, vmax=5, rasterized=True, marker='.', 
                           cmap='RdBu_r', s=10, alpha=0.6)
        
        cbar = fig.colorbar(scatter, ax=ax, label="Bayes Factor (log₁₀)")
        cbar.set_label("Bayes Factor (log₁₀)", fontsize=12)
        
        # Add contours at BF = 0, 1, 2
        from scipy.interpolate import griddata
        grid_slope = np.linspace(self.slope.min(), self.slope.max(), 200)
        grid_amp = np.linspace(self.amp.min(), self.amp.max(), 200)
        grid_slope_2d, grid_amp_2d = np.meshgrid(grid_slope, grid_amp)
        
        try:
            grid_bf = griddata((self.slope, self.amp), self.BF_tot, 
                             (grid_slope_2d, grid_amp_2d), method='cubic')
            
            contours = ax.contour(grid_slope_2d, grid_amp_2d, grid_bf, 
                                 levels=[0, 1, 2], colors='black', 
                                 linewidths=[1, 1.5, 2], linestyles=['--', '-', '-'])
            ax.clabel(contours, inline=True, fontsize=10, 
                     fmt={0: 'BF=1', 1: 'BF=10', 2: 'BF=100'})
        except:
            logger.warning("  Could not add contours (insufficient data coverage)")
        
        ax.set_xlabel('Slope)', fontsize=12)
        ax.set_ylabel('Amplitude log10', fontsize=12)
        ax.set_title('Bayes Factor: H1 (Signal) vs H0 (Noise Only)', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/bayes_factor_contour_plot.png', dpi=300, bbox_inches='tight')
        logger.info(f"  Saved Bayes factor plot to '{save_dir}/bayes_factor_contour_plot.png'")
        plt.close()
        
        # Plot 2: Histogram of Bayes factors
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(self.BF_tot, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='BF = 1 (Equal evidence)')
        ax.axvline(x=1, color='orange', linestyle='--', linewidth=2, label='BF = 10 (Strong)')
        ax.axvline(x=2, color='green', linestyle='--', linewidth=2, label='BF = 100 (Very strong)')
        
        ax.set_xlabel('log10(Bayes Factor)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Bayes Factors Across Parameter Grid', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/bayes_factor_histogram.png', dpi=300, bbox_inches='tight')
        logger.info(f"  Saved histogram to '{save_dir}/bayes_factor_histogram.png'")
        plt.close()
        
        # Plot 3: SNR comparison (optional, commented out for speed)
        # Uncomment if you want SNR analysis
        """
        logger.info("  Computing SNR values (this may take a while)...")
        snr_values = np.zeros_like(self.amp)
        for i in range(len(snr_values)):
            par = {'Amp': self.amp[i], 'slope': self.slope[i]}
            snr_values[i] = self.snr(par)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # SNR plot
        scatter = axes[0].scatter(self.slope, self.amp, c=np.log10(snr_values), 
                                 vmin=np.log10(10), vmax=np.log10(3000), 
                                 rasterized=True, marker='.', cmap='viridis', s=10)
        cbar = fig.colorbar(scatter, ax=axes[0], label="SNR")
        cbar.set_ticks(np.log10([10, 25, 70, 170, 450, 1000, 3000]))
        cbar.set_ticklabels([10, 25, 70, 170, 450, 1000, 3000])
        axes[0].set_xlabel('Slope (γ)')
        axes[0].set_ylabel('Amplitude (log₁₀ Ωₕ²)')
        axes[0].set_title('Signal-to-Noise Ratio')
        axes[0].grid(True, alpha=0.3)
        
        # BF vs SNR scatter
        axes[1].scatter(snr_values, 10**self.BF_tot, alpha=0.3, s=10)
        axes[1].set_xlabel('SNR')
        axes[1].set_ylabel('Bayes Factor')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title('Bayes Factor vs SNR')
        axes[1].grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/snr_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"  Saved SNR analysis to '{save_dir}/snr_analysis.png'")
        plt.close()
        """
        
        logger.info("=" * 70)
        logger.info("PLOTTING COMPLETE")
        logger.info("=" * 70)


#    def save_results(self, filename='bayes_factor_results.npz'):
#        """
#        Save results to file.
#        
#        Args:
#            filename: Output filename
#        """
#        np.savez(filename,
#                 amp=self.amp,
#                 slope=self.slope,
#                 BF=np.array(self.BF_tot),
#                 freq=np.array(self.freq),
#                 alpha=np.array(self.alpha) if self.alpha is not None else None,
#                 alpha_0=np.array(self.alpha_0),
#                 n_samp=self.n_samp,
#                 dof_used=self.dof_used,
#                 Tobs=self.Tobs)
#        
#        logger.info(f"Results saved to '{filename}'")
#    
#    def load_results(self, filename='bayes_factor_results.npz'):
#        """
#        Load results from file.
#        
#        Args:
#            filename: Input filename
#        """
#        data = np.load(filename)
#        self.amp = data['amp']
#        self.slope = data['slope']
#        self.BF_tot = jnp.array(data['BF'])
#        
#        logger.info(f"Results loaded from '{filename}'")
#        logger.info(f"  Grid size: {len(self.amp)}")
#        logger.info(f"  BF range: [{jnp.min(self.BF_tot):.2f}, {jnp.max(self.BF_tot):.2f}]")


