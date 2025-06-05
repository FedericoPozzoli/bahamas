---
title: 'Bahamas: BAyesian inference with HAmiltonian Montecarlo for Astrophysical Stochastic background.'
tags:
  - Python
  - astrophysics
  - gravitational waves
  - stochastic signal
authors:
  - name: Federico Pozzoli
    orcid: 0009-0009-6265-584X
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Riccardo Buscicchio
    affiliation: 2
  - name: Antoine Klein
    affiliation: 3
  - name: Daniele Chirico
    affiliation: 2
affiliations:
 - name: Università degli studi dell'Insubria, Italy
   index: 1
   ror: 00hx57361
 - name: Università degli studi Milano-Bicocca, Italy
   index: 2
 - name: Universitity of Birmingham
   index: 3
date: 29 April 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The Laser Interferometer Space Antenna (LISA) [@LISA:2018] is an upcoming space-based mission designed to detect gravitational waves (GWs) of both astrophysical and cosmological origin in the milli-hertz band. LISA is expected to observe thousands of white dwarf (WD) binaries within the Milky Way simultaneously, while an unresolved population of such binaries will overlap incoherently, forming the so-called galactic foreground.
One of the central challenges of the so-called global fit [@Katz:2025] is to jointly model both the resolvable and unresolvable WD populations. In particular, reconstructing the galactic foreground is extremely difficult due to both computational and modeling complexities.
In this article, we introduce `bahamas`, a tool designed to address some of these challenges from a global fit perspective.

# Statement of need

The main idea behind the global fit algorithm is to use a Blocked Gibbs sampling technique to jointly analyze different GW sources, including stochastic backgrounds, instrumental noise, and the galactic foreground. LISA is expected to sample data at $\sim 10\mathrm{s}$, with a nominal mission duration of four years. This results in an extremely large dataset for a full-band analysis of the stochastic components. Consequently, computational cost becomes a significant concern for the stochastic sector. Traditional sampling techniques, such as nested sampling or standard Markov Chain Monte Carlo (MCMC), might become prohibitively slow for this task. To address this issue, bahamas employs the No-U-Turn Sampler (NUTS) [@Hoffman:2011], an adaptive variant of Hamiltonian MCMC, which significantly enhances sampling efficiency. It uses the implementation provided by NumPyro [@Phan:2019], which is totally based in JAX, enabling possibly convenient migration to GPU/TPU architectures GPU/TPU architecture [@Bradbury:2018]. 

Another issue in the reconstruction of the Galactic foreground is its non-stationary nature. The sky distribution of unresolved Galactic WDs is highly anisotropic. Coupling this anisotropy with LISA’s annual motion and its time-dependent antenna pattern results in a cyclostationary process—a stochastic process with periodic time-dependent properties. In the time domain, the Galactic noise appears modulated. For this reason, even when using chunked data to mitigate non-stationarity, the spectral amplitude may vary inconsistently between chunks. 
There are currently no global fit pipelines accounting for this feature. In `bahamas`, we adopt a time-frequency approach that incorporates the modulation proposed in [@Buscicchio:2024] to model the evolution of spectral amplitude in chunks. The key advantage is that we employ a parameterizable modulation that is both analytical and easily evaluable. Specifically, it decomposes the squared time-evolving, sky-location-dependent antenna pattern (since we aim to track the spectral amplitude evolution) as a sum of sinusoids, and computes the overall envelope using the characteristic function, assuming a Gaussian distribution for the WDs in the sky. 

The algorithm also allows for the inclusion of stationary, isotropic, and Gaussian stochastic process (e.g., a signal characterized by a power-law power spectral density), enabling the evaluation of the impact of multiple overlapping sources.


# Software Description 
The package includes two main command-line interfaces:

  - `run_data.py`: Data simulation and preprocessing. 

  - `run_pe.py`: Parameter estimation and minimal diagnostics

Both scripts require two input files:

  - `--config config.yaml`: Specifies the simulation and inference settings, including data injection parameters, sampler configuration, runtime options, and output paths.

  - `--sources sources.yaml`: Defines the sources to be injected and/or recovered. This includes the true physical parameters of the sources as well as the prior ranges used for inference.

The data consist of two datastreams—the A and E channels—which are specific combinations of Time-Delay Interferometry (TDI) variables [@Tinto:2021]. In `bahamas`, the data are generated in the frequency domain, chunk by chunk. This represents a simplification, as it neglects potential biases arising in the time domain, such as windowing effects and spectral leakage. We also note that the duration of each chunk—and consequently the frequency resolution of each segment—can be set arbitrarily in config.yaml. However, we recommend not using time lengths shorter than $10^4 \mathrm{s}$, which corresponds to a frequency resolution of approximately $\Delta f \sim 0.1 \mathrm{mHz}$, below which the characterization of LISA's instrumental noise is not guaranteed.  
The algorithm provides flexibility to perform analyses with either full-resolution data or coarse-grained data over different chunks. In the former case, the likelihood describing the data follows a Whittle distribution [@Moran:1951] in each segment, while in the latter, it collapses to a Gamma distribution [@Appourchaux:2003] with degrees of freedom equal to the number of bins used in the averaging process. 

# Performance

![Caption for the figure](joss_corner.png)

# Acknowledgements

bla bla

# References