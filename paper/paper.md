---
title: 'Bahamas: '
tags:
  - Python
  - astrophysics
  - gravitational waves
  - stochastic signal
authors:
  - name: Federico Pozzoli
    orcid: 0009-0009-6265-584X
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Riccardo Buscicchio
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Antoine Klein
    affiliation: 3
    equal-contrib: true
  - name: Daniele Chirico
    affiliation: 2
    equal-contrib: true
affiliations:
 - name: Università degli studi dell'Insubria Italy
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

The Laser Interferometer Space Antenna (LISA) [@LISA:2018] is an upcoming space-based mission designed to detect gravitational waves (GWs) of both astrophysical and cosmological origin in the milli-hertz band. LISA is expected to observe thousands of white dwarf binaries within the Milky Way simultaneously, while an unresolved population of such binaries will overlap incoherently, forming the so-called galactic foreground.
One of the central challenges of the so-called global fit [@Katz:2025] is to jointly model both the resolvable and unresolvable white dwarf populations. In particular, reconstructing the galactic foreground is extremely difficult due to both computational and modeling complexities.
In this article, we introduce `bahamas`, a tool designed to address some of these challenges from a global fit perspective.

# Statement of need

The main idea behind the global fit algorithm is to use a Blocked Gibbs sampling technique to jointly analyze different gravitational wave (GW) sources, including stochastic backgrounds, instrumental noise, and the galactic foreground. LISA is expected to sample data at $\sim 10,\mathrm{s}$, with a nominal mission duration of four years. This results in an extremely large dataset for a full-band analysis of the stochastic components. Consequently, computational cost becomes a significant concern for the stochastic sector. Traditional sampling techniques, such as nested sampling or standard Markov Chain Monte Carlo (MCMC), might become prohibitively slow for this task. To overcome this issue, `bahamas` leverages the No-U-Turn Sampler (NUTS) [@Hoffman:2011], an adaptive variant of Hamiltonian MCMC, which significantly improves sampling efficiency.

Another issue in the galactic foregorund reconstruction is its non-stationary nature. The distribution in the sky of the unresolved galactic WDs is higly unisotropic. Coupling the anisotropy with the annual motion of LISA and the time-dependent antenna pattern result in a cyclostationary process: a stochastic process with periodic properties in time. 

# Software Description 

# Illustrative Example

# Quality Control

# Performance

# Acknowledgements

bla bla

# References