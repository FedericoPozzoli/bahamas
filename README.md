# BAHAMAS

[![JOSS draft](https://github.com/fede121/bahamas/actions/workflows/paper.yml/badge.svg?branch=main)](https://github.com/fede121/bahamas/actions/workflows/paper.yml) [![Tests](https://github.com/fede121/bahamas/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/fede121/bahamas/actions/workflows/testing.yml)
[![Documentation Status](https://github.com/fede121/bahamas/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/fede121/bahamas/actions/workflows/docs.yml)


**Bahamas: BAyesian inference with HAmiltonian Montecarlo for Astrophysical Stochastic background.**

BAHAMAS is a Python package for the inference of stochastic gravitational wave background signals (SGWB) using Hamiltonian Markov Chain Monte Carlo (HMCMC) as implemented in [NumPyro](https://num.pyro.ai/en/stable/getting_started.html#what-is-numpyro), which relies on JAX for automatic differentiation and JIT compilation to GPU/CPU.  
BAHAMAS is under active development, so be aware of potential brittleness, bugs, and changes to the API as the design evolves.

## Features

The package includes two main command-line interfaces:

  - `run_data.py`: Data simulation and preprocessing. 

  - `run_pe.py`: Parameter estimation and minimal diagnostics

Both scripts require two input files:

  - `--config config.yaml`: Specifies the simulation and inference settings, including data injection parameters, sampler configuration, runtime options, and output paths.

  - `--sources sources.yaml`: Defines the sources to be injected and/or recovered. This includes the true physical parameters of the sources as well as the prior ranges used for inference.

The data consist of two datastreams—the A and E channels—which are specific combinations of Time-Delay Interferometry (TDI) variables. In `bahamas`, the data are generated in the frequency domain, chunk by chunk. This represents a simplification, as it neglects potential biases arising in the time domain, such as windowing effects and spectral leakage. We also note that the duration of each chunk—and consequently the frequency resolution of each segment—can be set arbitrarily in config.yaml. However, we recommend not using time lengths shorter than $10^4 \mathrm{s}$, which corresponds to a frequency resolution of approximately $\Delta f \sim 0.1 \mathrm{mHz}$, below which the characterization of LISA's instrumental noise is not guaranteed.  
The algorithm provides flexibility to perform analyses with either full-resolution data or coarse-grained data over different chunks. In the former case, the likelihood describing the data follows a Whittle distribution in each segment, while in the latter, it collapses to a Gamma distribution with degrees of freedom equal to the number of bins used in the averaging process.

## Installation

You can install BAHAMAS via pip:

```bash
pip install bahamas
```

Or install from source:

```bash
git clone https://github.com/fede121/bahamas.git
cd bahamas
pip install .
```

## License

This project is open-source and available under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

For more information, examples, or to contribute, please visit the [GitHub repository](https://github.com/fede121/bahamas.git).
