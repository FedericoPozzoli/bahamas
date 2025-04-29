# BAHAMAS

[![JOSS draft](https://github.com/fede121/bahamas/actions/workflows/paper.yml/badge.svg?branch=main)](https://github.com/fede121/bahamas/actions/workflows/paper.yml) [![Tests](https://github.com/fede121/bahamas/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/fede121/bahamas/actions/workflows/testing.yml)

**BAyesian HAmiltonian Montecarlo Analysis for Stochastic Gravitational Wave Signals**

BAHAMAS is a Python package for the inference of stochastic gravitational wave background signals (SGWB) using Hamiltonian Markov Chain Monte Carlo (HMCMC) as implemented in [NumPyro](https://num.pyro.ai/en/stable/getting_started.html#what-is-numpyro), which relies on JAX for automatic differentiation and JIT compilation to GPU/CPU.  
BAHAMAS is under active development, so be aware of potential brittleness, bugs, and changes to the API as the design evolves.

## Features

- 🔊 **SGWB Signal Generation**
  - Generate synthetic SGWB data directly in the frequency domain
  - Simulate single full-spectrum signals or split the signal into multiple time chunks

- 📈 **Flexible Inference Engine**
  - Perform parameter estimation of multiple stochastic signals using different statistical approaches:
    - **Gamma Likelihood**: for coarse-grained binned data
    - **Whittle Likelihood**: for full-resolution frequency data
  - Joint inference of multiple overlapping SGWB components

- 🌌 **Galactic Foreground Support**
  - Analyze chunked data to reconstruct both the spectrum and the time-dependent **modulation** of Galactic foregrounds

## Installation

You can install BAHAMAS via pip:

```bash
pip install bahamas
```

Or install from source:

```bash
git clone https://github.com/yourusername/bahamas.git
cd bahamas
pip install .
```

## License

This project is open-source and available under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

For more information, examples, or to contribute, please visit the [GitHub repository](https://github.com/yourusername/bahamas).
