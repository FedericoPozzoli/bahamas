"""
This script runs the BAHAMAS inference for LISA data using JAX or NumPy.
It is designed to be run from the command line with specified arguments.
It handles the loading of configuration files, sources, and the inference process.
It also includes functionality for logging and debugging.
The script uses the BAHAMAS framework for analyzing gravitational wave signals.
It supports both Hamiltonian Monte Carlo (HMC) methods.
It also includes functions for loading configuration files, running inference, and saving results.
"""

# Configure matplotlib to use a non-GUI backend
import matplotlib
matplotlib.use('Agg')

# Import other modules after initializing the backend
import argparse
from bahamas.backend_context import initialize_backend
import logging
from distutils.util import strtobool

from bahamas.logger_config import logger

def main():
    # Set up argument parser
    
    # import parser
    parser = argparse.ArgumentParser(description="Bayesian PLS computation with JAX")
    parser.add_argument('--n_samp', type=int, default=1000, help='Number of samples for Wishart distribution')
    parser.add_argument('--alpha', type=float, default=None, help='Degrees of freedom for Wishart distribution')
    parser.add_argument('--alpha_0', type=float, default=None, help='Degrees of freedom for Wishart distribution')
    parser.add_argument('--dof', type=int, default=1000, help='Degrees of freedom')
    parser.add_argument('--dof_used', type=int, default=None, help='Used degrees of freedom')
    parser.add_argument('--Tobs', type=float, default=1, help='Observation time in year')
    parser.add_argument('--grid_points', type=int, default=10000, help='Number of grid points for computation')
    parser.add_argument('--use_jax', type=lambda x: bool(strtobool(x)), default=True, help='Whether to use JAX backend')
    

    args = parser.parse_args()

    # Initialize the backend based on the --use_jax argument
    initialize_backend(use_jax=args.use_jax)
    from bahamas import bahamas_bpls

    if args.use_jax:
        import jax
        logger.info("JAX devices available: %s", jax.devices())
    else:
        logger.info("Using NumPy backend (JAX disabled).")

   
    logger.info(f"Producing Bayesian PLS computation with the following parameters:")
    logger.info(f"Number of samples: {args.n_samp}")
    logger.info(f"Degrees of freedom (alpha): {args.alpha}")
    logger.info(f"Degrees of freedom (dof): {args.dof}")
    logger.info(f"Observation time (Tobs): {args.Tobs} year(s)")

    # Run inference
    
    bpls = bahamas_bpls.BahamasBPLS(
        n_samp=args.n_samp,
        alpha=args.alpha,
        alpha_0=args.alpha_0,
        #include_signal_uncertainty=False,
        #signal_model_uncertainty=0.15,  # 15% uncertainty in signal model
        dof=args.dof,
        Tobs=args.Tobs,
        n_grid_points=args.grid_points
    )
    BF_advanced = bpls.run()
    bpls.plot_results(save_dir='.')

    logger.info("Bayesian PLS computation completed.")
    #logger.info("Results saved to: %s", inference.config['inference']["file_post"])

if __name__ == "__main__":
    main()

# example of command line to run the script
# python utilities/run_bpls.py --n_samp 1000 --alpha 50 --dof 1000 --Tobs 1 --grid_points 10000 --use_jax True

