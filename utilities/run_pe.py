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

# Set up logging
logger = logging.getLogger('BAHAMAS_Inference')
logger.setLevel(logging.DEBUG)

# Add a console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Change to DEBUG to see debug messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run BAHAMAS inference for LISA data')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--sources', type=str, required=True, help='YAML sources file')
    parser.add_argument('--use_jax', type=lambda x: bool(strtobool(x)), default=True,
                    help='Use JAX as the backend (True/False)')
    args = parser.parse_args()

    # Initialize the backend based on the --use_jax argument
    initialize_backend(use_jax=args.use_jax)
    from bahamas import bahamas_inference

    if args.use_jax:
        import jax
        logger.info("JAX devices available: %s", jax.devices())
    else:
        logger.info("Using NumPy backend (JAX disabled).")

    logger.info(f"Running inference with config: {args.config} and sources: {args.sources}")
    logger.info("Running inference...")

    # Run inference
    inference = bahamas_inference.BayesianInference(args.config, args.sources)
    inference.run()

    logger.info("Inference completed successfully.")
    logger.info("Results saved to: %s", inference.config['inference']["file_post"])

if __name__ == "__main__":
    main()




