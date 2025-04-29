
"""
This script runs the BAHAMAS inference for LISA data using JAX.
It is designed to be run from the command line with specified arguments.
It handles the loading of configuration files, sources, and the inference process.
It also includes functionality for logging and debugging.
The script uses the BAHAMAS framework for analyzing gravitational wave signals.
It supports both Hamiltonian Monte Carlo (HMC) methods.
It also includes functions for loading configuration files, running inference, and saving results.
"""

import argparse
from bahamas import bahamas_inference
import logging
import jax

# Set up logging
logger = logging.getLogger('BAHAMAS_Inference')
logger.setLevel(logging.DEBUG)

# Add a console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Change to DEBUG to see debug messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if __name__ == "__main__":
    # Check if JAX is available and print devices
    logger.info("JAX devices available: %s", jax.devices())

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run BAHAMAS inference for LISA data')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--sources', type=str, required=True, help='YAML sources file')
    args = parser.parse_args()

    logger.info(f"Running inference with config: {args.config} and sources: {args.sources}")
    logger.info("JAX devices available: %s", jax.devices())
    logger.info("Running inference...")
    
    # Load configuration and sources
    config = bahamas_inference.load_yaml(args.config)
    sources = bahamas_inference.load_yaml(args.sources)['sources']

    # Initialize and run inference
    inference = bahamas_inference.BayesianInference(config, sources)
    inference.run()

    logger.info("Inference completed successfully.")

    logger.info("Results saved to: %s", inference.config['inference']["file_post"])


