
"""
This script is used to run the BAHAMAS data processing pipeline.
It handles the loading of configuration files, sources, and the processing of signals.
It also includes functionality for simulating data, saving results, and plotting power spectral density (PSD).
It is designed to be run from the command line with specified arguments.
"""

import argparse
import logging
from bahamas.backend_context import initialize_backend
initialize_backend(use_jax=True)  # Initialize backend with JAX
from bahamas import bahamas_data

from bahamas.logger_config import logger

def main():
    parser = argparse.ArgumentParser(description='Run BAHAMAS data processing pipeline')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--sources', type=str, required=True, help='YAML sources file')
    args = parser.parse_args()

    # Parse arguments and load configuration
    args = bahamas_data.parse_arguments()
    config = bahamas_data.load_yaml(args.config)
    sources = bahamas_data.load_yaml(args.sources)['sources']
    
    # Initialize and process signals
    processor = bahamas_data.SignalProcessor(config, sources)
    processor.handle_series()
    processor.simulate_data()
    processor.save_data()
    processor.plot_psd()
    processor.compute_SNR()
    logger.info("Processing completed.")

if __name__ == "__main__":
    main()