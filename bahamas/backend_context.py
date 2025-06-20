"""
This module handles the initialization and management of the computational backend
for the BAHAMAS framework. It supports both JAX and NumPy backends.
"""
import os
from bahamas.logger_config import logger
# Global variables to store backend components
np = None
jit = None
lax = None
_backend_initialized = False

def initialize_backend(use_jax=True, force_reinitialize=False):
    """
    Initialize the backend and store the components globally.

    Args:
        use_jax (bool): Whether to use JAX as the backend. If False, NumPy is used.
        force_reinitialize (bool): If True, reinitialize the backend even if already initialized.
    """
    global np, jit, lax, _backend_initialized
    if _backend_initialized and not force_reinitialize:
        print(f"Backend already initialized. np is from: {np.__name__}")
        return  # Prevent reinitialization unless forced

    from bahamas.backend import get_backend
    np, jit, lax = get_backend(use_jax=use_jax)
    _backend_initialized = True
    # Set the backend globally
    
    logger.info(f"Using {'JAX' if use_jax else 'NumPy'} backend.")
    logger.info(f"np is from: {np.__name__}")


def get_backend_components():
    """
    Retrieve the current backend components.

    Returns:
        tuple: A tuple containing the backend components (np, jit, lax).
    """
    global np, jit, lax
    return np, jit, lax