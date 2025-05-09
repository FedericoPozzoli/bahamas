"""
This module provides a function to retrieve the appropriate backend components
for the BAHAMAS framework, either JAX or NumPy.
"""

def get_backend(use_jax=True):
    """
    Return the appropriate backend components (np, jit, lax).

    Args:
        use_jax (bool): Whether to use JAX as the backend. If False, NumPy is used.

    Returns:
        tuple: A tuple containing the backend components (np, jit, lax).
    """
    if use_jax:
        import jax.numpy as np
        from jax import jit, lax
    else:
        import numpy as np
        jit = lambda x: x  # No-op for NumPy
        lax = None  # Not applicable for NumPy
    return np, jit, lax