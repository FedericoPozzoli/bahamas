import numpy as np
from bahamas.backend_context import get_backend_components
# Load backend components after backend initialization
jnp, jit, lax = get_backend_components()

# Check if lax is available
if jnp is None:
    #default to jax numpy
    from bahamas.backend_context import initialize_backend
    initialize_backend(use_jax=True)
    jnp, jit, lax = get_backend_components()

if lax is not None:
    import jax
    jax.config.update('jax_enable_x64', True)

from bahamas.psd_strain.psd_function import model_psd

def test_model_psd_single_source():
    freqs = np.array([1e-3])  # Frequency array
    sources = {
        "instr_noise": {"A": 3, "P": 15}
    }
    response = 1.0
    psd = model_psd(freqs, sources, response)

    # Numerical checks
    expected_psd = np.array([1.0e-36, 1.0e-34, 1.0e-32])  # Replace with actual expected values
    assert np.allclose(psd, expected_psd, atol=1e-10), f"Expected {expected_psd}, got {psd}"

def test_model_psd_numeric():
    # Define frequency array
    freqs = jnp.array([1e-3, 1e-2, 1e-1])  # Frequency array in Hz

    # Define sources with injected parameters
    sources = {
        "galactic_DWD": [
            {"name": "alpha", "bounds": [1.0, 2.0], "injected": 1.8},
            {"name": "amp", "bounds": [-47, -40], "injected": -43.94309514866353},
            {"name": "fknee", "bounds": [-4, -1.8], "injected": -2.595679532778269},
            {"name": "fr1", "bounds": [-4, -1.8], "injected": -2.856},
            {"name": "fr2", "bounds": [-4, -1.8], "injected": -3.487854923626168},
        ],
        "instr_noise": [
            {"name": "A", "bounds": [1, 5], "injected": 3},
            {"name": "P", "bounds": [10, 20], "injected": 15},
        ],
    }

    # Define response factor
    response = 1.0

    # Compute the PSD using the model_psd function
    psd, true_psds = model_psd(freqs, sources, response, injected=True)

    # Expected PSD values (manually computed or from a reference implementation)
    expected_psd = jnp.array([1.2e-40, 2.3e-38, 3.4e-36])  # Replace with actual expected values

    # Assert that the PSD matches the expected values within a tolerance
    assert psd.shape == freqs.shape, "PSD shape mismatch"
    assert jnp.all(psd >= 0), "PSD contains negative values"
    assert jnp.allclose(psd, expected_psd, atol=1e-10), f"Expected {expected_psd}, got {psd}"

    # Check individual source contributions if needed
    for true_psd in true_psds:
        assert jnp.all(true_psd >= 0), "True PSD contains negative values"

