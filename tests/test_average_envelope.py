import jax.numpy as jnp
from bahamas.psd_response.average_envelope import average_envelopes_gaussian

def test_average_envelopes_gaussian_exact_value():
    # Define fixed input parameters
    EclipticLatitude = 0.5
    EclipticLongitude = 1.0
    Sigma1 = 0.1
    Sigma2 = 0.2
    Psi = 0.5
    t1 = 0.0
    t2 = 1.0
    LISA_Orbital_Freq = 1 / 31557600.0  # 1/year in seconds
    alpha0 = 0.0
    beta0 = 0.0
    tdi = 0

    # Compute the result using the function
    result = average_envelopes_gaussian(
        EclipticLatitude, EclipticLongitude, Sigma1, Sigma2, Psi, t1, t2, 
        LISA_Orbital_Freq, alpha0, beta0, tdi
    )

    # Expected value (computed manually or using a simplified version of the function)
    # This is a placeholder value; replace it with the actual computed value
    expected_value = 1.8409162940416053  # Replace with the exact value

    # Assert that the result matches the expected value within a tolerance
    assert jnp.isclose(result, expected_value, atol=1e-5), f"Expected {expected_value}, got {result}"