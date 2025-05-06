import numpy as np
from bahamas.psd_response.orbits import SpacecraftOrbit

def test_sc_positions():
    # Initialize the SpacecraftOrbit object
    mySC = SpacecraftOrbit()

    # Test at t = 0
    t = 0
    positions = mySC.SC_positions(t)

    # Expected positions at t = 0 (manually computed or simplified)
    expected_positions = np.array([[496.59152337, 0., -4.13007351],
          [500.18984301, -4.14003675, 2.10987136],
          [500.18984301, 4.14003675, 2.10987136]])


    # Assert the positions are close to the expected values
    assert positions.shape == (3, 3)
    assert np.allclose(positions, expected_positions, atol=1e-5), f"Expected {expected_positions}, got {positions}"

def test_link_versor():
    # Initialize the SpacecraftOrbit object
    mySC = SpacecraftOrbit()

    # Test at t = 0 for link 1
    t = 0
    link = 1
    versor, distance = mySC.link_versor(t, link)

    # Expected versor and distance for link 1 at t = 0 (manually computed or simplified)
    expected_distance = 8.280073507643198
    expected_versor = np.array([0., -1., 0.])  # Simplified for this specific case

    # Assert the versor and distance are close to the expected values
    assert np.isclose(distance, expected_distance, atol=1e-5), f"Expected distance {expected_distance}, got {distance}"
    assert np.allclose(versor, expected_versor, atol=1e-5), f"Expected versor {expected_versor}, got {versor}"