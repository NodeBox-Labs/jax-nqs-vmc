import numpy as np
from nqs_vmc.reference.exact_diag import exact_1d


def test_harmonic_eigenvalues():
    vals, _, _ = exact_1d(lambda x: 0.5 * x**2, L=10.0, n_grid=2048, k=6)
    expected = np.arange(6) + 0.5
    # Grid discretization gives ~1e-4 error. atol=1e-3 is standard.
    np.testing.assert_allclose(vals, expected, atol=1e-3)


def test_double_well_parity():
    vals, x, vecs = exact_1d(lambda x: -0.5 * x**2 + 0.25 * x**4, L=8.0, n_grid=2048, k=4)
    psi0, psi1 = vecs[:, 0], vecs[:, 1]
    assert np.allclose(psi0, psi0[::-1], atol=1e-3)
    assert np.allclose(psi1, -psi1[::-1], atol=1e-3)