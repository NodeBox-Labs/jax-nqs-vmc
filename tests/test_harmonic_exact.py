import numpy as np

from nqs_vmc.reference.exact_diag import double_well_exact, harmonic_exact


def test_harmonic_eigenvalues():
    vals, _, _ = harmonic_exact(n_max=6, L=10.0, n_grid=2048)
    expected = np.arange(6) + 0.5
    np.testing.assert_allclose(vals, expected, atol=1e-6)


def test_double_well_symmetry():
    vals, x, vecs = double_well_exact(a=-0.5, b=0.25, n_max=4)
    assert vals[0] < vals[1] < vals[2] < vals[3]
    # parity: ground state even, first excited odd
    psi0 = vecs[:, 0]
    psi1 = vecs[:, 1]
    assert np.allclose(psi0, psi0[::-1], atol=1e-3)
    assert np.allclose(psi1, -psi1[::-1], atol=1e-3)