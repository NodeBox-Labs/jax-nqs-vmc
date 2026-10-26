from __future__ import annotations

import numpy as np


def gpe_imaginary_time(
    x: np.ndarray,
    omega: float = 1.0,
    g: float = 1.0,
    psi0: np.ndarray | None = None,
    dt: float = 0.005,
    n_steps: int = 20000,
    tol: float = 1e-12,
):
    if psi0 is None:
        # start from the noninteracting ground state
        psi0 = (omega / np.pi) ** 0.25 * np.exp(-0.5 * omega * x**2)
    psi = psi0.astype(np.complex128).copy()
    psi /= np.sqrt(np.trapz(np.abs(psi) ** 2, x))

    k = 2 * np.pi * np.fft.fftfreq(x.size, d=x[1] - x[0])
    V = 0.5 * omega**2 * x**2
    K = 0.5 * k**2  # kinetic dispersion

    def half_nonlin(psi, sign):
        V_eff = V + g * np.abs(psi) ** 2
        return psi * np.exp(sign * 0.5 * (-0.5j * 0) ).real if False else psi  # noqa

    mu_prev = np.inf
    mus = []
    for step in range(n_steps):
        # half-step potential (nonlinear)
        V_eff = V + g * np.abs(psi) ** 2
        psi = psi * np.exp(-0.5 * dt * V_eff)
        # full-step kinetic
        psi = np.fft.ifft(np.exp(-dt * K) * np.fft.fft(psi))
        # half-step potential
        V_eff = V + g * np.abs(psi) ** 2
        psi = psi * np.exp(-0.5 * dt * V_eff)

        norm = np.trapz(np.abs(psi) ** 2, x)
        psi /= np.sqrt(norm)

        if step % 50 == 0:
            # chemical potential: μ = E[ψ]/N - (g/2)∫|ψ|⁴ for GPE; we just log
            mu = _chemical_potential(psi, x, V, g)
            mus.append(mu)
            if abs(mu - mu_prev) < tol:
                break
            mu_prev = mu

    return psi, np.array(mus)


def _chemical_potential(psi, x, V, g):
    dx = x[1] - x[0]
    dpsi = np.gradient(psi, dx)
    kin = 0.5 * np.trapz(np.abs(dpsi) ** 2, x)
    pot = np.trapz(V * np.abs(psi) ** 2, x)
    inter = 0.5 * g * np.trapz(np.abs(psi) ** 4, x)
    n = np.trapz(np.abs(psi) ** 2, x)
    # E = kin+pot+inter; μ = E + (g/2) ∫|ψ|⁴ ... but for uniform we use
    # μ ≈ E/N for small N; here N=1 so μ = E + (g/2)∫|ψ|⁴ - (g)∫|ψ|⁴ = kin+pot-(g/2)∫|ψ|⁴
    return kin + pot - 0.5 * g * np.trapz(np.abs(psi) ** 4, x) / n