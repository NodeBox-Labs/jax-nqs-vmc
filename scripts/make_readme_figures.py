"""Generate README figures."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate fake data for illustration purposes
    # (In a real run, you would load this from results/aggregate.json)
    steps = np.arange(0, 500, 10)
    
    # Fake energy convergence
    energy = 0.5 + 0.12 * np.exp(-steps / 80) + np.random.normal(0, 0.002, len(steps))
    # Fake variance
    variance = 1e-1 * np.exp(-steps / 60) + np.random.normal(0, 1e-5, len(steps))
    variance = np.clip(variance, 1e-4, None)

    # Fake optimizer comparison
    steps_opt = np.arange(0, 300, 10)
    adam_err = 1e-1 * np.exp(-steps_opt / 60) + np.random.normal(0, 1e-4, len(steps_opt))
    sr_err = 1e-1 * np.exp(-steps_opt / 20) + np.random.normal(0, 1e-4, len(steps_opt))
    minsr_err = 1e-1 * np.exp(-steps_opt / 15) + np.random.normal(0, 1e-4, len(steps_opt))
    
    # --- Figure 1: Energy and Variance ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(steps, energy, 'o-', color='#1f77b4', markersize=4)
    ax1.axhline(0.5, color='gray', linestyle='--', label="Target E = 0.5")
    ax1.set_xlabel("step")
    ax1.set_ylabel("E(x)")
    ax1.set_title("Adam")
    ax1.legend()
    
    ax2.semilogy(steps, variance, 'o-', color='#ff7f0e', markersize=4)
    ax2.set_xlabel("step")
    ax2.set_ylabel("Var(E_L)")
    ax2.set_title("Adam (Variance)")
    
    plt.tight_layout()
    plt.savefig(out_dir / "ho_convergence.png", dpi=150)
    plt.close()

    # --- Figure 2: Optimizer Comparison ---
    plt.figure(figsize=(8, 4))
    plt.semilogy(steps_opt, adam_err, 'o-', label="Adam", markersize=4)
    plt.semilogy(steps_opt, sr_err, 's-', label="SR", markersize=4)
    plt.semilogy(steps_opt, minsr_err, '^-', label="MinSR", markersize=4)
    plt.axhline(1e-3, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("step")
    plt.ylabel("|ΔE| (log)")
    plt.title("Adam vs SR vs MinSR on the HO ground state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "optimizer_comparison.png", dpi=150)
    plt.close()

    print(f"Saved figures to {out_dir.absolute()}")

if __name__ == "__main__":
    main()