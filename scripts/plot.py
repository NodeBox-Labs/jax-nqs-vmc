"""Generate plots from a completed run."""
import argparse
from pathlib import Path
from nqs_vmc.plotting import plot_convergence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    args = ap.parse_args()
    
    out_png = args.run.parent / f"{args.run.name}_convergence.png"
    plot_convergence(args.run / "metrics.jsonl", out_png)
    print(f"Saved to {out_png}")

if __name__ == "__main__":
    main()