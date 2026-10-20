"""Plot the Monte Carlo results from results/montecarlo_reps.csv (can be re-run without repeating the 10-minute simulation)."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from plotting import BLUE, GRAY, ORANGE, MUTED

ROOT = os.path.join(os.path.dirname(__file__), "..")


def plot():
    R = pd.read_csv(f"{ROOT}/results/montecarlo_reps.csv")
    out = json.load(open(f"{ROOT}/results/montecarlo.json"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axs[0]
    ax.hist(R.cv_rel * 100, bins=20, color=BLUE, alpha=0.9, label="Clean population", edgecolor="#fcfcfb")
    ax.hist(R.cvn_rel * 100, bins=20, color=ORANGE, alpha=0.8, label="Naive (bug ignored)", edgecolor="#fcfcfb")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.35)
    tr = R.cv_truth_w.mean() * 100
    ax.axvline(tr, color=MUTED, ls="--", lw=1.2)
    ax.text(tr + 0.2, top * 1.22, "true effect", color=MUTED, fontsize=9)
    ax.set_xlabel("Estimated conversion lift (relative, %), 100 replications"); ax.set_ylabel("Replications")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Ignoring the logging bug biases the lift upward", fontsize=10)
    ax = axs[1]
    ax.hist(R.m_est - R.m_truth_w, bins=20, color=GRAY, edgecolor="#fcfcfb")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Margin/visitor estimate − truth ($), 100 replications"); ax.set_ylabel("Replications")
    ax.set_title(f"No detectable bias; 95% CI coverage = {out['margin']['coverage_window']:.0%}", fontsize=10)
    from matplotlib.ticker import MaxNLocator
    for a_ in axs: a_.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout(); fig.savefig(f"{ROOT}/figures/montecarlo_coverage.png"); plt.close(fig)


if __name__ == "__main__":
    plot()
