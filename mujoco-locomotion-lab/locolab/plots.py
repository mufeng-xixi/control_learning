"""Plot learning curves from one or more runs' metrics.csv files.

Example:
    python -m locolab.plots runs/InvertedPendulum-v5_seed0 --out results/curve.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(run_dir: str | Path) -> tuple[list[int], list[float]]:
    steps, rets = [], []
    with (Path(run_dir) / "metrics.csv").open() as f:
        for row in csv.DictReader(f):
            if row["mean_ep_return"] != "nan":
                steps.append(int(row["steps"]))
                rets.append(float(row["mean_ep_return"]))
    return steps, rets


def plot(run_dirs: list[str], out: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for rd in run_dirs:
        steps, rets = load_metrics(rd)
        ax.plot(steps, rets, label=Path(rd).name, linewidth=1.8)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean episode return (training)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default="results/curve.png")
    a = p.parse_args()
    plot(a.run_dirs, a.out)


if __name__ == "__main__":
    main()
