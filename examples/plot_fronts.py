"""Render a reproducible overview of six implemented analytical Pareto fronts.

Install the plot extra first. This writes an image without opening a GUI window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mop_problems import get_problem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/assets/pareto_fronts.png",
    )
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 6.8), layout="constrained", facecolor="#f8fafc")
    titles = {
        "ZDT1": "Convex",
        "ZDT2": "Concave",
        "ZDT3": "Disconnected",
        "ZDT4": "Multimodal distance",
        "ZDT6": "Restricted objective range",
    }
    for index, (name, label) in enumerate(titles.items(), start=1):
        ax = fig.add_subplot(2, 3, index)
        F = get_problem(name).pareto_front(450)
        ax.scatter(F[:, 0], F[:, 1], s=4, color="#0f766e", linewidths=0)
        ax.set(title=f"{name}  |  {label}", xlabel="$f_1$", ylabel="$f_2$")
        ax.grid(alpha=0.2)
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    F = get_problem("DTLZ2").pareto_front(700, seed=42)
    ax.scatter(*F.T, s=5, c=F[:, 2], cmap="viridis", alpha=0.7, linewidths=0)
    ax.set(title="DTLZ2  |  Three objectives", xlabel="$f_1$", ylabel="$f_2$", zlabel="$f_3$")
    ax.view_init(elev=24, azim=40)
    fig.suptitle("MOP-Problems / Analytical Pareto fronts", fontsize=17, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
