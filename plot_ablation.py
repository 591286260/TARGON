"""Render the Figure 2 ablations from the published cross-validation summary."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = ("Davis", "DrugBank", "KIBA")
METRICS = ("ACC", "F1", "AUC", "AUPR")
SERIES = (
    ("no-diff", "w/o Diff", "#A2ACCD"),
    ("no-trans", "w/o Trans", "#DAD0E5"),
    ("single-strength", r"Single-$\alpha$", "#CBE0E5"),
    ("no-topology", "w/o Topo", "#F39891"),
    ("concat", "Concat", "#F2D8B5"),
    ("full", "TARGON", "#C6D1E4"),
)


def load_summary(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(row["dataset"], row["variant"]): row for row in rows}
    expected = {(dataset, variant) for dataset in DATASETS for variant, _, _ in SERIES}
    if len(keys) != len(rows) or set(keys) != expected:
        raise ValueError("The summary must contain exactly one row per dataset and Figure 2 variant.")
    for row in rows:
        means = np.array([float(row[f"{metric}_mean"]) for metric in METRICS])
        std = np.array([float(row[f"{metric}_std"]) for metric in METRICS])
        if not (np.isfinite(means).all() and np.isfinite(std).all()
                and np.all((0 <= means) & (means <= 1)) and np.all(std >= 0)):
            raise ValueError(f"Invalid means or standard deviations for {row['dataset']}, {row['variant']}.")
    return keys


def draw(results, output):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "axes.linewidth": .75, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8))
    fig.subplots_adjust(left=.059, right=.990, bottom=.185, top=.804, wspace=.255)
    x = np.arange(len(METRICS))
    width = .152
    for axis, dataset in zip(axes, DATASETS):
        for index, (variant, label, color) in enumerate(SERIES):
            row = results[dataset, variant]
            means = [float(row[f"{m}_mean"]) for m in METRICS]
            errors = [float(row[f"{m}_std"]) for m in METRICS]
            axis.bar(x + width * (index - (len(SERIES) - 1) / 2), means,
                     width, yerr=errors, label=label, color=color, edgecolor="black", linewidth=.45,
                     error_kw={"elinewidth": .55, "capsize": 2, "capthick": .55})
        bounds = (.85, 1.00) if dataset == "KIBA" else (.78, .96)
        axis.set(xlim=(-.5, len(METRICS)-.5), ylim=bounds, yticks=np.linspace(*bounds, 4))
        axis.set_xticks(x, ("Acc.", "F1", "AUC", "AUPR"))
        axis.set_xlabel(dataset, labelpad=9, fontsize=12)
        axis.set_axisbelow(True)
        axis.grid(axis="y", linestyle=(0, (3, 2)), color="#9a9a9a", linewidth=.45)
        axis.tick_params(length=3, pad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = (0, 1, 3, 2, 4, 5)
    fig.legend([handles[i] for i in legend_order], [labels[i] for i in legend_order],
               ncol=len(SERIES), loc="upper center",
               bbox_to_anchor=(.51,.984), frameon=False, columnspacing=1.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "ablation_summary.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "fig2.pdf")
    args = parser.parse_args()
    draw(load_summary(args.input), args.output)


if __name__ == "__main__":
    main()
