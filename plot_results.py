"""
Create tables and plots from experiment_results.json.

Example:
  python3 plot_results.py --results results_limit50/experiment_results.json --outdir reports/limit50
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


STRATEGIES = [
    ("greedy", "Greedy"),
    ("majority_vote", "Majority Vote"),
    ("cpw", "CPW"),
    ("scw", "SCW"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create experiment result tables and plots")
    parser.add_argument("--results", default="results/experiment_results.json",
                        help="Path to experiment_results.json")
    parser.add_argument("--outdir", default="reports",
                        help="Directory for generated tables and plots")
    return parser.parse_args()


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def rows_from_results(results):
    rows = []
    for result in results.values():
        model = result["model"]
        dataset = result["dataset"]
        greedy = result.get("greedy")

        for key, label in STRATEGIES:
            if key not in result:
                continue
            accuracy = float(result[key])
            improvement = accuracy - greedy if greedy is not None else 0.0
            rows.append({
                "model": model,
                "dataset": dataset,
                "strategy": label,
                "accuracy": accuracy,
                "improvement_over_greedy": improvement,
            })
    return rows


def strategy_averages(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row["accuracy"])
    return {strategy: sum(values) / len(values) for strategy, values in grouped.items()}


def model_averages(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["strategy"])].append(row["accuracy"])
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "dataset", "strategy", "accuracy", "improvement_over_greedy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(results, path):
    lines = [
        "| Model / Dataset | Greedy | Majority Vote | CPW | SCW |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, result in results.items():
        lines.append(
            "| {key} | {greedy:.1f} | {majority_vote:.1f} | {cpw:.1f} | {scw:.1f} |".format(
                key=key,
                greedy=result.get("greedy", 0.0),
                majority_vote=result.get("majority_vote", 0.0),
                cpw=result.get("cpw", 0.0),
                scw=result.get("scw", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    })
    return plt


def _save_bar_plot(plt, labels, values, title, ylabel, path, color="#4C78A8"):
    width = max(7, len(labels) * 0.7)
    fig, ax = plt.subplots(figsize=(width, 4.5))
    bars = ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(100, max(values) + 8))
    ax.tick_params(axis="x", rotation=35)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_strategy_averages(rows, outdir):
    plt = _load_matplotlib()
    averages = strategy_averages(rows)
    labels = [label for _, label in STRATEGIES if label in averages]
    values = [averages[label] for label in labels]
    _save_bar_plot(
        plt,
        labels,
        values,
        "Average Accuracy by Strategy",
        "Accuracy (%)",
        outdir / "average_accuracy_by_strategy.png",
        color="#4C78A8",
    )


def plot_model_averages(rows, outdir):
    plt = _load_matplotlib()
    averages = model_averages(rows)
    models = sorted({row["model"] for row in rows})
    labels = []
    values = []
    for model in models:
        for _, strategy in STRATEGIES:
            key = (model, strategy)
            if key in averages:
                labels.append(f"{model}\n{strategy}")
                values.append(averages[key])
    _save_bar_plot(
        plt,
        labels,
        values,
        "Average Accuracy by Model and Strategy",
        "Accuracy (%)",
        outdir / "average_accuracy_by_model_strategy.png",
        color="#59A14F",
    )


def plot_grouped_results(results, outdir):
    plt = _load_matplotlib()
    groups = list(results.keys())
    labels = [label for _, label in STRATEGIES]
    x_positions = list(range(len(groups)))
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(max(9, len(groups) * 0.9), 5.0))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    for idx, (key, label) in enumerate(STRATEGIES):
        offsets = [x + (idx - 1.5) * bar_width for x in x_positions]
        values = [float(results[group].get(key, 0.0)) for group in groups]
        bars = ax.bar(offsets, values, width=bar_width, label=label, color=colors[idx])
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7)

    ax.set_title("Accuracy by Model, Dataset, and Strategy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(outdir / "accuracy_by_model_dataset_strategy.png")
    plt.close(fig)


def plot_improvement_over_greedy(rows, outdir):
    plt = _load_matplotlib()
    improved = [row for row in rows if row["strategy"] != "Greedy"]
    labels = [f"{row['model']}/{row['dataset']}\n{row['strategy']}" for row in improved]
    values = [row["improvement_over_greedy"] for row in improved]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in values]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.42), 5.0))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Improvement Over Greedy")
    ax.set_ylabel("Accuracy points")
    ax.tick_params(axis="x", rotation=80)
    ax.bar_label(bars, fmt="%+.1f", padding=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "improvement_over_greedy.png")
    plt.close(fig)


def write_reports(results_path, outdir):
    results_path = Path(results_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / ".cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    results = load_results(results_path)
    rows = rows_from_results(results)

    write_csv(rows, outdir / "results_long.csv")
    write_markdown_table(results, outdir / "results_table.md")
    plot_grouped_results(results, outdir)
    plot_strategy_averages(rows, outdir)
    plot_model_averages(rows, outdir)
    plot_improvement_over_greedy(rows, outdir)

    return {
        "csv": outdir / "results_long.csv",
        "markdown": outdir / "results_table.md",
        "plots": [
            outdir / "accuracy_by_model_dataset_strategy.png",
            outdir / "average_accuracy_by_strategy.png",
            outdir / "average_accuracy_by_model_strategy.png",
            outdir / "improvement_over_greedy.png",
        ],
    }


def main():
    args = parse_args()
    outputs = write_reports(args.results, args.outdir)
    print(f"Wrote CSV: {outputs['csv']}")
    print(f"Wrote Markdown table: {outputs['markdown']}")
    for plot in outputs["plots"]:
        print(f"Wrote plot: {plot}")


if __name__ == "__main__":
    main()
