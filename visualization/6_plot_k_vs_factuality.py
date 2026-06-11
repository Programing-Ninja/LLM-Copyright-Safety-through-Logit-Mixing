import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install dependencies with: "
        "pip install -r requirement.txt"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    BASE_DIR / "outputs" / "factuality_scores.json",
    BASE_DIR / "factuality_output" / "outputs" / "factuality_scores.json",
]


def default_input_file():
    for path in DEFAULT_INPUTS:
        if path.exists():
            return path
    return DEFAULT_INPUTS[0]


def load_scores(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    for idx, row in enumerate(rows):
        for key in ("method", "k", "factuality_score"):
            if key not in row:
                raise ValueError(f"Row {idx} in {path} is missing required key: {key}")
    return rows


def to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    return sum(values) / len(values) if values else 0.0


def plot_summary_on_axis(ax, summary, title, with_errorbars=False):
    by_method = defaultdict(list)
    for row in summary:
        by_method[row["method"]].append(row)

    for method, group in sorted(by_method.items()):
        group = sorted(group, key=lambda row: row["k"])
        xs = [row["k"] for row in group]
        ys = [row["mean_factuality"] for row in group]
        if with_errorbars:
            ax.errorbar(
                xs,
                ys,
                yerr=[row["sem_factuality"] for row in group],
                marker="o",
                linewidth=2,
                capsize=4,
                label=method,
            )
        else:
            ax.plot(xs, ys, marker="o", linewidth=2, label=method)

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel("Mean factuality")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.7)


def summarize_by_k(rows):
    groups = defaultdict(list)
    for row in rows:
        k = to_float(row.get("k"))
        score = to_float(row.get("factuality_score"))
        if k is None or score is None:
            continue
        groups[(row.get("method"), k)].append(score)

    if not groups:
        raise ValueError("No rows with non-null k found. Nothing to plot.")

    summary = []
    for (method, k), scores in sorted(groups.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        avg = mean(scores)
        variance = mean([(score - avg) ** 2 for score in scores]) if len(scores) > 1 else 0.0
        std = math.sqrt(variance)
        summary.append(
            {
                "method": method,
                "k": k,
                "mean_factuality": avg,
                "std_factuality": std,
                "n": len(scores),
                "sem_factuality": std / math.sqrt(len(scores)) if scores else 0.0,
            }
        )
    return summary


def plot_overall(summary, out_file):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    plot_summary_on_axis(ax, summary, "Overall", with_errorbars=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def plot_by_domain(rows, out_file):
    groups = defaultdict(list)
    for row in rows:
        domain = row.get("domain")
        k = to_float(row.get("k"))
        score = to_float(row.get("factuality_score"))
        if domain is None or k is None or score is None:
            continue
        groups[(domain, row.get("method"), k)].append(score)

    if not groups:
        return False

    overall_summary = summarize_by_k(rows)
    summary = []
    for (domain, method, k), scores in sorted(groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]), x[0][2])):
        summary.append(
            {
                "domain": domain,
                "method": method,
                "k": k,
                "mean_factuality": mean(scores),
            }
        )

    domains = sorted({row["domain"] for row in summary})
    panel_count = len(domains) + 1
    cols = min(3, panel_count)
    rows = (panel_count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.ravel()

    plot_summary_on_axis(axes_flat[0], overall_summary, "Overall", with_errorbars=True)

    for ax, domain in zip(axes_flat[1:], domains):
        by_method = defaultdict(list)
        for row in summary:
            if row["domain"] == domain:
                by_method[row["method"]].append(row)
        for method, group in sorted(by_method.items()):
            group = sorted(group, key=lambda row: row["k"])
            ax.plot(
                [row["k"] for row in group],
                [row["mean_factuality"] for row in group],
                marker="o",
                linewidth=2,
                label=method,
            )
        ax.set_xscale("log")
        ax.set_title(str(domain))
        ax.set_xlabel("K")
        ax.set_ylabel("Mean factuality")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.7)

    for ax in axes_flat[panel_count:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle("K vs factuality score", y=1.02)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_summary_csv(summary, path):
    fieldnames = ["method", "k", "mean_factuality", "std_factuality", "sem_factuality", "n"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser(description="Plot K vs factuality scores.")
    parser.add_argument("--input", type=Path, default=default_input_file())
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    input_file = args.input
    if not input_file.exists():
        raise FileNotFoundError(f"Could not find factuality scores file: {input_file}")

    out_dir = args.out_dir or input_file.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_scores(input_file)
    summary = summarize_by_k(rows)
    summary_file = out_dir / "k_vs_factuality_summary.csv"
    save_summary_csv(summary, summary_file)

    domain_plot = out_dir / "k_vs_factuality_by_domain.svg"
    made_domain_plot = plot_by_domain(rows, domain_plot)

    print(f"Saved summary: {summary_file}")
    if made_domain_plot:
        print(f"Saved combined plot: {domain_plot}")


if __name__ == "__main__":
    main()
