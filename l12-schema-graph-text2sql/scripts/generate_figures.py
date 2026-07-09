#!/usr/bin/env python3
"""Generate all visualization figures for the paper.

Produces:
  - Fig: Ablation bar chart (5-run mean +/- SD)
  - Fig: Few-shot sensitivity (k vs accuracy by difficulty)
  - Fig: Dictionary size sensitivity
  - Fig: Multi-axis radar chart (by difficulty)
  - Fig: Model comparison
  - Fig: Error type distribution

All figures saved to paper/figures/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Font sizes (doubled per user preference for presentations)
TITLE_SIZE = 24
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 16

plt.rcParams.update({
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.dpi": 150,
})


def fig_ablation_bar():
    """5-run ablation bar chart with error bars."""
    path = PROJECT / "evaluation" / "ablation_multirun_stats.json"
    with open(path) as f:
        stats = json.load(f)

    conditions_order = ["full", "no_fewshot", "no_dict", "no_reranker", "no_guard", "no_nbest", "no_graph"]
    labels = ["Full", "No Few-shot", "No Dict", "No Reranker", "No Guard", "No N-best", "No Graph"]
    means = []
    sds = []
    colors_list = []

    sig_tests = stats.get("significance_tests", {})
    for cond in conditions_order:
        c = stats["conditions"][cond]
        means.append(c["overall_mean"] * 100)
        sds.append(c["overall_std"] * 100)
        p_val = sig_tests.get(cond, {}).get("p_value", 1.0)
        if cond == "full":
            colors_list.append("#2196F3")
        elif p_val < 0.05:
            colors_list.append("#F44336")
        else:
            colors_list.append("#9E9E9E")

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sds, capsize=5, color=colors_list, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation Study (5-run mean $\\pm$ SD)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(70, 90)
    ax.axhline(y=means[0], color="#2196F3", linestyle="--", alpha=0.3)

    # Add significance markers
    for i, cond in enumerate(conditions_order):
        p = sig_tests.get(cond, {}).get("p_value", 1.0)
        if cond != "full":
            if p < 0.001:
                ax.text(i, means[i] + sds[i] + 0.5, "***", ha="center", fontsize=LEGEND_SIZE)
            elif p < 0.01:
                ax.text(i, means[i] + sds[i] + 0.5, "**", ha="center", fontsize=LEGEND_SIZE)
            elif p < 0.05:
                ax.text(i, means[i] + sds[i] + 0.5, "*", ha="center", fontsize=LEGEND_SIZE)
            else:
                ax.text(i, means[i] + sds[i] + 0.5, "n.s.", ha="center", fontsize=TICK_SIZE - 2)

    plt.tight_layout()
    out = FIG_DIR / "ablation_bar.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_fewshot_sensitivity():
    """Few-shot k sensitivity: k vs accuracy by difficulty."""
    path = PROJECT / "evaluation" / "fewshot_sensitivity_results.json"
    with open(path) as f:
        data = json.load(f)

    k_values = data["k_values"]
    diffs = ["easy", "medium", "hard", "very_hard"]
    diff_labels = ["Easy", "Medium", "Hard", "Very Hard"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for diff, label, color, marker in zip(diffs, diff_labels, colors, markers):
        accs = []
        for k in k_values:
            cond = data["conditions"][f"k={k}"]
            accs.append(cond["by_difficulty"].get(diff, 0) * 100)
        ax.plot(k_values, accs, marker=marker, color=color, label=label, linewidth=2, markersize=10)

    # Overall
    overall = [data["conditions"][f"k={k}"]["overall"] * 100 for k in k_values]
    ax.plot(k_values, overall, marker="*", color="black", label="Overall", linewidth=3, markersize=12)

    ax.set_xlabel("Number of Few-shot Examples (k)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Few-shot Sensitivity Analysis")
    ax.set_xticks(k_values)
    ax.legend(loc="lower right")
    ax.set_ylim(40, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "fewshot_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_dict_sensitivity():
    """Dictionary size sensitivity bar chart."""
    path = PROJECT / "evaluation" / "dict_sensitivity_results.json"
    with open(path) as f:
        data = json.load(f)

    configs = ["dict_full", "dict_50%", "dict_25%", "dict_0%"]
    labels = ["Full (61)", "50% (30)", "25% (15)", "0% (none)"]
    diffs = ["easy", "medium", "hard", "very_hard"]
    diff_labels = ["Easy", "Medium", "Hard", "Very Hard"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(labels))
    width = 0.18

    for i, (diff, dl, color) in enumerate(zip(diffs, diff_labels, colors)):
        vals = []
        for cfg in configs:
            vals.append(data["conditions"][cfg]["by_difficulty"].get(diff, 0) * 100)
        ax.bar(x + i * width - 1.5 * width, vals, width, label=dl, color=color, edgecolor="black", linewidth=0.5)

    # Overall line
    overall = [data["conditions"][cfg]["overall"] * 100 for cfg in configs]
    ax.plot(x, overall, marker="*", color="black", linewidth=3, markersize=14, label="Overall", zorder=5)

    ax.set_xlabel("Dictionary Size")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Domain Dictionary Size Sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(30, 105)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "dict_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_multiaxis_radar():
    """Multi-axis radar chart by difficulty."""
    path = PROJECT / "evaluation" / "multiaxis_results.json"
    with open(path) as f:
        data = json.load(f)

    categories = ["Recall", "Precision", "F1", "EM", "SELECT Col", "JOIN Match"]
    diffs = ["easy", "medium", "hard", "very_hard"]
    diff_labels = ["Easy", "Medium", "Hard", "Very Hard"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    N = len(categories)
    angles = [float(a) for a in np.linspace(0, 2 * np.pi, N, endpoint=False)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for diff, label, color in zip(diffs, diff_labels, colors):
        d = data["by_difficulty"].get(diff, {})
        values = [
            d.get("recall", 0) * 100,
            d.get("precision", 0) * 100,
            d.get("f1", 0) * 100,
            d.get("exact_match", 0) * 100,
            d.get("select_col_prec", 0) * 100,
            d.get("join_match", 0) * 100,
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", color=color, label=label, linewidth=2, markersize=8)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=TICK_SIZE)
    ax.set_ylim(0, 105)
    ax.set_title("Multi-axis Evaluation by Difficulty", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    out = FIG_DIR / "multiaxis_radar.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_model_comparison():
    """Model comparison bar chart."""
    # GPT-5.5 baseline from ablation stats
    stat_path = PROJECT / "evaluation" / "ablation_multirun_stats.json"
    with open(stat_path) as f:
        stats = json.load(f)
    full_cond = stats["conditions"]["full"]

    # GPT-4o from model comparison
    mc_path = PROJECT / "evaluation" / "model_comparison_results.json"
    with open(mc_path) as f:
        mc = json.load(f)
    gpt4o = mc["models"]["gpt-4o"]

    diffs = ["easy", "medium", "hard", "very_hard"]
    diff_labels = ["Easy", "Medium", "Hard", "Very Hard"]

    # Build data
    gpt55_by_diff = full_cond["by_difficulty"]
    gpt4o_by_diff = gpt4o["by_difficulty"]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(diffs))
    width = 0.35

    gpt55_vals = [gpt55_by_diff.get(d, {}).get("mean", 0) * 100 for d in diffs]
    gpt55_sds = [gpt55_by_diff.get(d, {}).get("std", 0) * 100 for d in diffs]
    gpt4o_vals = [gpt4o_by_diff.get(d, 0) * 100 for d in diffs]

    ax.bar(x - width / 2, gpt55_vals, width, yerr=gpt55_sds, capsize=5,
           label=f"GPT-5.5 ({full_cond['overall_mean']*100:.1f}%)", color="#2196F3",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, gpt4o_vals, width,
           label=f"GPT-4o ({gpt4o['overall']*100:.1f}%)", color="#FF9800",
           edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("LLM Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(diff_labels)
    ax.legend()
    ax.set_ylim(30, 105)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = FIG_DIR / "model_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_error_distribution():
    """Error type distribution pie/bar chart from failure analysis."""
    path = PROJECT / "evaluation" / "failure_analysis.json"
    with open(path) as f:
        data = json.load(f)

    # Classify error types
    error_types = {
        "Wrong Columns": 0,
        "Missing GROUP BY": 0,
        "JOIN Mismatch": 0,
        "Value Mismatch": 0,
        "Missing WHERE": 0,
    }

    for fail in data["failures"]:
        gen = fail["gen_sql"].upper()
        gold = fail["gold_sql"].upper()
        if fail["select_col_prec"] < 0.5:
            error_types["Wrong Columns"] += 1
        if "GROUP BY" in gold and "GROUP BY" not in gen:
            error_types["Missing GROUP BY"] += 1
        if fail["join_match_rate"] < 1.0:
            error_types["JOIN Mismatch"] += 1
        if "WHERE" in gold and "WHERE" not in gen:
            error_types["Missing WHERE"] += 1
        # Count value mismatch if none of above
        counted = (
            (fail["select_col_prec"] < 0.5) or
            ("GROUP BY" in gold and "GROUP BY" not in gen) or
            (fail["join_match_rate"] < 1.0) or
            ("WHERE" in gold and "WHERE" not in gen)
        )
        if not counted:
            error_types["Value Mismatch"] += 1

    # Remove zero entries
    error_types = {k: v for k, v in error_types.items() if v > 0}

    fig, ax = plt.subplots(figsize=(10, 7))
    labels = list(error_types.keys())
    values = list(error_types.values())
    colors = ["#F44336", "#FF9800", "#FFC107", "#9C27B0", "#607D8B"][:len(labels)]

    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Count (in 10 representative failures)")
    ax.set_title("Error Type Distribution in Failure Cases")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(val), ha="left", va="center", fontsize=TICK_SIZE)

    plt.tight_layout()
    out = FIG_DIR / "error_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    print("Generating figures...")
    fig_ablation_bar()
    fig_fewshot_sensitivity()
    fig_dict_sensitivity()
    fig_multiaxis_radar()
    fig_model_comparison()
    fig_error_distribution()
    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
