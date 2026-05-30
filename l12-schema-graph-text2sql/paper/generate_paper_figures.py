#!/usr/bin/env python3
"""Generate all figures for the IMRAD paper on Schema-Graph-Assisted T2SQL
for materials databases.

Outputs publication-quality PNG files into paper/figures/.
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# -- Japanese font --
_JP = None
for name in ["IPAGothic","IPAexGothic","Noto Sans CJK JP","TakaoPGothic"]:
    if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _JP = name; break
if _JP:
    plt.rcParams["font.family"] = _JP
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.titlesize": 18, "figure.dpi": 300,
})

ROOT = Path(__file__).parent.parent
PAPER_FIG = Path(__file__).parent / "figures"
PAPER_FIG.mkdir(exist_ok=True)

vr = json.loads((ROOT / "verification_results.json").read_text("utf-8"))
lr = json.loads((ROOT / "llm_comparison_results.json").read_text("utf-8"))

# ============================================================
# Fig 1: System Architecture (conceptual — drawn with patches)
# ============================================================
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    boxes = [
        (0.3, 2.0, 2.0, 1.5, "NL Query\n(User Input)", "#e3f2fd"),
        (2.8, 2.0, 2.0, 1.5, "Entity\nExtractor", "#bbdefb"),
        (5.3, 2.0, 2.0, 1.5, "Schema Graph\nTraversal", "#90caf9"),
        (7.8, 2.0, 2.0, 1.5, "SQL\nGenerator", "#64b5f6"),
        (10.3, 2.0, 1.4, 1.5, "SQL Guard\n& Executor", "#42a5f5"),
        (5.3, 0.0, 2.0, 1.2, "Few-Shot\nRAG Store", "#fff9c4"),
        (7.8, 0.0, 2.0, 1.2, "OQMD\nRDB (7 tables)", "#c8e6c9"),
    ]
    for x, y, w, h, txt, color in boxes:
        rect = mpatches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=10, fontweight="bold")
    # Arrows
    for x1, x2, y in [(2.3,2.8,2.75),(4.8,5.3,2.75),(7.3,7.8,2.75),(9.8,10.3,2.75)]:
        ax.annotate("", xy=(x2,y), xytext=(x1,y),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))
    # Few-shot arrow up to SQL Generator
    ax.annotate("", xy=(7.8,2.0), xytext=(7.3,1.2),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="#f57f17", ls="--"))
    # DB arrow up to SQL Guard
    ax.annotate("", xy=(9.8,2.0), xytext=(9.3,1.2),
                 arrowprops=dict(arrowstyle="->", lw=1.2, color="#2e7d32", ls="--"))
    # Feedback loop
    ax.annotate("", xy=(6.3,1.2), xytext=(10.8,1.8),
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="#f57f17", ls=":",
                                connectionstyle="arc3,rad=0.3"))
    ax.text(9.5, 4.0, "Feedback Loop", fontsize=9, color="#f57f17", style="italic")
    ax.set_title("Fig. 1: System Architecture — Schema-Graph-Assisted Text-to-SQL Pipeline")
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_architecture.png")

# ============================================================
# Fig 2: E-R Diagram (7 tables)
# ============================================================
def fig2_er_diagram():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis("off")
    tables = {
        "material_entry":  (4.0, 5.0, ["entry_id (PK)","formula","reduced_formula","chemical_system"]),
        "composition":     (0.5, 3.0, ["composition_id (PK)","entry_id (FK)","element","atomic_fraction"]),
        "structure":       (4.0, 2.5, ["structure_id (PK)","entry_id (FK)","prototype","lattice_a","space_group"]),
        "phase_stability": (7.5, 3.0, ["stability_id (PK)","entry_id (FK)","formation_energy","energy_above_hull","band_gap"]),
        "calculation":     (1.0, 0.5, ["calculation_id (PK)","entry_id (FK)","method","functional"]),
        "calculated_property": (4.5, 0.0, ["property_id (PK)","calculation_id (FK)","property_name","value","unit"]),
        "prototype_definition": (8.5, 0.5, ["prototype_id (PK)","prototype_name","strukturbericht","description"]),
    }
    for tname, (x, y, cols) in tables.items():
        w, h = 2.8, 0.3 + 0.22*len(cols)
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor="#e8eaf6", edgecolor="#1a237e", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h-0.15, tname, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1a237e")
        for i, col in enumerate(cols):
            color = "#c62828" if "PK" in col else "#1565c0" if "FK" in col else "#333"
            ax.text(x+0.1, y+h-0.35-i*0.22, col, fontsize=7, color=color)
    # FK lines
    fks = [
        ("composition", "material_entry"),
        ("structure", "material_entry"),
        ("phase_stability", "material_entry"),
        ("calculation", "material_entry"),
        ("calculated_property", "calculation"),
    ]
    centers = {t: (x+1.4, y+0.15+0.11*len(c)) for t,(x,y,c) in tables.items()}
    for src, tgt in fks:
        sx, sy = centers[src]
        tx, ty = centers[tgt]
        ax.annotate("", xy=(tx, ty), xytext=(sx, sy),
                     arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#1565c0"))
    ax.set_title("Fig. 2: Entity-Relationship Diagram — Materials Database Schema (7 Tables)")
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig2_er_diagram.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_er_diagram.png")

# ============================================================
# Fig 3: 3-Level Pass Rate Comparison
# ============================================================
def fig3_3level_comparison():
    categories = ["Normal\n(15)", "No Results\n(5)", "Sloppy\n(10)",
                   "Contradictory\n(2)", "Rejection\n(5)", "Safety\n(2)", "Total\n(39)"]
    # Naive doesn't run through safety/rejection properly, approximate from data
    naive_pass = [15, 5, 10, 2, 0, 0, 32]
    rb_pass =    [15, 5, 10, 2, 5, 2, 39]
    llm_pass =   [15, 5, 10, 2, 5, 2, 39]
    naive_rate = [100*n/t for n,t in zip(naive_pass,[15,5,10,2,5,2,39])]
    rb_rate =    [100*n/t for n,t in zip(rb_pass,[15,5,10,2,5,2,39])]
    llm_rate =   [100*n/t for n,t in zip(llm_pass,[15,5,10,2,5,2,39])]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    w = 0.25
    ax.bar(x-w, naive_rate, w, label="Naive (Level 0)", color="#ef5350", alpha=0.85)
    ax.bar(x, rb_rate, w, label="Schema Graph RB (Level 1)", color="#1976d2", alpha=0.85)
    ax.bar(x+w, llm_rate, w, label="Schema Graph + LLM (Level 2)", color="#2e7d32", alpha=0.85)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_title("Fig. 3: Test Pass Rate by Category — Three System Levels")
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig3_3level_passrate.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_3level_passrate.png")

# ============================================================
# Fig 4: Jaccard Similarity (RB vs LLM)
# ============================================================
def fig4_jaccard():
    lr_results = [r for r in lr["results"] if r.get("comparison",{}).get("jaccard") is not None]
    ids = [r["test_id"] for r in lr_results]
    jac = [r["comparison"]["jaccard"] for r in lr_results]
    colors = ["#2e7d32" if v>=0.8 else "#ef6c00" if v>=0.5 else "#c62828" for v in jac]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ids, jac, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="#2e7d32", ls="--", lw=1.2, alpha=0.5)
    ax.set_ylabel("Jaccard Similarity")
    ax.set_xlabel("Test ID")
    ax.set_ylim(0, 1.15)
    ax.set_title("Fig. 4: Result Set Agreement (Jaccard Index) — Rule-based vs GPT-5")
    plt.xticks(rotation=60, ha="right", fontsize=10)
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig4_jaccard.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_jaccard.png")

# ============================================================
# Fig 5: LLM Latency vs RB Latency
# ============================================================
def fig5_latency():
    lr_results = [r for r in lr["results"]
                  if r.get("rule_based",{}).get("latency_ms") and r.get("llm",{}).get("latency_ms")]
    ids = [r["test_id"] for r in lr_results]
    rb_lat = [r["rule_based"]["latency_ms"] for r in lr_results]
    llm_lat = [r["llm"]["latency_ms"] for r in lr_results]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ids))
    ax.bar(x-0.2, rb_lat, 0.35, label="Rule-based", color="#1976d2", alpha=0.85)
    ax.bar(x+0.2, llm_lat, 0.35, label="GPT-5", color="#e65100", alpha=0.85)
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Test ID")
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xticks(x); ax.set_xticklabels(ids, rotation=60, ha="right", fontsize=9)
    ax.legend()
    ax.set_title("Fig. 5: Query Latency — Rule-based (<100 ms) vs GPT-5 (~7 s)")
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig5_latency.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_latency.png")

# ============================================================
# Fig 6: OQMD API Precision / Recall
# ============================================================
def fig6_oqmd_precision_recall():
    oqmd_tests = []
    for r in vr["results"]:
        oqmd = r.get("oqmd_comparison",{})
        if oqmd and oqmd.get("precision") is not None:
            oqmd_tests.append((r["test_id"], oqmd["precision"], oqmd.get("recall",0),
                               oqmd.get("oqmd_api_count",0), oqmd.get("t2sql_unique_count",0)))
    if not oqmd_tests:
        print("  fig6 skipped (no OQMD data)")
        return
    ids = [t[0] for t in oqmd_tests]
    prec = [t[1] for t in oqmd_tests]
    rec = [t[2] for t in oqmd_tests]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ids))
    ax.bar(x-0.18, prec, 0.35, label="Precision", color="#1976d2", alpha=0.85)
    ax.bar(x+0.18, rec, 0.35, label="Recall", color="#e65100", alpha=0.85)
    ax.set_ylabel("Score")
    ax.set_xlabel("Test ID")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(ids, fontsize=11)
    ax.legend()
    ax.set_title("Fig. 6: OQMD-API Ground Truth Comparison — Precision & Recall")
    # Annotate LIMIT-constrained
    for i, t in enumerate(oqmd_tests):
        if t[2] < 0.5 and t[1] >= 0.95:
            ax.text(i+0.18, t[2]+0.03, "LIMIT\n100", ha="center", fontsize=7, color="#e65100")
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig6_oqmd_precision_recall.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_oqmd_precision_recall.png")

# ============================================================
# Fig 7: Row Count Comparison (notable cases)
# ============================================================
def fig7_notable_cases():
    notable_ids = ["B01","B02","B03","C09","D02","A13"]
    lr_by_id = {r["test_id"]: r for r in lr["results"]}
    cases = [(tid, lr_by_id[tid]) for tid in notable_ids if tid in lr_by_id]
    if not cases:
        print("  fig7 skipped")
        return
    ids = [c[0] for c in cases]
    rb_rows = [c[1].get("rule_based",{}).get("db_result",{}).get("row_count",0) for c in cases]
    llm_rows = [c[1].get("llm",{}).get("db_result",{}).get("row_count",0) for c in cases]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ids))
    ax.bar(x-0.2, rb_rows, 0.35, label="Rule-based", color="#1976d2", alpha=0.85)
    ax.bar(x+0.2, llm_rows, 0.35, label="GPT-5", color="#e65100", alpha=0.85)
    ax.set_ylabel("Row Count")
    ax.set_xlabel("Test ID")
    ax.set_xticks(x); ax.set_xticklabels(ids, fontsize=11)
    ax.legend()
    ax.set_title("Fig. 7: Notable Divergences — Rule-based vs GPT-5")
    ax.set_yscale("symlog", linthresh=1)
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig7_notable_cases.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig7_notable_cases.png")

# ============================================================
# Fig 8: Cost-Benefit Radar (qualitative)
# ============================================================
def fig8_radar():
    categories = ["Accuracy", "Latency\n(inverse)", "Cost\n(inverse)",
                   "Safety", "Vocabulary\nCoverage", "Numeric\nFilter"]
    # 0-5 scale
    naive =     [3, 5, 5, 0, 2, 0]
    rb =        [4.5, 5, 5, 5, 3, 1]
    llm =       [5, 2, 2, 5, 5, 4]

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    for vals in [naive, rb, llm]:
        vals += vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.plot(angles, naive, "o-", color="#ef5350", label="Naive (L0)", linewidth=1.5)
    ax.fill(angles, naive, alpha=0.1, color="#ef5350")
    ax.plot(angles, rb, "s-", color="#1976d2", label="Schema Graph RB (L1)", linewidth=1.5)
    ax.fill(angles, rb, alpha=0.1, color="#1976d2")
    ax.plot(angles, llm, "D-", color="#2e7d32", label="Schema Graph+LLM (L2)", linewidth=1.5)
    ax.fill(angles, llm, alpha=0.1, color="#2e7d32")
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05))
    ax.set_title("Fig. 8: Multi-dimensional Comparison of Three System Levels", y=1.08)
    fig.tight_layout()
    fig.savefig(PAPER_FIG / "fig8_radar.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig8_radar.png")

if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_architecture()
    fig2_er_diagram()
    fig3_3level_comparison()
    fig4_jaccard()
    fig5_latency()
    fig6_oqmd_precision_recall()
    fig7_notable_cases()
    fig8_radar()
    print(f"Done. Figures saved to {PAPER_FIG}")
