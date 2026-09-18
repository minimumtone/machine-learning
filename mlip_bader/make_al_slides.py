#!/usr/bin/env python3
"""Build Japanese slides on the effect of Al in MACE-relaxed BCC HEA cells.

Reads voronoi_per_atom.csv and relax_results.csv, writes presentation figures to
slides/ and assembles slides/al_effect_bcc_hea.pptx.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "slides"
PER_ATOM_CSV = ROOT / "voronoi_per_atom.csv"
RELAX_CSV = ROOT / "relax_results.csv"
PPTX = OUT / "al_effect_bcc_hea.pptx"

PURE = ["Al", "Hf", "Nb", "Ta", "Ti", "V", "Zr"]
AL_PAIRS = ["Al-Nb", "Al-Ti", "Al-V"]
HEAS = ("HfNbTaTiZr", "AlNbTiV")
UNRELIABLE = 0.05

JP_FONT = "Noto Sans CJK JP"
plt.rcParams.update(
    {
        "font.family": [JP_FONT, "DejaVu Sans"],
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "axes.unicode_minus": False,
    }
)

C_AL = "#d62728"
C_OTHER = "#7f7f7f"
C_PRED = "#1f77b4"
C_HEA = "#ff7f0e"


def composition(label: str) -> dict[str, float]:
    if label == "HfNbTaTiZr":
        return {"Hf": 26 / 128, "Nb": 26 / 128, "Ta": 26 / 128, "Ti": 25 / 128, "Zr": 25 / 128}
    if label == "AlNbTiV":
        return {element: 0.25 for element in ("Al", "Nb", "Ti", "V")}
    first, second = label.split("-")
    return {first: 0.5, second: 0.5}


def analyse() -> dict:
    frame = pd.read_csv(PER_ATOM_CSV)
    relax = pd.read_csv(RELAX_CSV)
    grouped = frame.groupby(["label", "seed", "element"], as_index=False)["V_vor_A3"].mean()
    pure = grouped[grouped.label.isin(PURE)].groupby("element")["V_vor_A3"].mean().to_dict()

    binary_delta: dict[tuple[str, str], float] = {}
    omega: dict[str, float] = {}
    for label in sorted(grouped.label.unique()):
        if "-" not in label:
            continue
        first, second = label.split("-")
        binary = grouped[grouped.label == label]
        means = binary.groupby("element")["V_vor_A3"].mean().to_dict()
        for element, other in ((first, second), (second, first)):
            binary_delta[(element, other)] = means[element] - pure[element]
        vveg = (pure[first] + pure[second]) / 2
        omega[label] = binary["V_vor_A3"].mean() / vveg - 1

    hea: dict[str, list[dict]] = {}
    for label in HEAS:
        c = composition(label)
        cell = grouped[grouped.label == label]
        rows = []
        for element in c:
            samples = cell[cell.element == element]["V_vor_A3"]
            delta = float(samples.mean() - pure[element])
            predicted = sum(2 * c[n] * binary_delta[(element, n)] for n in c if n != element)
            rows.append(
                {
                    "element": element,
                    "delta_hea": delta,
                    "std": float(samples.std(ddof=1)),
                    "delta_pred": predicted,
                    "f": delta / predicted if abs(predicted) >= UNRELIABLE else np.nan,
                }
            )
        hea[label] = rows

    return {
        "frame": frame,
        "relax": relax,
        "pure": pure,
        "binary_delta": binary_delta,
        "omega": omega,
        "hea": hea,
    }


def savefig(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_pure_volumes(res) -> Path:
    pure = res["pure"]
    order = sorted(pure, key=pure.get)
    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    colors = [C_AL if e == "Al" else C_OTHER for e in order]
    bars = ax.bar(order, [pure[e] for e in order], color=colors)
    for bar, e in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{pure[e]:.2f}", ha="center", fontsize=18)
    ax.set_ylabel("純元素BCC体積 $V_i^{\\mathrm{pure}}$ (Å$^3$/atom)")
    ax.set_ylim(0, 26)
    ax.set_title("MACE-MP-0 緩和 BCC 純元素の原子体積（128原子セル）")
    return savefig(fig, "fig_pure_volumes.png")


def fig_omega_ranked(res) -> Path:
    omega = res["omega"]
    order = sorted(omega, key=omega.get)
    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    colors = [C_AL if p.startswith("Al-") else C_OTHER for p in order]
    ax.bar(order, [omega[p] * 100 for p in order], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("$\\Omega_{\\mathrm{MACE}}$ = $V/V_{\\mathrm{Vegard}}$ − 1 (%)")
    ax.set_title("15 二元系（50:50）の Vegard 則からの体積偏差（赤: Al 対）")
    ax.tick_params(axis="x", rotation=60)
    for i, p in enumerate(order):
        if p.startswith("Al-"):
            ax.text(i, 0.08, f"{omega[p]*100:.1f}%", ha="center", va="bottom", color=C_AL, fontsize=18, fontweight="bold")
    ax.set_ylim(-3.6, 0.9)
    return savefig(fig, "fig_omega_ranked.png")


def fig_al_binary_delta(res) -> Path:
    bd = res["binary_delta"]
    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    x = np.arange(len(AL_PAIRS))
    width = 0.38
    al = [bd[("Al", p.split("-")[1])] for p in AL_PAIRS]
    partner = [bd[(p.split("-")[1], "Al")] for p in AL_PAIRS]
    b1 = ax.bar(x - width / 2, al, width, color=C_AL, label="Al の過剰体積 $\\Delta V_{\\mathrm{Al}}$")
    b2 = ax.bar(x + width / 2, partner, width, color=C_OTHER, label="相手元素の過剰体積 $\\Delta V_X$")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (0.06 if h >= 0 else -0.06), f"{h:+.2f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=18)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{p}\n(X = {p.split('-')[1]})" for p in AL_PAIRS])
    ax.set_ylabel("二元系での過剰 Voronoi 体積 (Å$^3$)")
    ax.set_ylim(-2.4, 1.5)
    ax.legend(loc="upper left")
    ax.set_title("Al 含有二元系（Al–X 50:50, 3 seed 平均）")
    return savefig(fig, "fig_al_binary_delta.png")


def fig_al_hist(res) -> Path:
    frame = res["frame"]
    pure = res["pure"]
    fig, ax = plt.subplots(figsize=(12, 6.4), constrained_layout=True)
    bins = np.linspace(13.5, 19.0, 45)
    series = [("Al-Nb", "#1f77b4"), ("Al-Ti", "#2ca02c"), ("Al-V", "#9467bd"), ("AlNbTiV", C_AL)]
    for label, color in series:
        vals = frame[(frame.label == label) & (frame.element == "Al")]["V_vor_A3"]
        ax.hist(vals, bins=bins, histtype="stepfilled", alpha=0.35, color=color, edgecolor=color, linewidth=2,
                label=f"{label}  (平均 {vals.mean():.2f})", density=True)
    ax.axvline(pure["Al"], color="black", linestyle="--", linewidth=2, label=f"純 Al BCC ({pure['Al']:.2f})")
    ax.set_xlabel("Al 原子の Voronoi 体積 (Å$^3$)")
    ax.set_ylabel("確率密度")
    ax.set_title("Al 原子の局所体積分布：二元系 vs AlNbTiV")
    ax.legend(fontsize=16)
    return savefig(fig, "fig_al_hist.png")


def fig_alv_hist(res) -> Path:
    frame = res["frame"]
    pure = res["pure"]
    fig, ax = plt.subplots(figsize=(12, 6.4), constrained_layout=True)
    bins = np.linspace(12.5, 17.5, 45)
    sub = frame[frame.label == "Al-V"]
    for element, color in (("Al", C_AL), ("V", "#9467bd")):
        vals = sub[sub.element == element]["V_vor_A3"]
        ax.hist(vals, bins=bins, histtype="stepfilled", alpha=0.4, color=color, edgecolor=color, linewidth=2,
                label=f"{element} in Al–V  (平均 {vals.mean():.2f})", density=True)
        ax.axvline(pure[element], color=color, linestyle="--", linewidth=2, label=f"純 {element} BCC ({pure[element]:.2f})")
    ax.set_xlabel("Voronoi 体積 (Å$^3$)")
    ax.set_ylabel("確率密度")
    ax.set_title("Al–V 二元系：Al は収縮（−1.92）、V は膨張（+0.93）")
    ax.legend(fontsize=16)
    return savefig(fig, "fig_alv_hist.png")


def fig_hea_pred_vs_obs(res) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), constrained_layout=True)
    for ax, label in zip(axes, HEAS):
        rows = res["hea"][label]
        x = np.arange(len(rows))
        width = 0.38
        ax.bar(x - width / 2, [r["delta_pred"] for r in rows], width, color=C_PRED, label="二元系からの予測 $\\Delta V_{\\mathrm{pred}}$")
        ax.bar(x + width / 2, [r["delta_hea"] for r in rows], width, color=C_HEA, yerr=[r["std"] for r in rows], capsize=5,
               label="HEA 実測 (Voronoi) $\\Delta V_{\\mathrm{HEA}}$")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [r["element"] for r in rows])
        for tick in ax.get_xticklabels():
            if tick.get_text() == "Al":
                tick.set_color(C_AL)
                tick.set_fontweight("bold")
        ax.set_title(label)
        ax.set_ylabel("過剰 Voronoi 体積 (Å$^3$)")
        ax.legend(fontsize=15, loc="upper left")
    return savefig(fig, "fig_hea_pred_vs_obs.png")


def fig_survival(res) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6.4), constrained_layout=True)
    labels, values, colors, groups = [], [], [], []
    for label, base in zip(HEAS, ("#1f77b4", "#ff7f0e")):
        start = len(labels)
        for r in res["hea"][label]:
            labels.append(r["element"])
            values.append(r["f"])
            colors.append(C_AL if r["element"] == "Al" else base)
        groups.append((label, start, len(labels) - 1, base))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5)
    ax.text(0.5, 1.22, "破線: $f_i$ = 1（完全加法）", ha="center", fontsize=18)
    for i, v in enumerate(values):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=18, fontweight="bold" if labels[i] == "Al" else None)
    ax.set_xticks(x, labels, fontsize=20)
    for label, lo, hi, base in groups:
        ax.text((lo + hi) / 2, -0.17, label, ha="center", va="top", fontsize=20, color=base, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.axvline(groups[0][2] + 0.5, color="grey", linewidth=1, linestyle=":")
    ax.set_ylabel("生存率 $f_i = \\Delta V_{\\mathrm{HEA}} / \\Delta V_{\\mathrm{pred}}$")
    ax.set_ylim(0, 1.35)
    ax.set_title("二元系過剰体積の HEA での生存率：Al のみ大きく失われる")
    return savefig(fig, "fig_survival.png")


def fig_relax_cost(res) -> Path:
    relax = res["relax"]
    mixed = relax[relax.label.str.contains("-") | relax.label.isin(HEAS)]
    agg = mixed.groupby("label").agg(nsteps=("nsteps", "mean"), fmax=("fmax_final", "max")).reset_index()
    agg = agg.sort_values("nsteps")
    fig, ax = plt.subplots(figsize=(13, 6.4), constrained_layout=True)
    colors = [C_AL if ("Al" in lab) else C_OTHER for lab in agg.label]
    ax.bar(agg.label, agg.nsteps, color=colors)
    ax.set_ylabel("FIRE ステップ数（seed 平均）")
    ax.set_title("緩和コスト：Al 含有セルは 1.5〜2 倍のステップを要する")
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylim(0, agg.nsteps.max() * 1.15)
    for i, (lab, n, fm) in enumerate(zip(agg.label, agg.nsteps, agg.fmax)):
        if lab == "AlNbTiV":
            ax.annotate(f"500 ステップ上限で未収束\n$f_{{\\max}}$ ≤ {fm:.2f} eV/Å", xy=(i, n), xytext=(i - 4.5, n + 250),
                        ha="center", fontsize=16, color=C_AL, arrowprops={"arrowstyle": "->", "color": C_AL, "lw": 1.5})
    return savefig(fig, "fig_relax_cost.png")


# ---------------------------------------------------------------------------
# pptx helpers
# ---------------------------------------------------------------------------

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
RED = RGBColor(0xD6, 0x27, 0x28)
DARK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x66, 0x66, 0x66)


def set_font(run, size: int, bold=False, color=DARK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = JP_FONT


def add_rich(paragraph, segments, size: int, color=DARK, bold=False):
    """segments: list of str or (str, {'sup':True|'sub':True|'bold':True|'color':RGB})."""
    for seg in segments:
        text, opts = (seg, {}) if isinstance(seg, str) else seg
        run = paragraph.add_run()
        run.text = text
        set_font(run, size, bold=opts.get("bold", bold), color=opts.get("color", color))
        if opts.get("sup"):
            run.font._element.set("baseline", "30000")
        if opts.get("sub"):
            run.font._element.set("baseline", "-25000")


def add_title(slide, text: str):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), SLIDE_W - Inches(1.0), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    add_rich(p, [text], 28, color=NAVY, bold=True)
    line = slide.shapes.add_shape(1, Inches(0.5), Inches(1.15), SLIDE_W - Inches(1.0), Emu(28000))
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()


def add_bullets(slide, items, left, top, width, height, size=20):
    """items: list of segments (see add_rich) or (segments, level)."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        segments, level = (item, 0) if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], int)) else item
        if isinstance(segments, str):
            segments = [segments]
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        bullet = "• " if level == 0 else "– "
        p.space_after = Pt(8)
        add_rich(p, [bullet] + list(segments), size - 2 * level)


def add_picture_fit(slide, path: Path, left, top, max_w, max_h):
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    pw, ph = int(w * scale), int(h * scale)
    slide.shapes.add_picture(str(path), left + (max_w - pw) // 2, top + (max_h - ph) // 2, pw, ph)


def add_note(slide, text: str, top=None):
    top = Inches(6.95) if top is None else top
    box = slide.shapes.add_textbox(Inches(0.5), top, SLIDE_W - Inches(1.0), Inches(0.4))
    p = box.text_frame.paragraphs[0]
    add_rich(p, [text], 13, color=GREY)


def add_table(slide, header, rows, left, top, width, height, size=16, highlight_col0="Al"):
    shape = slide.shapes.add_table(len(rows) + 1, len(header), left, top, width, height)
    table = shape.table
    for j, h in enumerate(header):
        cell = table.cell(0, j)
        cell.text = ""
        add_rich(cell.text_frame.paragraphs[0], [h] if isinstance(h, str) else h, size, bold=True)
    for i, row in enumerate(rows, start=1):
        is_al = str(row[0]).startswith(highlight_col0)
        for j, v in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            add_rich(cell.text_frame.paragraphs[0], [str(v)], size, color=RED if is_al else DARK, bold=is_al)
            cell.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT if j > 0 else PP_ALIGN.LEFT


A3 = [" (Å", ("3", {"sup": True}), ")"]


def build_pptx(res, figs: dict[str, Path]) -> None:
    pure = res["pure"]
    omega = res["omega"]
    bd = res["binary_delta"]
    hea = {label: {r["element"]: r for r in rows} for label, rows in res["hea"].items()}
    relax = res["relax"]
    al_hea = hea["AlNbTiV"]["Al"]
    alnbtiv = relax[relax.label == "AlNbTiV"]
    al_pairs_steps = relax[relax.label.isin(AL_PAIRS)]["nsteps"]
    other_pairs_steps = relax[relax.label.str.contains("-") & ~relax.label.isin(AL_PAIRS)]["nsteps"]

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    # 1. Title
    s = prs.slides.add_slide(blank)
    band = s.shapes.add_shape(1, 0, Inches(2.3), SLIDE_W, Inches(2.6))
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.7), Inches(2.5), SLIDE_W - Inches(1.4), Inches(1.4))
    tb.text_frame.word_wrap = True
    add_rich(tb.text_frame.paragraphs[0], ["BCC 高エントロピー合金における Al の影響"], 40, color=RGBColor(255, 255, 255), bold=True)
    p = tb.text_frame.add_paragraph()
    add_rich(p, ["MACE-MP-0 緩和 128 原子セルの元素別 Voronoi 過剰体積解析（mlip_bader）"], 22, color=RGBColor(0xDD, 0xE6, 0xF0))
    tb2 = s.shapes.add_textbox(Inches(0.7), Inches(5.3), SLIDE_W - Inches(1.4), Inches(1.2))
    add_rich(tb2.text_frame.paragraphs[0], ["対象系：純元素 7 種、二元系 15 対（Al–Nb, Al–Ti, Al–V を含む）、HEA 2 種（HfNbTaTiZr, AlNbTiV）"], 18, color=GREY)
    p = tb2.text_frame.add_paragraph()
    add_rich(p, ["データ: mlip_bader/relax_results.csv, voronoi_per_atom.csv, voronoi_summary.md"], 16, color=GREY)

    # 2. Background
    s = prs.slides.add_slide(blank)
    add_title(s, "背景と問い：二元系の過剰体積は HEA で生き残るか？")
    add_bullets(
        s,
        [
            ["HEA の格子定数は Vegard 則（純元素体積の組成平均）からずれる。ずれは二元系の構造因子 Ω", ("sf", {"sub": True}), " の加法和で記述できる（King/Alonso 型モデル）"],
            ["加法モデルの前提：二元系 A–B で生じた元素 i の過剰体積 ΔV", ("i", {"sub": True}), "(j) が、多元系でも近傍組成に比例してそのまま「生存」する"],
            ["本解析はこの前提を MLIP（MACE-MP-0）で直接検証：二元系 → HEA での", ("元素別生存率 f", {"bold": True}), ("i", {"sub": True, "bold": True}), " を Voronoi 体積で測る"],
            [("Al に注目する理由", {"bold": True})],
            (["Al は平衡構造が FCC で BCC は力学的に不安定 → BCC 環境での実効体積の定義自体が曖昧"], 1),
            (["耐火 BCC HEA（Nb, Ti, V, Ta, …）に軽量化・耐酸化目的で添加される主要元素"], 1),
            (["格子定数論文でも Al 含有 HEA（AlCoMnNiV）が独立テスト中で最大の Vegard 誤差を示し、Al–Co, Al–Mn など Al 対の Ω", ("sf", {"sub": True}), " が強い負値（体積収縮）"], 1),
        ],
        Inches(0.6), Inches(1.4), SLIDE_W - Inches(1.2), Inches(5.4), size=20,
    )

    # 3. Method
    s = prs.slides.add_slide(blank)
    add_title(s, "手法：MACE-MP-0 緩和 + 周期 Voronoi 体積")
    add_bullets(
        s,
        [
            ["モデル：MACE-MP-0 small（float64, CPU, スピン自由度なし）。FIRE 緩和 ≤ 500 ステップ（未収束セルは +1500）"],
            ["セル：BCC 4×4×4 = 128 原子。純元素 7 種（Al, Hf, Nb, Ta, Ti, V, Zr）、二元系 15 対 × 3 seed（50:50 ランダム占有）、HEA 2 種 × 3 seed"],
            ["体積：scipy Voronoi（3×3×3 イメージ）で元素別原子体積 V", ("i", {"sub": True}), " を算出（Σ V", ("i", {"sub": True}), " = セル体積を検証）"],
            ["二元系過剰体積：ΔV", ("i", {"sub": True}), "(j) = V", ("i", {"sub": True}), "(A–B) − V", ("i", {"sub": True}), ("pure", {"sup": True}), "、 Ω", ("MACE", {"sub": True}), " = V/V", ("Vegard", {"sub": True}), " − 1"],
            ["HEA 予測（完全加法）：ΔV", ("i", {"sub": True}), ("pred", {"sup": True}), " = Σ", ("j≠i", {"sub": True}), " 2c", ("j", {"sub": True}), " ΔV", ("i", {"sub": True}), "(j)"],
            ["生存率：", ("f", {"bold": True}), ("i", {"sub": True, "bold": True}), (" = ΔV", {"bold": True}), ("i", {"sub": True, "bold": True}), ("HEA", {"sup": True, "bold": True}), (" / ΔV", {"bold": True}), ("i", {"sub": True, "bold": True}), ("pred", {"sup": True, "bold": True}), "　（|ΔV", ("pred", {"sup": True}), "| < 0.05 Å", ("3", {"sup": True}), " は判定不能扱い）"],
            ["f", ("i", {"sub": True}), " ≈ 1：二元系の効果がそのまま転写　／　f", ("i", {"sub": True}), " ≪ 1：多元系環境で相殺・消失"],
        ],
        Inches(0.6), Inches(1.4), SLIDE_W - Inches(1.2), Inches(5.4), size=20,
    )

    # 4. Pure volumes
    s = prs.slides.add_slide(blank)
    add_title(s, "純元素 BCC 体積：Al は Ti と V の間の「中間サイズ」")
    add_picture_fit(s, figs["pure"], Inches(0.4), Inches(1.35), Inches(8.2), Inches(5.5))
    add_bullets(
        s,
        [
            [f"Al (BCC, MACE): V = {pure['Al']:.2f} Å", ("3", {"sup": True}), f"（a = {relax[relax.label=='Al'].a_bcc_A.iloc[0]:.3f} Å）"],
            ["実験 FCC Al の原子体積 16.6 Å", ("3", {"sup": True}), " とほぼ同じ → MACE は BCC-Al を FCC 相当の体積で記述"],
            [f"Ti ({pure['Ti']:.2f}) > Al ({pure['Al']:.2f}) > V ({pure['V']:.2f})：AlNbTiV 中では Al は「大きい元素」ではない"],
            ["純 Al BCC セルの緩和には 120 ステップ（他の純元素は 17–31）→ 力学的不安定性の兆候"],
        ],
        Inches(8.7), Inches(1.6), Inches(4.3), Inches(5.2), size=17,
    )

    # 5. Omega ranked
    s = prs.slides.add_slide(blank)
    add_title(s, "二元系の Vegard 偏差：Al 対が収縮側の上位（1・2・5 位）")
    add_picture_fit(s, figs["omega"], Inches(0.4), Inches(1.35), Inches(8.6), Inches(5.5))
    add_bullets(
        s,
        [
            [f"Al–V: Ω = {omega['Al-V']*100:.1f}%、Al–Ti: {omega['Al-Ti']*100:.1f}%、Al–Nb: {omega['Al-Nb']*100:.1f}%"],
            [f"非 Al 対の最大収縮は Ta–Zr ({omega['Ta-Zr']*100:.1f}%)、Nb–Zr ({omega['Nb-Zr']*100:.1f}%)；他は |Ω| < 0.6%"],
            ["Al を含む 3 対がすべて収縮側の上位 5 位以内 → Al–遷移金属間の d–sp 混成・電荷移動による結合短縮を示唆"],
            ["格子定数論文の DFT-B2 Ω", ("sf", {"sub": True}), "（Al 対は全て負）と符号が整合"],
        ],
        Inches(9.0), Inches(1.6), Inches(4.0), Inches(5.2), size=17,
    )

    # 6. Al binary delta
    s = prs.slides.add_slide(blank)
    add_title(s, "Al–X 二元系の元素別過剰体積：収縮は主に Al 側で起こる")
    add_picture_fit(s, figs["al_binary"], Inches(0.4), Inches(1.35), Inches(8.2), Inches(5.5))
    add_bullets(
        s,
        [
            ["Al–V: ΔV", ("Al", {"sub": True}), f" = {bd[('Al','V')]:+.2f}、ΔV", ("V", {"sub": True}), f" = {bd[('V','Al')]:+.2f} Å", ("3", {"sup": True}), " → Al が大きく縮み、小さい V が膨らむ（体積の再分配）"],
            [f"Al–Ti: 両元素とも縮む（Al {bd[('Al','Ti')]:+.2f}, Ti {bd[('Ti','Al')]:+.2f}）→ 対全体の純収縮"],
            [f"Al–Nb: 両元素とも縮む（Al {bd[('Al','Nb')]:+.2f}, Nb {bd[('Nb','Al')]:+.2f}）"],
            ["非 Al 対（Hf–Ti, Nb–V など）は「大きい元素が縮み小さい元素が膨らむ」ほぼ対称な再分配で Ω ≈ 0"],
            ["Al 対は再分配に加えて", ("正味の収縮", {"bold": True}), "が乗る点が特徴"],
        ],
        Inches(8.7), Inches(1.6), Inches(4.3), Inches(5.2), size=17,
    )

    # 7. Al-V hist
    s = prs.slides.add_slide(blank)
    add_title(s, "Al–V 二元系の原子体積分布：Al と V のサイズ差が縮まる")
    add_picture_fit(s, figs["alv_hist"], Inches(0.4), Inches(1.35), Inches(8.4), Inches(5.5))
    sub = res["frame"][res["frame"].label == "Al-V"]
    add_bullets(
        s,
        [
            [f"純元素差 {pure['Al']-pure['V']:.2f} Å", ("3", {"sup": True}), f" → Al–V 中では {sub[sub.element=='Al'].V_vor_A3.mean()-sub[sub.element=='V'].V_vor_A3.mean():.2f} Å", ("3", {"sup": True}), " まで縮小"],
            ["Al の分布は純元素値より完全に低体積側にシフト；V は高体積側へ"],
            ["分布幅（seed 内標準偏差 ≈ 0.3–0.6 Å", ("3", {"sup": True}), "）は局所近傍の組成揺らぎを反映"],
            ["Al は「周囲に合わせて縮む」柔らかい元素として振る舞う"],
        ],
        Inches(8.9), Inches(1.6), Inches(4.1), Inches(5.2), size=17,
    )

    # 8. HEA predicted vs observed
    s = prs.slides.add_slide(blank)
    add_title(s, "HEA 検証：HfNbTaTiZr は加法的、AlNbTiV は Al の収縮が消失")
    add_picture_fit(s, figs["hea"], Inches(0.3), Inches(1.3), Inches(9.2), Inches(5.6))
    add_bullets(
        s,
        [
            ["HfNbTaTiZr：5 元素すべて予測と実測が一致（誤差 ≤ 0.3 Å", ("3", {"sup": True}), "）"],
            [f"AlNbTiV の Al：予測 {al_hea['delta_pred']:+.2f} → 実測 {al_hea['delta_hea']:+.2f} Å", ("3", {"sup": True}), "（seed 間 σ = ", f"{al_hea['std']:.2f}", "）"],
            [f"Ti も予測 {hea['AlNbTiV']['Ti']['delta_pred']:+.2f} → 実測 {hea['AlNbTiV']['Ti']['delta_hea']:+.2f} と半減"],
            [f"V は {hea['AlNbTiV']['V']['delta_hea']:+.2f}（予測 {hea['AlNbTiV']['V']['delta_pred']:+.2f}）で加法的に膨張"],
            ["エラーバー：3 seed の元素平均体積の標準偏差"],
        ],
        Inches(9.5), Inches(1.6), Inches(3.6), Inches(5.2), size=16,
    )

    # 9. Survival ratio
    s = prs.slides.add_slide(blank)
    add_title(s, f"生存率 f：Al だけが f = {al_hea['f']:.2f}（二元系効果の 7 割を喪失）")
    add_picture_fit(s, figs["survival"], Inches(0.4), Inches(1.35), Inches(8.6), Inches(5.5))
    hz = hea["HfNbTaTiZr"]
    add_bullets(
        s,
        [
            [f"HfNbTaTiZr：f = {min(r['f'] for r in hz.values()):.2f}–{max(r['f'] for r in hz.values()):.2f} → 二元系の過剰体積がほぼ完全に転写（q ≈ 1 の描像と整合）"],
            [f"AlNbTiV：V {hea['AlNbTiV']['V']['f']:.2f}、Nb {hea['AlNbTiV']['Nb']['f']:.2f} は加法的だが、Ti {hea['AlNbTiV']['Ti']['f']:.2f}、", ("Al ", {"bold": True, "color": RED}), (f"{al_hea['f']:.2f}", {"bold": True, "color": RED}), " と Al 周りで加法性が破れる"],
            ["セル全体でも AlNbTiV は f", ("cell", {"sub": True}), " = −0.45（Vegard より膨張）vs HfNbTaTiZr 1.46"],
            ["解釈：Al–V の強い収縮（−1.92）は Al–V 対が多数を占める二元系特有で、4 元系では Al 近傍の V 濃度が半分になり Nb/Ti に置き換わるため、Al–V 電荷移動による収縮が非線形に弱まる"],
        ],
        Inches(9.0), Inches(1.6), Inches(4.0), Inches(5.2), size=16,
    )

    # 10. Al local volume hist
    s = prs.slides.add_slide(blank)
    add_title(s, "Al 原子の局所体積：HEA 中では純 Al に近い値へ戻る")
    add_picture_fit(s, figs["al_hist"], Inches(0.4), Inches(1.35), Inches(8.6), Inches(5.5))
    fr = res["frame"]
    al_in = {lab: fr[(fr.label == lab) & (fr.element == "Al")].V_vor_A3 for lab in AL_PAIRS + ["AlNbTiV"]}
    add_bullets(
        s,
        [
            [f"Al–V 中の Al（{al_in['Al-V'].mean():.2f}）だけが大きく低体積側；Al–Nb ({al_in['Al-Nb'].mean():.2f})、Al–Ti ({al_in['Al-Ti'].mean():.2f}) は純 Al ({pure['Al']:.2f}) に近い"],
            [f"AlNbTiV 中の Al は {al_in['AlNbTiV'].mean():.2f} Å", ("3", {"sup": True}), f"（σ = {al_in['AlNbTiV'].std():.2f}）：分布が広がり、平均は純 Al の −0.33 のみ"],
            ["二元系の「Al–V 収縮」は V 濃度 50% という極端な環境でのみ顕在化 → 組成に対して非線形"],
            ["Al の体積は近傍の V 数に強く依存すると推測（Bader 電荷での確認が次段階）"],
        ],
        Inches(9.0), Inches(1.6), Inches(4.0), Inches(5.2), size=16,
    )

    # 11. Relaxation cost / caveats
    s = prs.slides.add_slide(blank)
    add_title(s, "注意点：Al 含有セルは緩和が困難（BCC-Al の不安定性）")
    add_picture_fit(s, figs["relax"], Inches(0.4), Inches(1.35), Inches(8.2), Inches(5.5))
    add_bullets(
        s,
        [
            [f"Al–X 二元系：平均 {al_pairs_steps.mean():.0f} ステップ（非 Al 対 {other_pairs_steps.mean():.0f}）。9 セル全て 500 超で延長緩和が必要"],
            ["AlNbTiV：3 seed とも 500 ステップで未収束。seed 0 は f", ("max", {"sub": True}), f" = {alnbtiv.fmax_final.min():.3f}、seed 1/2 は ≈ {alnbtiv.fmax_final.max():.2f} eV/Å"],
            [f"seed 間で体積が {alnbtiv.volume_A3.min()/128:.2f}–{alnbtiv.volume_A3.max()/128:.2f} Å", ("3", {"sup": True}), "/atom と 2% ばらつく → f", ("Al", {"sub": True}), " = 0.28 の不確かさは大きい"],
            ["Al-Ti seed 1 も他 seed より 1.2% 大きい体積で収束 → 複数の局所安定構造の存在"],
            ["MACE-MP-0 はスピン非分極、占有はランダム（SQS ではない）、Voronoi 体積 ≠ Bader 体積"],
        ],
        Inches(8.7), Inches(1.6), Inches(4.3), Inches(5.2), size=16,
    )

    # 12. Next steps
    s = prs.slides.add_slide(blank)
    add_title(s, "次のステップ：Al の影響を電子構造で裏付ける")
    add_bullets(
        s,
        [
            ["AlNbTiV 3 seed の延長緩和（+1500 ステップ）で f", ("Al", {"sub": True}), " を再評価し、seed 依存性を確定させる"],
            ["VASP 一点計算 + Henkelman Bader 解析（入力は vasp_bader/ に生成済み：Al, Al–Nb, Al–V, AlNbTiV ほか）"],
            (["Bader 体積で Voronoi 結果を検証、Bader 電荷 ΔQ で Al → 遷移金属の電荷移動量を定量（Al–V vs AlNbTiV）"], 1),
            ["Al 原子ごとの近傍組成（V 数、Nb 数…）と Voronoi 体積の相関解析 → 「Al の収縮は V 近傍数に比例するか」を検証"],
            ["組成スイープ（Al", ("x", {"sub": True}), "(NbTiV)", ("1−x", {"sub": True}), "）で f", ("Al", {"sub": True}), " の Al 濃度依存を取得し、Ω", ("sf", {"sub": True}), " 加法モデルへの非線形補正の形を決める"],
            ["icet SQS 占有・MACE medium での再計算により、ランダム占有・小モデル由来の誤差を切り分ける"],
        ],
        Inches(0.6), Inches(1.4), SLIDE_W - Inches(1.2), Inches(5.4), size=20,
    )

    # 13. Summary
    s = prs.slides.add_slide(blank)
    add_title(s, "まとめ：Al は BCC HEA で「加法的でない」唯一の元素")
    add_bullets(
        s,
        [
            [("二元系：", {"bold": True}), f"Al 対は Vegard 収縮側の上位（Ω = {omega['Al-V']*100:.1f}〜{omega['Al-Nb']*100:.1f}%、Al–V が 15 対中最大）。特に Al–V では Al が −1.92 Å", ("3", {"sup": True}), " 縮み V が +0.93 膨らむ"],
            [("HEA：", {"bold": True}), "HfNbTaTiZr は 5 元素とも f ≈ 0.88–1.12 で完全加法。AlNbTiV では ", ("f", {"bold": True, "color": RED}), ("Al", {"sub": True, "bold": True, "color": RED}), (" = 0.28", {"bold": True, "color": RED}), "、Ti も 0.45 と Al 周辺で加法性が破れる"],
            [("機構仮説：", {"bold": True}), "Al の収縮は Al–V 電荷移動による近傍依存効果で、4 元系で V 近傍が希釈されると非線形に弱まる"],
            [("含意：", {"bold": True}), "Ω", ("sf", {"sub": True}), " 加法モデル（q = 1）は耐火 BCC HEA では妥当だが、Al 含有 HEA では Al 対の Ω を過大に効かせる → 格子定数を過小予測する方向のバイアス"],
            [("留保：", {"bold": True}), "AlNbTiV は未収束・seed 依存性あり。Bader 解析と延長緩和で定量値を確定する必要がある"],
        ],
        Inches(0.6), Inches(1.4), SLIDE_W - Inches(1.2), Inches(5.4), size=20,
    )

    # 14. Appendix table
    s = prs.slides.add_slide(blank)
    add_title(s, "付録：数値一覧（Voronoi, MACE-MP-0 small, 3 seed 平均）")
    header = ["系 / 元素", ["ΔV", ("pred", {"sub": True})] + A3, ["ΔV", ("HEA", {"sub": True})] + A3, ["seed σ"] + A3, ["f", ("i", {"sub": True})]]
    rows = []
    for label in HEAS:
        for r in res["hea"][label]:
            rows.append([f"{r['element']} in {label}", f"{r['delta_pred']:+.3f}", f"{r['delta_hea']:+.3f}", f"{r['std']:.3f}", f"{r['f']:.3f}"])
    add_table(s, header, rows, Inches(0.6), Inches(1.4), Inches(7.2), Inches(4.8), size=14)
    header2 = ["Al–X 対", ["ΔV", ("Al", {"sub": True})] + A3, ["ΔV", ("X", {"sub": True})] + A3, ["Ω", ("MACE", {"sub": True}), " (%)"]]
    rows2 = [[p, f"{bd[('Al', p.split('-')[1])]:+.3f}", f"{bd[(p.split('-')[1], 'Al')]:+.3f}", f"{omega[p]*100:.2f}"] for p in AL_PAIRS]
    add_table(s, header2, rows2, Inches(8.1), Inches(1.4), Inches(4.7), Inches(1.8), size=14)
    add_note(s, "出典: mlip_bader/voronoi_summary.md, relax_results.csv（コミット済みデータから make_al_slides.py が再計算）", top=Inches(6.6))

    OUT.mkdir(exist_ok=True)
    prs.save(PPTX)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    res = analyse()
    figs = {
        "pure": fig_pure_volumes(res),
        "omega": fig_omega_ranked(res),
        "al_binary": fig_al_binary_delta(res),
        "alv_hist": fig_alv_hist(res),
        "hea": fig_hea_pred_vs_obs(res),
        "survival": fig_survival(res),
        "al_hist": fig_al_hist(res),
        "relax": fig_relax_cost(res),
    }
    build_pptx(res, figs)
    for label, rows in res["hea"].items():
        for r in rows:
            print(f"{label} {r['element']}: pred={r['delta_pred']:+.3f} hea={r['delta_hea']:+.3f} f={r['f']:.3f}")
    print(f"Wrote {PPTX}")


if __name__ == "__main__":
    main()
