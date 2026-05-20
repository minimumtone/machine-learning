#!/usr/bin/env python3
"""
全元素ペアのVegard組成軸プロットをHTML形式で出力.
各ページのプロットをbase64エンコードしてHTMLに直接埋め込む.
"""

import base64
import io
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["IPAPGothic", "IPAGothic", "DejaVu Sans"],
})

KING_ATOMIC_VOLUMES = {
    "Al":16.602,"Cu":11.810,"Ni":10.941,"Pd":14.716,"Pt":15.095,
    "Au":16.966,"Ag":17.061,"Ir":14.155,"Rh":13.754,
    "Co":11.073,"Ti":17.649,"Zr":23.279,"Hf":22.312,
    "Ru":13.571,"Os":13.977,"Re":14.712,"Mn":12.210,"Zn":15.207,
    "Fe":11.776,"Cr":12.008,"V":13.824,"Nb":17.978,"Mo":15.583,
    "Ta":18.014,"W":15.850,"Si":20.024,"Ge":22.634,"Be":8.111,
    "Mg":23.240,"Y":33.018,"La":37.168,"Ce":34.367,"Sc":24.987,
    "B":7.241,"P":23.000,"Sn":27.053,"Pb":30.321,
    "Er":30.66,"Tb":32.09,"Dy":31.54,"Ca":43.63,
    "Ba":50.0,"Sr":56.0,"Bi":35.39,"Tl":28.59,"In":26.16,
    "Cd":21.58,"Ga":19.58,"As":21.39,"Se":25.81,"Te":34.32,
}

HEA_ELEMENTS = sorted([
    "Al","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Nb","Mo","Hf","Ta","W","Re","Pd","Pt","Au","Ag",
    "Sc","Y","Mg","Si","Ge","Sn","Ru","Rh","Os","Ir",
])

OUT = Path("vegard_comparison_output")


def plot_vegard_composition(ax, row, show_ylabel=True, show_xlabel=True):
    elX = row["elX"]
    elY = row["elY"]
    vX = KING_ATOMIC_VOLUMES.get(elX, np.nan)
    vY = KING_ATOMIC_VOLUMES.get(elY, np.nan)
    if np.isnan(vX) or np.isnan(vY):
        ax.set_visible(False)
        return

    c = np.linspace(0, 1, 100)
    v_vegard = (1 - c) * vY + c * vX
    ax.plot(c, v_vegard, "k--", lw=1.2, alpha=0.7, label="Vegard")

    if not pd.isna(row.get("V_Y3X_DFT", np.nan)):
        ax.plot(0.25, row["V_Y3X_DFT"], "o", ms=7, color="blue",
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"L1$_2$ {elY}$_3${elX}", zorder=5)
    if not pd.isna(row.get("V_B2_DFT", np.nan)):
        ax.plot(0.50, row["V_B2_DFT"], "s", ms=7, color="red",
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"B2 {elX}{elY}", zorder=5)
    if not pd.isna(row.get("V_X3Y_DFT", np.nan)):
        ax.plot(0.75, row["V_X3Y_DFT"], "^", ms=7, color="green",
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"L1$_2$ {elX}$_3${elY}", zorder=5)

    ax.plot(0, vY, "D", ms=5, color="gray", markeredgecolor="black",
            markeredgewidth=0.5, zorder=5)
    ax.plot(1, vX, "D", ms=5, color="gray", markeredgecolor="black",
            markeredgewidth=0.5, zorder=5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_title(f"{elX}–{elY}", fontsize=11, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel(f"$c_{{{elX}}}$")
    if show_ylabel:
        ax.set_ylabel(r"$V$ [Å$^3$/atom]")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0\n" + elY, "0.25", "0.5", "0.75", "1\n" + elX], fontsize=8)

    annotations = []
    if not pd.isna(row.get("Omega_Y3X", np.nan)):
        annotations.append(f"$\\Omega_{{Y_3X}}$={row['Omega_Y3X']:.3f}")
    if not pd.isna(row.get("Omega_B2", np.nan)):
        annotations.append(f"$\\Omega_{{B2}}$={row['Omega_B2']:.3f}")
    if not pd.isna(row.get("Omega_X3Y", np.nan)):
        annotations.append(f"$\\Omega_{{X_3Y}}$={row['Omega_X3Y']:.3f}")
    if annotations:
        ax.text(0.02, 0.98, "\n".join(annotations), transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8))


def fig_to_base64(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def generate_html(table, subset="all"):
    if subset == "hea":
        df = table[table["elX"].isin(HEA_ELEMENTS) & table["elY"].isin(HEA_ELEMENTS)].copy()
        title_label = "HEA元素"
    else:
        df = table.copy()
        title_label = "全元素"

    df = df.sort_values(["elX", "elY"]).reset_index(drop=True)
    n_total = len(df)
    ncols = 6
    nrows = 5
    per_page = ncols * nrows
    n_pages = (n_total + per_page - 1) // per_page

    print(f"  Generating HTML: {n_total} pairs, {n_pages} pages ({subset})...")

    # Build HTML
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Vegard則 vs DFT: 組成軸プロット ({title_label}, {n_total}ペア)</title>
<style>
  body {{
    font-family: "Hiragino Kaku Gothic Pro", "Yu Gothic", "Meiryo", sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
    background: #fafaf8;
    color: #222;
  }}
  h1 {{ font-size: 26px; color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 10px; }}
  h2 {{ font-size: 20px; color: #2c5f8a; margin-top: 30px; }}
  .page-img {{ width: 100%; max-width: 1400px; margin: 10px 0 30px 0; border: 1px solid #ccc; }}
  .nav {{ background: #eef4fa; padding: 10px 15px; border-radius: 6px; margin: 10px 0; }}
  .nav a {{ margin: 0 5px; text-decoration: none; color: #2c5f8a; }}
  .nav a:hover {{ text-decoration: underline; }}
  .legend-box {{
    background: #fff; border: 1px solid #ccc; border-radius: 8px;
    padding: 15px; margin: 15px 0; display: inline-block;
  }}
  .legend-box span {{ margin-right: 20px; }}
  .summary {{ background: #eef4fa; padding: 15px; border-radius: 8px; margin: 15px 0; }}
  table {{ border-collapse: collapse; margin: 10px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
  th {{ background: #e8eef4; }}
</style>
</head>
<body>

<h1>Vegard則 vs DFT: 組成軸プロット ({title_label})</h1>

<div class="summary">
<strong>データ:</strong> {n_total}元素ペア / 全3構造 (L1$_2$ X$_3$Y, B2 XY, L1$_2$ Y$_3$X)<br>
<strong>横軸:</strong> $c_X$ (元素Xの組成分率 0→1)<br>
<strong>縦軸:</strong> 原子あたり体積 $V$ [Å$^3$/atom]
</div>

<div class="legend-box">
  <span>&#9644;&#9644; Vegard則 (直線補間)</span>
  <span style="color:blue;">&#9679; L1$_2$ Y$_3$X ($c_X$=0.25)</span>
  <span style="color:red;">&#9632; B2 XY ($c_X$=0.50)</span>
  <span style="color:green;">&#9650; L1$_2$ X$_3$Y ($c_X$=0.75)</span>
  <span style="color:gray;">&#9670; 純元素</span>
</div>

<div class="nav">
<strong>ページナビ:</strong>
""")

    for p in range(n_pages):
        s = p * per_page + 1
        e = min((p + 1) * per_page, n_total)
        html_parts.append(f'<a href="#page{p+1}">P{p+1} ({s}-{e})</a>')

    html_parts.append("</div>\n")

    # Generate each page
    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, n_total)
        n_this = end - start

        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 18))

        for idx in range(per_page):
            r = idx // ncols
            c_idx = idx % ncols
            ax = axes[r, c_idx]
            if idx < n_this:
                row = df.iloc[start + idx]
                plot_vegard_composition(ax, row,
                                       show_ylabel=(c_idx == 0),
                                       show_xlabel=(r == nrows - 1) or (start + idx >= n_total - ncols))
            else:
                ax.set_visible(False)

        fig.suptitle(
            f"Vegard則 vs DFT ({title_label}) — Page {page+1}/{n_pages} "
            f"(pairs {start+1}–{end}/{n_total})",
            fontsize=16, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        b64 = fig_to_base64(fig, dpi=120)
        plt.close(fig)

        # Element pairs in this page
        pairs_str = ", ".join(f"{df.iloc[start+i]['elX']}–{df.iloc[start+i]['elY']}"
                              for i in range(n_this))

        html_parts.append(f"""
<h2 id="page{page+1}">Page {page+1}/{n_pages} (pairs {start+1}–{end})</h2>
<p style="font-size:12px; color:#666;">{pairs_str}</p>
<img class="page-img" src="data:image/png;base64,{b64}" alt="Page {page+1}">
""")

        if (page + 1) % 5 == 0:
            print(f"    page {page+1}/{n_pages}")

    html_parts.append("""
</body>
</html>
""")

    html_path = OUT / f"vegard_composition_{subset}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    fsize = html_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {html_path} ({fsize:.1f} MB)")
    return html_path


def main():
    print("=" * 60)
    print("Vegard組成軸プロット HTML生成")
    print("=" * 60)

    table = pd.read_csv(OUT / "comparison_table.csv")
    has_all3 = table.dropna(subset=["a_X3Y", "a_Y3X", "a_B2"])
    print(f"Pairs with all 3 structures: {len(has_all3)}")

    # HEA elements
    path_hea = generate_html(has_all3, subset="hea")

    # All elements
    path_all = generate_html(has_all3, subset="all")

    print(f"\n完了:")
    print(f"  HEA: {path_hea}")
    print(f"  All: {path_all}")


if __name__ == "__main__":
    main()
