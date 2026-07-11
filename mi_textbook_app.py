"""
マテリアルズ・インフォマティクス (MI) 講義用アプリケーション
============================================================
対象: 大学3回生 材料工学専攻 初心者向け
想定: 講義1コマで一連のMIワークフローを体験

セクション構成:
  1. MIとは
  2. データ探索
  3. 次元削減 PCA (BiPlot)
  4. 回帰問題
  5. 分類問題（Hume-Rothery則）
  6. 交差検証・汎化性能評価
  7. 正則化・モデル選択
  8. データ増強
  9. 特徴量生成
 10. まとめ + レポート課題

データセット（全て実データ）:
  - 鉄鋼: matminer steel_strength (Citrine Informatics, 312件)
  - 超伝導体: UCI/NIMS SuperCon (21,263件 → 500件抽出)
  - HEA相分類: Zenodo ACHIEF project (1,103件)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as font_manager
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict, KFold, LeaveOneOut,
    learning_curve, validation_curve, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, silhouette_score,
    accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import warnings
import inspect

warnings.filterwarnings("ignore")

# 学生PCでの安定動作を優先し、並列処理は1プロセスに固定する
# （Windows/macOSの環境差によるフリーズやメモリ不足を避けるため）
N_JOBS = 1
KMEANS_N_INIT = 3
SILHOUETTE_SAMPLE_SIZE = 500

# 講義室のスクリーン投影でも見やすいよう、散布図などの点は全体的に大きめにする。
PLOT_POINT_SIZE = 11

# ---------------------------------------------------------------------------
# ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="マテリアルズ・インフォマティクス応用（MI）",
    page_icon="🔬",
    layout="wide",
)

# Matplotlib 日本語フォント設定
try:
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
    plt.rcParams["font.family"] = "IPAGothic"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

# Plotly 日本語フォント設定
_JP_FONT = "Yu Gothic, YuGothic, Meiryo, Hiragino Sans, Hiragino Kaku Gothic ProN, Noto Sans JP, sans-serif"
_plotly_template = pio.templates["plotly"]
_plotly_template.layout.font = dict(family=_JP_FONT)
pio.templates.default = "plotly"

st.markdown(f"""<style>
.js-plotly-plot text, .js-plotly-plot .gtitle, .js-plotly-plot .xtitle,
.js-plotly-plot .ytitle, .js-plotly-plot .legendtext {{
    font-family: {_JP_FONT} !important;
}}
</style>""", unsafe_allow_html=True)


def _enlarge_plot_markers(fig, min_size=PLOT_POINT_SIZE):
    """Plotly 図中の点を、教材用に見やすい最小サイズへそろえる。"""
    try:
        for trace in getattr(fig, "data", []):
            trace_type = getattr(trace, "type", "")
            mode = getattr(trace, "mode", "") or ""
            if trace_type == "splom":
                marker = getattr(trace, "marker", None)
            elif trace_type in {"scatter", "scattergl"} and "markers" in mode:
                marker = getattr(trace, "marker", None)
            else:
                continue

            if marker is None:
                continue
            current_size = getattr(marker, "size", None)
            if current_size is None:
                marker.size = min_size
            elif isinstance(current_size, (int, float, np.integer, np.floating)):
                marker.size = max(float(current_size), min_size)
    except Exception:
        pass
    return fig


def _plotly_chart(fig):
    """Streamlit の版差を吸収して Plotly 図を横幅いっぱいに表示する。"""
    fig = _enlarge_plot_markers(fig)
    try:
        return st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        return st.plotly_chart(fig)


def _plotly_chart_fixed(fig):
    """正方形プロットなど、図の縦横比を保ちたい場合に使う。"""
    fig = _enlarge_plot_markers(fig)
    try:
        return st.plotly_chart(fig, use_container_width=False)
    except TypeError:
        return st.plotly_chart(fig)


def _create_pairplot_with_upper_corr(df, columns, title, max_points=1200):
    """下三角に散布図、対角にヒストグラム、上三角に相関係数を置くペアプロットを作る。"""
    numeric_columns = [c for c in columns if c in df.columns and is_numeric_dtype(df[c])]
    if len(numeric_columns) < 2:
        return None, 0, 0

    data_all = df[numeric_columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data_all) < 2:
        return None, len(data_all), 0

    if len(data_all) > max_points:
        data_plot = data_all.sample(max_points, random_state=42)
    else:
        data_plot = data_all

    n = len(numeric_columns)
    fig_size = max(780, min(1360, 320 * n))
    fig = make_subplots(
        rows=n,
        cols=n,
        horizontal_spacing=0.018,
        vertical_spacing=0.018,
    )

    for i, y_col in enumerate(numeric_columns):
        for j, x_col in enumerate(numeric_columns):
            row = i + 1
            col = j + 1

            if i == j:
                fig.add_trace(
                    go.Histogram(
                        x=data_plot[x_col],
                        nbinsx=24,
                        showlegend=False,
                        hovertemplate=f"{x_col}<br>値=%{{x}}<br>度数=%{{y}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            elif i > j:
                fig.add_trace(
                    go.Scattergl(
                        x=data_plot[x_col],
                        y=data_plot[y_col],
                        mode="markers",
                        marker=dict(size=PLOT_POINT_SIZE, opacity=0.60),
                        showlegend=False,
                        hovertemplate=(
                            f"{x_col}=%{{x:.3g}}<br>"
                            f"{y_col}=%{{y:.3g}}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            else:
                pair = data_all[[x_col, y_col]].dropna()
                r = pair[x_col].corr(pair[y_col]) if len(pair) >= 2 else np.nan
                corr_text = "r = --" if not np.isfinite(r) else f"r = {r:.2f}"
                fig.add_trace(
                    go.Scatter(
                        x=[0.5],
                        y=[0.5],
                        mode="text",
                        text=[corr_text],
                        textfont=dict(size=24),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                fig.update_xaxes(range=[0, 1], visible=False, row=row, col=col)
                fig.update_yaxes(range=[0, 1], visible=False, row=row, col=col)

            if i != n - 1:
                fig.update_xaxes(showticklabels=False, title_text="", row=row, col=col)
            else:
                fig.update_xaxes(
                    title_text=x_col,
                    title_font=dict(size=11),
                    tickfont=dict(size=9),
                    row=row,
                    col=col,
                )

            if j != 0:
                fig.update_yaxes(showticklabels=False, title_text="", row=row, col=col)
            else:
                fig.update_yaxes(
                    title_text=y_col,
                    title_font=dict(size=11),
                    tickfont=dict(size=9),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title=title,
        height=fig_size,
        width=fig_size,
        autosize=False,
        showlegend=False,
        bargap=0.05,
        margin=dict(l=80, r=40, t=80, b=80),
    )
    return fig, len(data_all), len(data_plot)


def _add_centroid_markers(fig, x, y, name="クラスタ重心"):
    """k-meansの重心がデータ点に埋もれないよう、白抜き背景 + 黒い×で強調する。"""
    fig.add_scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=38, color="white", symbol="circle", line=dict(width=3, color="black")),
        name=f"{name}（背景）", showlegend=False,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=34, color="black", symbol="x"),
        name=name, showlegend=True,
    )
    return fig


def _csv_bytes(df):
    """Excelでも文字化けしにくいUTF-8 BOM付きCSVを返す。"""
    return df.to_csv(index=False).encode("utf-8-sig")


def _safe_rerun():
    """Streamlit の版差を吸収して再実行する。"""
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # 古い Streamlit 向けフォールバック
        st.experimental_rerun()


def _notify(message):
    """トースト表示に対応していれば使い、なければ通常のメッセージにする。"""
    if hasattr(st, "toast"):
        st.toast(message)
    else:
        st.info(message)


# 軽量なMAGPIE風特徴量生成に使う最小限の元素表。
# matminerが使えない授業PCでも、特徴量生成の考え方を実演できるようにする。
_ELEMENT_PROPS = {
    "Ag": {"Z": 47, "mass": 107.8682, "radius": 144, "en": 1.93, "tm": 1234.9, "group": 11},
    "Al": {"Z": 13, "mass": 26.9815, "radius": 143, "en": 1.61, "tm": 933.5, "group": 13},
    "Au": {"Z": 79, "mass": 196.9666, "radius": 144, "en": 2.54, "tm": 1337.3, "group": 11},
    "B": {"Z": 5, "mass": 10.81, "radius": 87, "en": 2.04, "tm": 2349, "group": 13},
    "Be": {"Z": 4, "mass": 9.0122, "radius": 112, "en": 1.57, "tm": 1560, "group": 2},
    "Bi": {"Z": 83, "mass": 208.9804, "radius": 160, "en": 2.02, "tm": 544.7, "group": 15},
    "C": {"Z": 6, "mass": 12.011, "radius": 67, "en": 2.55, "tm": 3823, "group": 14},
    "Ca": {"Z": 20, "mass": 40.078, "radius": 197, "en": 1.00, "tm": 1115, "group": 2},
    "Cd": {"Z": 48, "mass": 112.414, "radius": 151, "en": 1.69, "tm": 594.2, "group": 12},
    "Ce": {"Z": 58, "mass": 140.116, "radius": 181, "en": 1.12, "tm": 1068, "group": 3},
    "Co": {"Z": 27, "mass": 58.9332, "radius": 125, "en": 1.88, "tm": 1768, "group": 9},
    "Cr": {"Z": 24, "mass": 51.9961, "radius": 128, "en": 1.66, "tm": 2180, "group": 6},
    "Cu": {"Z": 29, "mass": 63.546, "radius": 128, "en": 1.90, "tm": 1357.8, "group": 11},
    "Dy": {"Z": 66, "mass": 162.5, "radius": 178, "en": 1.22, "tm": 1680, "group": 3},
    "Er": {"Z": 68, "mass": 167.259, "radius": 176, "en": 1.24, "tm": 1802, "group": 3},
    "Fe": {"Z": 26, "mass": 55.845, "radius": 126, "en": 1.83, "tm": 1811, "group": 8},
    "Ga": {"Z": 31, "mass": 69.723, "radius": 135, "en": 1.81, "tm": 302.9, "group": 13},
    "Gd": {"Z": 64, "mass": 157.25, "radius": 180, "en": 1.20, "tm": 1585, "group": 3},
    "Ge": {"Z": 32, "mass": 72.63, "radius": 125, "en": 2.01, "tm": 1211.4, "group": 14},
    "Hf": {"Z": 72, "mass": 178.49, "radius": 159, "en": 1.30, "tm": 2506, "group": 4},
    "Ho": {"Z": 67, "mass": 164.9303, "radius": 176, "en": 1.23, "tm": 1734, "group": 3},
    "In": {"Z": 49, "mass": 114.818, "radius": 167, "en": 1.78, "tm": 429.7, "group": 13},
    "La": {"Z": 57, "mass": 138.9055, "radius": 187, "en": 1.10, "tm": 1193, "group": 3},
    "Li": {"Z": 3, "mass": 6.94, "radius": 152, "en": 0.98, "tm": 453.7, "group": 1},
    "Lu": {"Z": 71, "mass": 174.967, "radius": 174, "en": 1.27, "tm": 1925, "group": 3},
    "Mg": {"Z": 12, "mass": 24.305, "radius": 160, "en": 1.31, "tm": 923, "group": 2},
    "Mn": {"Z": 25, "mass": 54.9380, "radius": 127, "en": 1.55, "tm": 1519, "group": 7},
    "Mo": {"Z": 42, "mass": 95.95, "radius": 139, "en": 2.16, "tm": 2896, "group": 6},
    "N": {"Z": 7, "mass": 14.007, "radius": 56, "en": 3.04, "tm": 63.2, "group": 15},
    "Na": {"Z": 11, "mass": 22.9898, "radius": 186, "en": 0.93, "tm": 371, "group": 1},
    "Nb": {"Z": 41, "mass": 92.9064, "radius": 146, "en": 1.60, "tm": 2750, "group": 5},
    "Nd": {"Z": 60, "mass": 144.242, "radius": 182, "en": 1.14, "tm": 1297, "group": 3},
    "Ni": {"Z": 28, "mass": 58.6934, "radius": 124, "en": 1.91, "tm": 1728, "group": 10},
    "P": {"Z": 15, "mass": 30.9738, "radius": 98, "en": 2.19, "tm": 317.3, "group": 15},
    "Pb": {"Z": 82, "mass": 207.2, "radius": 175, "en": 2.33, "tm": 600.6, "group": 14},
    "Pd": {"Z": 46, "mass": 106.42, "radius": 137, "en": 2.20, "tm": 1828, "group": 10},
    "Pr": {"Z": 59, "mass": 140.9077, "radius": 182, "en": 1.13, "tm": 1208, "group": 3},
    "Pt": {"Z": 78, "mass": 195.084, "radius": 139, "en": 2.28, "tm": 2041.4, "group": 10},
    "Re": {"Z": 75, "mass": 186.207, "radius": 137, "en": 1.90, "tm": 3459, "group": 7},
    "Rh": {"Z": 45, "mass": 102.9055, "radius": 134, "en": 2.28, "tm": 2237, "group": 9},
    "Ru": {"Z": 44, "mass": 101.07, "radius": 134, "en": 2.20, "tm": 2607, "group": 8},
    "Sb": {"Z": 51, "mass": 121.760, "radius": 145, "en": 2.05, "tm": 903.8, "group": 15},
    "Sc": {"Z": 21, "mass": 44.9559, "radius": 162, "en": 1.36, "tm": 1814, "group": 3},
    "Si": {"Z": 14, "mass": 28.085, "radius": 111, "en": 1.90, "tm": 1687, "group": 14},
    "Sm": {"Z": 62, "mass": 150.36, "radius": 180, "en": 1.17, "tm": 1345, "group": 3},
    "Sn": {"Z": 50, "mass": 118.710, "radius": 145, "en": 1.96, "tm": 505.1, "group": 14},
    "Sr": {"Z": 38, "mass": 87.62, "radius": 215, "en": 0.95, "tm": 1050, "group": 2},
    "Ta": {"Z": 73, "mass": 180.9479, "radius": 146, "en": 1.50, "tm": 3290, "group": 5},
    "Tb": {"Z": 65, "mass": 158.9254, "radius": 177, "en": 1.20, "tm": 1629, "group": 3},
    "Ti": {"Z": 22, "mass": 47.867, "radius": 147, "en": 1.54, "tm": 1941, "group": 4},
    "Tm": {"Z": 69, "mass": 168.9342, "radius": 176, "en": 1.25, "tm": 1818, "group": 3},
    "V": {"Z": 23, "mass": 50.9415, "radius": 134, "en": 1.63, "tm": 2183, "group": 5},
    "W": {"Z": 74, "mass": 183.84, "radius": 139, "en": 2.36, "tm": 3695, "group": 6},
    "Y": {"Z": 39, "mass": 88.9058, "radius": 180, "en": 1.22, "tm": 1799, "group": 3},
    "Yb": {"Z": 70, "mass": 173.045, "radius": 176, "en": 1.10, "tm": 1097, "group": 3},
    "Zn": {"Z": 30, "mass": 65.38, "radius": 134, "en": 1.65, "tm": 692.7, "group": 12},
    "Zr": {"Z": 40, "mass": 91.224, "radius": 160, "en": 1.33, "tm": 2128, "group": 4},
    # HEA以外（酸化物・水素化物など）の一般的な化学式にも対応できるよう、
    # 代表的な軽元素・非金属を追加する。radiusはClementi計算原子半径(pm)。
    "H": {"Z": 1, "mass": 1.008, "radius": 53, "en": 2.20, "tm": 14.01, "group": 1},
    "O": {"Z": 8, "mass": 15.999, "radius": 48, "en": 3.44, "tm": 54.36, "group": 16},
    "F": {"Z": 9, "mass": 18.998, "radius": 42, "en": 3.98, "tm": 53.53, "group": 17},
    "S": {"Z": 16, "mass": 32.06, "radius": 88, "en": 2.58, "tm": 388.36, "group": 16},
    "Cl": {"Z": 17, "mass": 35.45, "radius": 79, "en": 3.16, "tm": 171.6, "group": 17},
    "K": {"Z": 19, "mass": 39.098, "radius": 243, "en": 0.82, "tm": 336.7, "group": 1},
    "Br": {"Z": 35, "mass": 79.904, "radius": 94, "en": 2.96, "tm": 265.8, "group": 17},
    "I": {"Z": 53, "mass": 126.904, "radius": 115, "en": 2.66, "tm": 386.85, "group": 17},
}


def _normalize_formula_text(formula):
    """HEAデータ中の空白や Al のOCR風表記ゆれを少しだけ直す。"""
    import re
    text = str(formula).strip().replace(" ", "")
    # 例: A13 Fe36... は Al3 Fe36... の表記ゆれとみなす。
    text = re.sub(r"A1(?=\d)", "Al", text)
    return text


def _parse_formula_counts(formula):
    """Fe2Ni のような簡単な化学式を元素:量の辞書に変換する。"""
    import re
    text = _normalize_formula_text(formula)
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", text)
    if not tokens:
        raise ValueError("元素記号を読み取れない")
    counts = {}
    for elem, num in tokens:
        if elem not in _ELEMENT_PROPS:
            raise ValueError(f"未対応元素: {elem}")
        amount = float(num) if num else 1.0
        counts[elem] = counts.get(elem, 0.0) + amount
    if sum(counts.values()) <= 0:
        raise ValueError("組成量が0である")
    return counts


def _weighted_stats(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    wmean = float(np.sum(values * weights))
    return {
        "mean": wmean,
        "range": float(np.max(values) - np.min(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "avg_dev": float(np.sum(weights * np.abs(values - wmean))),
        "std": float(np.sqrt(np.sum(weights * (values - wmean) ** 2))),
    }


def _fallback_magpie_features(formulas):
    """matminerなしで動く、少数のMAGPIE風組成特徴量を作る。"""
    rows = []
    skipped = []
    for formula in formulas:
        try:
            counts = _parse_formula_counts(formula)
            total = sum(counts.values())
            elems = list(counts)
            weights = np.array([counts[e] / total for e in elems], dtype=float)
            props = {name: np.array([_ELEMENT_PROPS[e][name] for e in elems], dtype=float)
                     for name in ["Z", "mass", "radius", "en", "tm", "group"]}
            r_bar = float(np.sum(props["radius"] * weights))
            delta_radius = float(np.sqrt(np.sum(weights * (1 - props["radius"] / r_bar) ** 2)) * 100) if r_bar else np.nan
            z = _weighted_stats(props["Z"], weights)
            mass = _weighted_stats(props["mass"], weights)
            radius = _weighted_stats(props["radius"], weights)
            en = _weighted_stats(props["en"], weights)
            tm = _weighted_stats(props["tm"], weights)
            group = _weighted_stats(props["group"], weights)
            rows.append({
                "formula": formula,
                "n_elements": len(elems),
                "mean_atomic_number": z["mean"],
                "range_atomic_number": z["range"],
                "avg_dev_atomic_number": z["avg_dev"],
                "mean_atomic_mass": mass["mean"],
                "range_atomic_mass": mass["range"],
                "avg_dev_atomic_mass": mass["avg_dev"],
                "std_atomic_mass": mass["std"],
                "mean_atomic_radius": radius["mean"],
                "range_atomic_radius": radius["range"],
                "avg_dev_atomic_radius": radius["avg_dev"],
                "r_delta_percent": delta_radius,
                "mean_electronegativity": en["mean"],
                "range_electronegativity": en["range"],
                "avg_dev_electronegativity": en["avg_dev"],
                "mean_melting_point": tm["mean"],
                "range_melting_point": tm["range"],
                "avg_dev_melting_point": tm["avg_dev"],
                "mean_group": group["mean"],
                "range_group": group["range"],
            })
        except Exception as exc:
            skipped.append({"formula": formula, "reason": str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def _matminer_magpie_features(formulas, max_features=12):
    """matminerが入っている環境ではMAGPIE特徴量を少数だけ返す。"""
    try:
        from pymatgen.core import Composition
        from matminer.featurizers.composition import ElementProperty
    except Exception as exc:
        return None, (
            f"matminer/pymatgen が見つからないため、内蔵の軽量MAGPIE風特徴量に切り替える。詳細: {exc}"
        )

    try:
        featurizer = ElementProperty.from_preset("magpie")
        labels = featurizer.feature_labels()
        preferred = [
            "MagpieData mean Number",
            "MagpieData mean AtomicWeight",
            "MagpieData mean MendeleevNumber",
            "MagpieData mean MeltingT",
            "MagpieData mean Column",
            "MagpieData mean Row",
            "MagpieData mean CovalentRadius",
            "MagpieData mean Electronegativity",
            "MagpieData range Number",
            "MagpieData range AtomicWeight",
            "MagpieData range MeltingT",
            "MagpieData range Electronegativity",
            "MagpieData avg_dev AtomicWeight",
            "MagpieData avg_dev CovalentRadius",
            "MagpieData avg_dev Electronegativity",
        ]
        keep = [labels.index(x) for x in preferred if x in labels][:max_features]
        if not keep:
            keep = list(range(min(max_features, len(labels))))
        rows = []
        skipped = []
        for formula in formulas:
            try:
                comp = Composition(_normalize_formula_text(formula))
                vals = featurizer.featurize(comp)
                row = {"formula": formula}
                row.update({labels[i]: vals[i] for i in keep})
                rows.append(row)
            except Exception as exc:
                skipped.append({"formula": formula, "reason": str(exc)})
        return (pd.DataFrame(rows), pd.DataFrame(skipped)), None
    except Exception as exc:
        return None, f"MAGPIE特徴量生成に失敗した: {exc}"


# ---------------------------------------------------------------------------
# データ読み込み（実データ）
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"


def _read_csv_checked(filename):
    """配布ZIPの展開漏れなどを学生が気づきやすいメッセージにする。"""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"データファイルが見つかりません: {path}\n"
            "mi_textbook_app.py と data フォルダを同じフォルダ内に置くこと。"
        )
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _cached_rf_tuning(X_train, y_train, mode):
    """講義中の再実行を軽くするため、RFチューニング結果をキャッシュする。"""
    if mode == "軽量（講義向け）":
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [6, None],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt"],
        }
    else:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 8, 12, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.7, 1.0],
        }
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=N_JOBS),
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=N_JOBS,
    )
    rf_search.fit(X_train, y_train)
    return rf_search.best_params_, float(rf_search.best_score_)


@st.cache_data(show_spinner=False)
def _cached_regularization_paths(X_train, y_train, alphas):
    """Ridge/Lassoの正則化パスをまとめて計算する。

    パス曲線は X_train・y_train・alphas だけで決まり、7.2 のαスライダーには依存しない。
    ここでキャッシュしておくことで、スライダー操作時の再描画を軽くする。
    """
    ridge_coefs = []
    lasso_coefs = []
    for a in alphas:
        ridge_coefs.append(Ridge(alpha=a).fit(X_train, y_train).coef_)
        lasso_coefs.append(Lasso(alpha=a, max_iter=5000, tol=1e-3).fit(X_train, y_train).coef_)
    return np.array(ridge_coefs), np.array(lasso_coefs)


def _numeric_feature_columns(df, target, excluded=()):
    """目的変数・リークしやすい応答変数を除いた数値特徴量だけを返す。"""
    excluded = set(excluded)
    return [
        c for c in df.columns
        if c != target and c not in excluded and is_numeric_dtype(df[c])
    ]


@st.cache_data
def load_steel_data():
    """鉄鋼の機械的特性 — Citrine Informatics 実験データ (312件)
    出典: matminer steel_strength / Matbench"""
    df = _read_csv_checked("steel_strength.csv")
    return df


@st.cache_data
def load_superconductor_data():
    """超伝導体の臨界温度 — UCI/NIMS SuperCon (500件抽出)
    出典: NIMS supercon.nims.go.jp, UCI ML Repository"""
    df = _read_csv_checked("superconductor_500.csv")
    return df


@st.cache_data
def load_hea_data():
    """高エントロピー合金の相分類 — Zenodo ACHIEF (1,103件)
    出典: doi:10.5281/zenodo.5155150"""
    df = _read_csv_checked("HEA_phases.csv")
    # 旧版CSVの delta 列にも対応する。アプリ内では原子半径差であることが分かるよう r_delta と呼ぶ。
    if "delta" in df.columns and "r_delta" not in df.columns:
        df = df.rename(columns={"delta": "r_delta"})
    return df


# ---------------------------------------------------------------------------
# 外部CSVの読み込み（ユーザー自身のデータ）
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _parse_uploaded_csv(file_bytes, filename):
    """アップロードされたCSVを、日本語Excel由来の文字コードも考慮して読み込む。

    file_bytes をキャッシュキーにするため、同じファイルの再解析は行わない。
    """
    import io

    last_err = None
    # まずは一般的な区切り（カンマ）で、複数の文字コードを順に試す。
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception as exc:
            last_err = exc
    # 区切り文字が不明な場合（タブ区切り等）は自動判定も試す。
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep=None, engine="python")
        except Exception as exc:
            last_err = exc
    raise last_err


def _external_csv_config():
    """サイドバーで外部CSVを読み込み、目的変数・説明変数を選ばせる。

    戻り値は組み込みデータと同じ形:
        (df_main, target_col, feature_cols, dataset_desc, dataset_detail)

    ファイル未選択や設定不足の場合は案内を表示して st.stop() する。
    """
    st.sidebar.markdown("**外部CSVの設定**")
    uploaded = st.sidebar.file_uploader(
        "CSVファイルを選択",
        type=["csv", "tsv", "txt"],
        key="external_csv_uploader",
        help="1行目を列名とするCSV。数値列を目的変数・説明変数として選べる。",
    )

    if uploaded is None:
        st.title("📤 外部CSVの読み込み")
        st.info(
            "左のサイドバーからCSVファイルをアップロードすると、"
            "自分のデータで各セクション（データ探索・PCA・回帰など）を実行できる。"
        )
        st.markdown("""
        **CSVの条件**
        - 1行目を列名とする
        - 目的変数・説明変数には**数値列**を選ぶ（文字列列は特徴量には使われない）
        - 文字コードは UTF-8 / Shift-JIS(cp932) に対応
        - 数値列が **2つ以上** 必要（説明変数も2つ以上選ぶこと）
        """)
        st.stop()

    try:
        df_ext = _parse_uploaded_csv(uploaded.getvalue(), uploaded.name)
    except Exception as exc:
        st.title("📤 外部CSVの読み込み")
        st.error(f"CSVの読み込みに失敗した: {exc}")
        st.stop()

    df_ext.columns = [str(c) for c in df_ext.columns]
    numeric_cols = [c for c in df_ext.columns if is_numeric_dtype(df_ext[c])]

    if len(numeric_cols) < 2:
        st.title("📤 外部CSVの読み込み")
        st.error(
            "数値列が2つ以上必要である。"
            f"読み込めた数値列: {numeric_cols if numeric_cols else 'なし'}"
        )
        st.caption("読み込んだデータの先頭:")
        st.dataframe(df_ext.head(20), use_container_width=True)
        st.stop()

    st.sidebar.success(
        f"{uploaded.name}\n{len(df_ext)}行 × {len(df_ext.columns)}列"
    )

    # 新しいファイルを読み込んだ直後だけ「データ探索」へ自動移動する。
    # （目的変数・説明変数を変えただけの再実行では移動しない。）
    if st.session_state.get("_external_last_filename") != uploaded.name:
        st.session_state["_external_last_filename"] = uploaded.name
        st.session_state["_pending_section_label"] = "2. データ探索"
        st.session_state["_show_jump_message"] = True
        _safe_rerun()

    # 目的変数: 既定は最後の数値列（多くのデータセットで目的変数は末尾にある）。
    target = st.sidebar.selectbox(
        "目的変数（予測したい数値列）",
        numeric_cols,
        index=len(numeric_cols) - 1,
        key="external_target_col",
    )

    feature_candidates = [c for c in numeric_cols if c != target]
    features = st.sidebar.multiselect(
        "説明変数（入力に使う数値列）",
        feature_candidates,
        default=feature_candidates,
        key="external_feature_cols",
    )

    if len(features) < 2:
        st.title("📤 外部CSVの読み込み")
        st.warning(
            "説明変数を2つ以上選ぶこと（PCAや相関行列には2列以上必要である）。"
        )
        st.stop()

    # 目的変数・説明変数の欠損行は解析で使えないため、事前に除外しておく。
    n_before = len(df_ext)
    df_ext = df_ext.dropna(subset=[target] + features).reset_index(drop=True)
    n_dropped = n_before - len(df_ext)

    if len(df_ext) < 10:
        st.title("📤 外部CSVの読み込み")
        st.error(
            f"欠損を除外した後の有効データが {len(df_ext)} 件しかない。"
            "回帰・交差検証には最低でも10件程度が必要である。列の選択を見直すこと。"
        )
        st.stop()

    drop_note = f"（欠損を含む {n_dropped} 行を除外）" if n_dropped > 0 else ""
    desc = f"外部CSV「{uploaded.name}」から {target} を予測する{drop_note}"
    detail = f"""
**出典**: ユーザーがアップロードした外部CSV（{uploaded.name}）

- データ数: **{len(df_ext)}** 件{drop_note}
- 目的変数: **{target}**
- 説明変数（{len(features)}個）: {", ".join(features)}

数値列のみを目的変数・説明変数として選択できる。
文字列列（組成式・材料名など）はそのままでは特徴量に使われない。
目的変数・説明変数の選択は、サイドバーからいつでも変更できる。
"""
    return df_ext, target, features, desc, detail


# ---------------------------------------------------------------------------
# サイドバー: セクション選択
# ---------------------------------------------------------------------------
st.sidebar.title("📚 マテリアルズ・インフォマティクス応用")
st.sidebar.markdown("---")
st.sidebar.markdown("**講義の構成**")

SECTIONS = {
    "1. MIとは": "mi_intro",
    "2. データ探索": "data_exploration",
    "3. 次元削減 PCA": "pca",
    "4. 回帰問題": "regression",
    "5. 分類問題": "classification",
    "6. 交差検証・汎化性能": "cv_generalization",
    "7. 正則化・モデル選択": "regularization",
    "8. データ増強": "data_augmentation",
    "9. 特徴量生成": "feature_generation",
    "10. まとめ＋レポート課題": "summary_assignments",
}

# 外部CSVの読み込み時などに、指定セクションへ自動移動するための保留フラグ。
# ラジオ生成「前」に反映する（生成後に session_state を書き換えると Streamlit がエラーを出すため）。
_pending_section_label = st.session_state.pop("_pending_section_label", None)
if _pending_section_label in SECTIONS:
    st.session_state["section_radio"] = _pending_section_label

selected = st.sidebar.radio("セクションを選択", list(SECTIONS.keys()), key="section_radio")
section_key = SECTIONS[selected]

st.sidebar.markdown("---")
st.sidebar.markdown("**使用データセット**")
dataset_choice = st.sidebar.selectbox(
    "回帰用データ",
    ["鉄鋼（構造材料）", "超伝導体（機能材料）", "外部CSV（自分のデータ）"],
)

# データ読み込み
if dataset_choice == "鉄鋼（構造材料）":
    df_main = load_steel_data()
    target_col = "yield strength"
    # 鉄鋼データの tensile strength / elongation は同時測定された応答変数であり、
    # 入力特徴量として使うと「答えを少し見ている」データリークになりやすいため除外する。
    _response_cols = {"tensile strength", "elongation"}
    dataset_desc = "鉄鋼合金の組成から降伏強度を予測（Citrine Informatics 実験データ）"
    dataset_detail = """
**出典**: Citrine Informatics / Matbench (312件の実験データ)

| 特徴量 | 説明 | 単位 |
|:---|:---|:---|
| c | 炭素含有量 — 強度に最も影響する元素 | wt% |
| mn | マンガン — 固溶強化・靭性向上 | wt% |
| si | ケイ素 — 脱酸・固溶強化 | wt% |
| cr | クロム — 耐食性・焼入性向上 | wt% |
| ni | ニッケル — 靭性・耐食性向上 | wt% |
| mo | モリブデン — 焼入性・高温強度 | wt% |
| v | バナジウム — 析出強化 | wt% |
| n | 窒素 — 固溶強化 | wt% |
| nb | ニオブ — 結晶粒微細化 | wt% |
| co | コバルト — 高温強度 | wt% |
| w | タングステン — 高温強度 | wt% |
| al | アルミニウム — 脱酸 | wt% |
| ti | チタン — 析出強化 | wt% |

**目的変数**: yield strength (MPa) — 降伏強度
"""
    feature_cols = _numeric_feature_columns(df_main, target_col, excluded=_response_cols)

elif dataset_choice == "超伝導体（機能材料）":
    df_main = load_superconductor_data()
    target_col = "critical_temp"
    _response_cols = set()
    dataset_desc = "超伝導材料の元素特徴量から臨界温度を予測（NIMS SuperCon / UCI）"
    dataset_detail = """
**出典**: NIMS SuperCon Database / UCI ML Repository (21,263件から500件抽出)

| 特徴量 | 説明 | 単位 |
|:---|:---|:---|
| number_of_elements | 構成元素数 | — |
| mean_atomic_mass | 平均原子量 | g/mol |
| mean_fie | 平均第一イオン化エネルギー | eV |
| mean_atomic_radius | 平均原子半径 | pm |
| mean_Density | 平均密度 | g/cm³ |
| mean_ElectronAffinity | 平均電子親和力 | eV |
| mean_FusionHeat | 平均融解熱 | kJ/mol |
| mean_ThermalConductivity | 平均熱伝導率 | W/mK |
| mean_Valence | 平均価電子数 | — |

**目的変数**: critical_temp (K) — 超伝導臨界温度

**Kaggle**: https://www.kaggle.com/datasets/munumbutt/superconductor-dataset
"""
    feature_cols = _numeric_feature_columns(df_main, target_col, excluded=_response_cols)

else:  # 外部CSV（自分のデータ）
    df_main, target_col, feature_cols, dataset_desc, dataset_detail = _external_csv_config()
    _response_cols = set()

df_hea = load_hea_data()

# 外部CSV読み込み時の自動移動メッセージ（移動先セクションで一度だけ表示する）。
if st.session_state.pop("_show_jump_message", False):
    _notify("データ探索に移ります")

st.sidebar.caption("回帰用データは鉄鋼・超伝導体・外部CSVから選べる。分類問題ではHEA相分類データを使う。")
with st.sidebar.expander("📥 CSVダウンロード", expanded=False):
    st.download_button(
        "鉄鋼データCSV",
        _csv_bytes(load_steel_data()),
        file_name="steel_strength.csv",
        mime="text/csv",
        key="download_steel_sidebar",
    )
    st.download_button(
        "超伝導体データCSV",
        _csv_bytes(load_superconductor_data()),
        file_name="superconductor_500.csv",
        mime="text/csv",
        key="download_superconductor_sidebar",
    )
    st.download_button(
        "HEA相分類データCSV",
        _csv_bytes(load_hea_data()),
        file_name="HEA_phases.csv",
        mime="text/csv",
        key="download_hea_sidebar",
    )


# ---------------------------------------------------------------------------
# ユーティリティ: データ要約表示
# ---------------------------------------------------------------------------
def show_data_summary(df, features, target, desc, detail):
    """回帰セクション前にデータ要約を表示する。PCA BiplotはPCA章に集約する。"""
    st.subheader("📋 データセットの概要")
    st.markdown(f"**{desc}**")
    with st.expander("データセットの詳細説明", expanded=False):
        st.markdown(detail)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 要約統計量")
        st.dataframe(df[features + [target]].describe().round(3), use_container_width=True)
    with col2:
        st.markdown("#### 目的変数の分布")
        fig_target = px.histogram(df, x=target, nbins=30, marginal="box", title=f"{target} の分布")
        fig_target.update_layout(height=520)
        _plotly_chart(fig_target)
    st.caption("回帰章ではモデル評価に集中するため、Biplotは3章のPCAに集約している。")
    st.markdown("---")


def _create_biplot(scores, pca_model, feature_names, explained_ratio, target_values=None):
    """Biplot: スコア + 負荷量ベクトル"""
    fig = go.Figure()

    # スコアプロット
    if target_values is not None:
        fig.add_scatter(x=scores[:, 0], y=scores[:, 1], mode="markers",
                        marker=dict(size=5, color=target_values, colorscale="Viridis",
                                    showscale=True, colorbar=dict(title="目的変数")),
                        name="データ点")
    else:
        fig.add_scatter(x=scores[:, 0], y=scores[:, 1], mode="markers",
                        marker=dict(size=5, opacity=0.6), name="データ点")

    # 負荷量ベクトル（矢印）
    loadings = pca_model.components_.T
    # スケーリング: スコアの範囲に合わせる
    scale = np.abs(scores).max() * 0.8
    for i, feat in enumerate(feature_names):
        fig.add_annotation(
            x=loadings[i, 0] * scale, y=loadings[i, 1] * scale,
            ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2,
            arrowcolor="red"
        )
        fig.add_annotation(
            x=loadings[i, 0] * scale * 1.12, y=loadings[i, 1] * scale * 1.12,
            text=feat, showarrow=False, font=dict(size=10, color="red")
        )

    fig.update_layout(
        xaxis_title=f"PC1 ({explained_ratio[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({explained_ratio[1]*100:.1f}%)",
        title="Biplot（スコア＋負荷量ベクトル）",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=700, width=850,
    )
    return fig


# =====================================================================
# セクション 1: MIとは
# =====================================================================
if section_key == "mi_intro":
    st.title("🔬 マテリアルズ・インフォマティクス (MI) とは")
    st.markdown("---")

    st.header("1.1 マテリアルズ・インフォマティクスの定義")
    st.markdown(r"""
    **マテリアルズ・インフォマティクス (Materials Informatics; MI)** とは、
    材料科学にデータ科学・機械学習の手法を融合させた学際的分野である。

    従来の材料開発は **「経験と勘」** に頼る試行錯誤が中心であった。
    MIではデータ駆動型のアプローチにより、材料探索・設計・最適化を加速する。

    > **目標**: 実験・計算データから物性値を予測し、新材料の発見を加速する
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("従来の材料開発")
        st.markdown("""
        1. 仮説の立案（経験に基づく）
        2. 実験計画
        3. 合成・測定
        4. 結果の解析
        5. 1に戻る（数ヶ月〜数年）
        """)
    with col2:
        st.subheader("MI による材料開発")
        st.markdown("""
        1. データ収集（実験 + データベース）
        2. 特徴量設計・選択
        3. 機械学習モデル構築
        4. 予測・スクリーニング
        5. 有望候補の実験検証（数日〜数週間）
        """)

    st.header("1.2 MI のワークフロー")
    st.markdown(r"""
    ```
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ データ収集 │ → │ データ探索 │ → │ モデル構築 │ → │ モデル評価 │ → │ 材料設計  │
    │ OQMD      │   │ 可視化    │   │ 回帰/分類 │   │ CV/汎化   │   │ 逆設計    │
    │ MatBench  │   │ 前処理    │   │ 正則化    │   │ 性能評価   │   │ 最適化    │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
    ```
    """)

    st.header("1.3 主要な材料データベース")
    st.markdown("""
    | データベース | 内容 | URL |
    |:---|:---|:---|
    | **OQMD** | ~100万件の第一原理計算（DFT）結果。形成エネルギー・バンド構造等を収録。Open Quantum Materials Database の略。ノースウェスタン大学が運営。 | oqmd.org |
    | **Materials Project** | 結晶構造・電子状態・弾性定数 | materialsproject.org |
    | **MatBench** | 標準化された ML ベンチマーク（13タスク） | matbench.materialsproject.org |
    | **AFLOW** | 結晶構造・熱力学特性 | aflow.org |
    | **Open Catalyst** | 触媒反応の DFT データ | opencatalystproject.org |
    | **PolyInfo** | 高分子物性データ（NIMS運営） | polymer.nims.go.jp |
    | **Matminer** | 材料特徴量の自動生成ライブラリ + 45種のデータセット | hackingmaterials.lbl.gov/matminer |
    | **SuperCon** | 超伝導材料データベース（NIMS運営） | supercon.nims.go.jp |
    | **Kaggle** | 超伝導体臨界温度、高分子Tg等のコンペデータ | kaggle.com |
    """)

    with st.expander("💡 OQMDとは？（詳細）"):
        st.markdown("""
        **OQMD (Open Quantum Materials Database)** は、ノースウェスタン大学の Wolverton グループが
        構築した第一原理計算データベースである。

        - **収録数**: ~100万件以上の DFT (密度汎関数理論) 計算結果
        - **主な物性**: 形成エネルギー、バンドギャップ、安定性（convex hull からの距離）
        - **特徴**: ICSD（無機結晶構造データベース）の全構造を系統的に計算
        - **用途**: 新材料のスクリーニング、安定相の予測、機械学習モデルの訓練データ
        - **API**: RESTful API で大量データの取得が可能
        - **本アプリとの関係**: HEAデータの安定性評価や、鉄鋼合金の相安定性の参考に利用

        ```python
        # OQMDからデータ取得の例（qmpy_rester使用）
        from qmpy_rester import QMPYRester
        with QMPYRester() as q:
            data = q.get_oqmd_phases(element_set="Fe-Cr-Ni", stability="stable")
        ```
        """)

    st.header("1.4 本講義で使用するデータセット")
    st.info("""
    **本アプリでは以下の実データを使用する：**

    1. **鉄鋼 降伏強度** (312件) — Citrine Informatics 実験値 / Matbench
       - 13種の合金元素組成 → 降伏強度 (MPa)
    2. **超伝導体 臨界温度** (500件) — NIMS SuperCon / UCI
       - 元素特徴量の統計値 → 臨界温度 Tc (K)
    3. **高エントロピー合金 相分類** (1,103件) — Zenodo ACHIEF
       - VEC・原子半径差・電気陰性度差 → 相（SS/IM/AM）
       - **Hume-Rothery則** の機械学習による再現
    """)

    st.header("1.5 機械学習の基本的な枠組み")
    st.markdown(r"""
    機械学習の目標は、入力 $\mathbf{x}$ から出力 $y$ への写像 $f$ を学習することである。

    $$
    y = f(\mathbf{x}) + \varepsilon
    $$

    - $\mathbf{x} = (x_1, x_2, \dots, x_p)^T$：**特徴量**（説明変数）— 組成・プロセス条件など
    - $y$：**目的変数**（応答変数）— 物性値（強度、Tc など）
    - $\varepsilon$：ノイズ（測定誤差など）、$E[\varepsilon] = 0$

    **回帰問題**: $y$ が連続値（例：降伏強度 300 MPa）

    **分類問題**: $y$ がカテゴリ（例：相 = BCC / FCC / SS）
    """)

    st.markdown(r"""
    ### 損失関数

    モデルの予測 $\hat{y}$ と真の値 $y$ の誤差を測る関数：

    - **平均二乗誤差 (MSE)**:
    $$
    \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    $$

    - **平均絶対誤差 (MAE)**:
    $$
    \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
    $$

    - **決定係数** $R^2$:
    $$
    R^2 = 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}
    $$

    ここで、
    $$
    \mathrm{SSE} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
    $$
    は **誤差平方和** であり、予測値 $\hat{y}_i$ と実測値 $y_i$ のずれの二乗和である。小さいほど予測誤差が小さい。

    $$
    \mathrm{SST} = \sum_{i=1}^{n}(y_i - \bar{y})^2
    $$
    は **全平方和** であり、実測値 $y_i$ が平均値 $\bar{y}$ のまわりにどれだけ散らばっているかを表す。

    $R^2 = 1$ で完全な予測、$R^2 = 0$ で平均値予測と同等、$R^2 < 0$ では平均値予測より悪い予測である。
    """)


# =====================================================================
# セクション 2: データ探索
# =====================================================================
elif section_key == "data_exploration":
    st.title("📊 データ探索")
    st.markdown(f"**使用データ**: {dataset_desc}")
    st.markdown("---")

    st.header("2.1 データの概要")
    st.markdown("#### データセットの説明")
    st.markdown(dataset_detail)
    st.dataframe(df_main.head(20), use_container_width=True)
    st.markdown(f"データ数: **{len(df_main)}** 件、特徴量数: **{len(feature_cols)}** 個、目的変数: **{target_col}**")
    if dataset_choice == "鉄鋼（構造材料）":
        st.caption("注: tensile strength と elongation は同時測定された別の応答変数なので、予測モデルの入力特徴量からは除外している。")

    st.header("2.2 要約統計量")
    st.markdown(r"""
    **要約統計量** はデータの全体像を把握するための基本指標である。

    | 指標 | 数式 | 意味 |
    |:---|:---|:---|
    | 平均 | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | データの中心 |
    | 分散 | $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$ | データの散らばり |
    | 標準偏差 | $s = \sqrt{s^2}$ | 分散の平方根（元の単位） |
    | 中央値 | $\tilde{x}$ | データを並べた中央の値 |
    | 四分位範囲 | $\text{IQR} = Q_3 - Q_1$ | 外れ値検出の基準 |
    """)
    st.dataframe(df_main.describe().round(3), use_container_width=True)

    st.markdown("#### 四分位とIQRの確認")
    st.markdown("IQRは第3四分位数 Q3 と第1四分位数 Q1 の差であり、中央50%のデータがどれくらい広がっているかを表す。外れ値検出では、Q1よりかなり小さい点、またはQ3よりかなり大きい点を候補として扱う。")
    numeric_cols_for_quartile = [c for c in df_main.columns if is_numeric_dtype(df_main[c])]
    quartile_col_demo = st.selectbox("四分位を表示する変数", numeric_cols_for_quartile, index=numeric_cols_for_quartile.index(target_col) if target_col in numeric_cols_for_quartile else 0)
    q = df_main[quartile_col_demo].quantile([0, 0.25, 0.5, 0.75, 1.0])
    q_table = pd.DataFrame({
        "指標": ["最小値", "Q1（第1四分位）", "Q2（中央値）", "Q3（第3四分位）", "最大値", "IQR = Q3 - Q1"],
        "値": [q.loc[0], q.loc[0.25], q.loc[0.5], q.loc[0.75], q.loc[1.0], q.loc[0.75] - q.loc[0.25]],
    })
    st.dataframe(q_table.round(3), use_container_width=True)

    st.header("2.3 分布の可視化（ヒストグラム）")
    hist_col = st.selectbox("表示する変数", df_main.columns.tolist())
    fig_hist = px.histogram(df_main, x=hist_col, nbins=30, marginal="box",
                            title=f"{hist_col} の分布")
    fig_hist.update_layout(height=400)
    _plotly_chart(fig_hist)

    st.header("2.4 相関行列")
    st.markdown(r"""
    **ピアソン相関係数** は2変数間の線形関係の強さを表す：

    $$
    r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
    {\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2 \sum_{i=1}^{n}(y_i-\bar{y})^2}}
    $$
    """)
    corr = df_main[feature_cols + [target_col]].corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="相関行列ヒートマップ")
    fig_corr.update_layout(height=820, width=1100, xaxis_title="変数", yaxis_title="変数")
    _plotly_chart(fig_corr)

    st.header("2.5 ペアプロット（散布図行列）")
    pair_cols = st.multiselect("表示する変数（2〜4個推奨）",
                               df_main.columns.tolist(),
                               default=feature_cols[:3] + [target_col])
    if len(pair_cols) >= 2:
        pair_size = max(720, min(1120, 260 * len(pair_cols)))
        fig_pair = px.scatter_matrix(df_main[pair_cols], dimensions=pair_cols,
                                     height=pair_size, width=pair_size, title="ペアプロット（各セルが正方形になるよう表示）")
        fig_pair.update_traces(diagonal_visible=True, marker=dict(size=3))
        fig_pair.update_layout(autosize=False)
        _plotly_chart_fixed(fig_pair)
    else:
        st.warning("2つ以上の変数を選択すること。")

    st.header("2.6 異常データの検出（外れ値検出）")
    st.markdown(r"""
    材料データには測定ミスや特殊条件のデータが含まれることがある。

    ### IQR 法（四分位範囲法）
    $$
    \text{外れ値条件}: \quad x < Q_1 - 1.5 \times \text{IQR} \quad \text{or} \quad x > Q_3 + 1.5 \times \text{IQR}
    $$

    ### Isolation Forest
    ランダムな分割で**孤立しやすいデータ**を異常とみなす。

    ### LOF (Local Outlier Factor)
    局所的なデータ密度を比較し、**周囲より密度が低い**点を異常とみなす。
    """)

    outlier_method = st.selectbox("外れ値検出手法", ["IQR法", "Isolation Forest", "LOF"])
    _outlier_numeric_cols = [c for c in df_main.columns if is_numeric_dtype(df_main[c])]
    _outlier_default_idx = (
        _outlier_numeric_cols.index(target_col)
        if target_col in _outlier_numeric_cols
        else len(_outlier_numeric_cols) - 1
    )
    outlier_col = st.selectbox("対象変数", _outlier_numeric_cols,
                               index=_outlier_default_idx)

    df_outlier = df_main.copy()
    if outlier_method == "IQR法":
        Q1 = df_outlier[outlier_col].quantile(0.25)
        Q3 = df_outlier[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_outlier["外れ値"] = ((df_outlier[outlier_col] < lower) |
                              (df_outlier[outlier_col] > upper))
        st.markdown(f"Q1 = {Q1:.2f}, Q2（中央値） = {df_outlier[outlier_col].median():.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
        st.markdown(f"外れ値候補の範囲: {lower:.2f} 未満、または {upper:.2f} 超過")
    elif outlier_method == "Isolation Forest":
        contamination = st.slider("contamination（異常割合）", 0.01, 0.2, 0.05)
        iso = IsolationForest(contamination=contamination, random_state=42)
        X_num = df_main[feature_cols].values
        preds = iso.fit_predict(X_num)
        df_outlier["外れ値"] = preds == -1
    else:
        n_neighbors = st.slider("近傍数", 5, 50, 20)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        X_num = df_main[feature_cols].values
        preds = lof.fit_predict(X_num)
        df_outlier["外れ値"] = preds == -1

    n_outliers = df_outlier["外れ値"].sum()
    st.metric("検出された外れ値の数", f"{n_outliers} / {len(df_outlier)}")

    fig_out = px.scatter(df_outlier, x=df_outlier.index, y=outlier_col,
                         color=df_outlier["外れ値"].map({True: "外れ値", False: "正常"}),
                         color_discrete_map={"外れ値": "red", "正常": "blue"},
                         title=f"{outlier_method} による外れ値検出結果")
    fig_out.update_layout(height=400)
    _plotly_chart(fig_out)


# =====================================================================
# セクション 3: 次元削減 PCA (Biplot)
# =====================================================================
elif section_key == "pca":
    st.title("🔄 主成分分析 (PCA) — Biplot 可視化")
    st.markdown(f"**使用データ**: {dataset_desc}")
    st.markdown("---")

    st.header("3.1 PCA の理論")
    st.markdown(r"""
    **主成分分析 (Principal Component Analysis; PCA)** は、高次元データを少数の主成分に
    射影して次元を削減する手法である。

    ### 手順
    1. データの**標準化**（平均0、分散1に変換）
    2. **共分散行列** $\mathbf{C}$ の計算:
    $$
    \mathbf{C} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}
    $$
    3. 共分散行列の**固有値問題**を解く:
    $$
    \mathbf{C} \mathbf{v}_k = \lambda_k \mathbf{v}_k
    $$
    4. 固有値の大きい順に主成分を選択

    ### Biplot（バイプロット）
    PCA の結果を **スコア（データ点の射影）** と **負荷量（特徴量の方向）** を
    同一図上に描画する可視化手法：

    - **点**: 各サンプルの PC1-PC2 スコア
    - **矢印**: 各特徴量の負荷量ベクトル（元の変数がどの方向に寄与するか）
    - 矢印が長い = その主成分への寄与が大きい
    - 矢印が同じ方向 = 特徴量間に正の相関がある
    """)

    st.header("3.2 PCAで使うデータの確認")
    st.markdown("PCAには、目的変数を除いた数値特徴量を入力する。標準化してから主成分を計算するため、単位やスケールが異なる特徴量を同じ土俵で比較できる。")
    with st.expander("データセットの説明", expanded=True):
        st.markdown(dataset_detail)
    st.dataframe(df_main[feature_cols + [target_col]].head(12), use_container_width=True)
    st.download_button(
        "このPCA用データをCSVとして保存",
        _csv_bytes(df_main[feature_cols + [target_col]]),
        file_name="pca_input_data.csv",
        mime="text/csv",
        key=f"download_pca_input_{dataset_choice}",
    )

    st.markdown("#### PCA前のペアプロット")
    st.caption("下三角に散布図、対角に分布、上三角にピアソン相関係数 r を表示する。PCA前に、変数間の線形相関と冗長性を確認するための図である。")
    pca_pair_options = [c for c in df_main.columns if is_numeric_dtype(df_main[c])]
    default_pca_pair = [c for c in feature_cols[:3] + [target_col] if c in pca_pair_options][:4]
    pca_pair_cols = st.multiselect(
        "PCA前に関係を見る変数（2〜4個推奨）",
        pca_pair_options,
        default=default_pca_pair,
        key="pca_pair_cols",
    )
    if len(pca_pair_cols) >= 2:
        fig_pca_pair, n_all_pair, n_plot_pair = _create_pairplot_with_upper_corr(
            df_main,
            pca_pair_cols,
            title="PCA前のペアプロット（上三角は相関係数）",
        )
        if fig_pca_pair is not None:
            if n_plot_pair < n_all_pair:
                st.caption(f"表示を軽くするため、散布図は {n_all_pair} 件中 {n_plot_pair} 件を抽出して描画している。相関係数は全 {n_all_pair} 件で計算している。")
            _plotly_chart_fixed(fig_pca_pair)
        else:
            st.warning("選択した変数で有効な数値データが不足している。")
    else:
        st.warning("2つ以上の数値変数を選択すること。")

    st.header("3.3 PCA の実行")
    X = df_main[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(len(feature_cols), X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    st.subheader("寄与率と累積寄与率")
    st.caption("寄与率は、各主成分が元データのばらつきをどれだけ説明しているかを表す。累積寄与率は、PC1から順に足し合わせた説明量である。")
    fig_var = go.Figure()
    fig_var.add_bar(x=[f"PC{i+1}" for i in range(len(explained))],
                    y=explained * 100, name="寄与率")
    fig_var.add_scatter(x=[f"PC{i+1}" for i in range(len(cumulative))],
                       y=cumulative * 100, name="累積寄与率",
                       mode="lines+markers")
    fig_var.update_layout(
        title="寄与率と累積寄与率",
        xaxis_title="主成分",
        yaxis_title="寄与率 (%)",
        height=640,
        bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_var.update_yaxes(range=[0, max(105, float(np.max(cumulative * 100)) + 5)])
    _plotly_chart(fig_var)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=feature_cols,
    )
    st.subheader("主成分負荷量ヒートマップ")
    st.caption("主成分負荷量は、各特徴量が各主成分にどの向き・強さで効いているかを表す。PC1 から最終主成分まで省略せず表示する。")
    fig_load = px.imshow(loadings,
                         text_auto=".2f",
                         color_continuous_scale="RdBu_r",
                         title="主成分負荷量（全主成分を表示）")
    pc_labels = [f"PC{i+1}" for i in range(n_components)]
    fig_load.update_layout(
        height=max(760, 52 * len(feature_cols) + 280),
        width=max(1180, 90 * n_components + 380),
        xaxis_title="主成分",
        yaxis_title="特徴量",
        margin=dict(l=120, r=40, t=90, b=110),
    )
    fig_load.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_components)),
        ticktext=pc_labels,
        tickangle=-45,
    )
    _plotly_chart_fixed(fig_load)
    with st.expander("主成分負荷量の数値表（全主成分）", expanded=False):
        st.dataframe(loadings.round(3), use_container_width=True, height=min(620, 40 * len(feature_cols) + 100))

    # Biplot（正方形）
    st.header("3.4 Biplot（スコア＋負荷量ベクトル）")
    if n_components >= 2:
        fig_biplot = _create_biplot(
            X_pca, pca, feature_cols, explained,
            target_values=df_main[target_col].values
        )
        _plotly_chart(fig_biplot)

        st.markdown(r"""
        **Biplot の読み方:**
        - 各**点**はサンプル（色は目的変数の値）
        - 各**赤い矢印**は特徴量の負荷量ベクトル
        - 矢印の方向: その特徴量が増加する方向
        - 矢印の長さ: その主成分への寄与の大きさ
        - 近い方向の矢印 → 特徴量同士に正の相関
        - 反対方向の矢印 → 負の相関
        """)

    st.info(r"""
    **PCA の材料科学での活用例**
    - 合金組成空間の可視化（多元系合金の組成を2Dで表示）
    - 類似材料のグルーピング
    - 特徴量の冗長性の発見（相関の高い特徴量を圧縮）
    - ノイズ除去（低寄与率の主成分を除外）
    """)


# =====================================================================
# セクション 4: 回帰問題
# =====================================================================
elif section_key == "regression":
    st.title("📈 回帰問題")
    st.markdown(f"**使用データ**: {dataset_desc}")
    st.markdown("---")

    st.header("4.1 線形回帰用CSVデータの表示・編集")
    st.markdown("""
    回帰モデルに入れる数値データを、表形式で確認・編集できる画面である。
    ここでの編集はアプリ上の一時的な変更であり、元のCSVファイル自体は上書きしない。
    値を変更した後に「編集後データを使う」を有効にすると、下の線形回帰・SVR・Random Forest の計算に反映される。
    """)

    editable_cols = feature_cols + [target_col]
    edited_table = st.data_editor(
        df_main[editable_cols],
        num_rows="fixed",
        height=620,
        use_container_width=True,
        key=f"regression_csv_editor_{dataset_choice}",
    )
    use_edited_data = st.checkbox(
        "編集後データを下の回帰計算に使う",
        value=False,
        help="授業中に値を変えたとき、モデルの係数・R²・パリティプロットがどう変わるかを確認するための設定である。",
    )
    st.download_button(
        "現在表示中の表をCSVとして保存",
        edited_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="edited_regression_data.csv",
        mime="text/csv",
    )

    df_reg = df_main.copy()
    if use_edited_data:
        edited_numeric = edited_table.copy()
        for col in editable_cols:
            edited_numeric[col] = pd.to_numeric(edited_numeric[col], errors="coerce")
        df_reg[editable_cols] = edited_numeric[editable_cols]
        n_before = len(df_reg)
        df_reg = df_reg.dropna(subset=editable_cols).reset_index(drop=True)
        if len(df_reg) < n_before:
            st.warning(f"数値に変換できない行または欠損を含む行を {n_before - len(df_reg)} 行除外した。")
        st.info("編集後データを使って下の回帰計算を行う。")
    else:
        st.caption("現在は元のCSVデータで計算している。表の編集は、チェックボックスを有効にするまで計算には反映されない。")

    if dataset_choice == "鉄鋼（構造材料）":
        st.subheader("鋼成分のゼロ近傍データ除外")
        st.markdown("""
        鋼の組成データでは、添加していない元素が **0 または 0 に近い値** として多く含まれる。
        ある成分の添加効果を見たい場合、ゼロ近傍の行を含めたままだと「未添加材」と「添加材」が混ざり、散布図や回帰の傾向が見えにくくなることがある。
        ここでは、選択した成分について、絶対値がしきい値以下の行を除外して解析できる。
        """)
        use_steel_zero_filter = st.checkbox(
            "鋼成分のゼロ近傍データを除外して下の機械学習に使う",
            value=False,
            key="steel_zero_filter_enabled",
        )
        if use_steel_zero_filter:
            steel_component_cols = [
                c for c in feature_cols
                if c in df_reg.columns and is_numeric_dtype(df_reg[c])
            ]
            default_zero_cols = ["c"] if "c" in steel_component_cols else steel_component_cols[:1]
            zero_filter_cols = st.multiselect(
                "ゼロ近傍を除外する成分",
                steel_component_cols,
                default=default_zero_cols,
                key="steel_zero_filter_cols",
            )
            zero_threshold = st.number_input(
                "ゼロ近傍とみなすしきい値（wt%）",
                min_value=0.0,
                max_value=1.0,
                value=0.001,
                step=0.001,
                format="%.4f",
                key="steel_zero_filter_threshold",
            )
            zero_filter_mode = st.radio(
                "複数成分を選んだときの残し方",
                ["選択成分のいずれかがしきい値を超える行を残す", "選択成分のすべてがしきい値を超える行を残す"],
                horizontal=True,
                key="steel_zero_filter_mode",
            )

            if zero_filter_cols:
                values_for_filter = df_reg[zero_filter_cols].abs()
                if zero_filter_mode.startswith("選択成分のすべて"):
                    keep_mask = values_for_filter.gt(zero_threshold).all(axis=1)
                else:
                    keep_mask = values_for_filter.gt(zero_threshold).any(axis=1)

                n_before_zero_filter = len(df_reg)
                n_after_zero_filter = int(keep_mask.sum())
                metric_cols = st.columns(3)
                metric_cols[0].metric("除外前", f"{n_before_zero_filter} 件")
                metric_cols[1].metric("除外後", f"{n_after_zero_filter} 件")
                metric_cols[2].metric("除外数", f"{n_before_zero_filter - n_after_zero_filter} 件")

                if n_after_zero_filter < 10:
                    st.error("ゼロ近傍除外後のデータ数が10件未満である。モデル評価が不安定になるため、この条件は適用しない。しきい値を下げるか、選択成分を変更すること。")
                else:
                    df_reg = df_reg.loc[keep_mask].reset_index(drop=True)
                    st.info(f"{', '.join(zero_filter_cols)} について、|成分量| > {zero_threshold:g} wt% を満たす行を使って下の解析を行う。")
                    with st.expander("ゼロ近傍除外後のデータ先頭行", expanded=False):
                        st.dataframe(df_reg[editable_cols].head(20), use_container_width=True)
            else:
                st.warning("ゼロ近傍を除外する成分を1つ以上選択すること。")

    if len(df_reg) < 10:
        st.error("解析に使えるデータが10件未満である。編集内容またはゼロ近傍除外条件を見直すこと。")
        st.stop()

    # データ要約の表示
    show_data_summary(df_reg, feature_cols, target_col, dataset_desc, dataset_detail)

    X = df_reg[feature_cols].values
    y = df_reg[target_col].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 4.1 線形回帰
    st.header("4.1 線形回帰 (Linear Regression)")
    st.markdown(r"""
    最も基本的な回帰モデルである。目的変数 $y$ を特徴量の**線形結合**で表現する：

    $$
    \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p = \mathbf{x}^T \boldsymbol{\beta}
    $$

    ### 最小二乗法 (OLS)
    $$
    \boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
    $$
    """)

    st.caption("回帰章ではBiplotを重複表示しない。データ構造の確認は3章のPCAで行う。")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_train_lr = lr.predict(X_train)
    r2_lr = r2_score(y_test, y_pred_lr)
    r2_train_lr = r2_score(y_train, y_pred_train_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    r2_baseline = r2_score(y_test, baseline_pred)
    sse_lr = float(np.sum((y_test - y_pred_lr) ** 2))
    sst_lr = float(np.sum((y_test - y_test.mean()) ** 2))
    sse_sst_ratio = sse_lr / sst_lr if sst_lr > 0 else np.nan
    r2_from_sse_sst = 1.0 - sse_sst_ratio if np.isfinite(sse_sst_ratio) else np.nan
    cv_lr = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_lr = cross_val_score(
        Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        X, y, cv=cv_lr, scoring="r2", n_jobs=N_JOBS,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("テスト R²", f"{r2_lr:.4f}")
    col2.metric("訓練 R²", f"{r2_train_lr:.4f}")
    col3.metric("5-fold CV 平均 R²", f"{cv_scores_lr.mean():.4f}")
    col4.metric("RMSE", f"{rmse_lr:.2f}")
    col5.metric("MAE", f"{mae_lr:.2f}")

    with st.expander("R² が負になるかどうかの検算", expanded=(r2_lr < 0)):
        st.markdown(fr"""
        R² は次式で定義される。

        $$
        R^2 = 1 - \frac{{\mathrm{{SSE}}}}{{\mathrm{{SST}}}}
        $$

        - $\mathrm{{SSE}} = \sum_i (y_i - \hat{{y}}_i)^2$：**誤差平方和**。実測値と予測値のずれの二乗和であり、小さいほどよい。
        - $\mathrm{{SST}} = \sum_i (y_i - \bar{{y}})^2$：**全平方和**。実測値が平均値のまわりにどれだけ散らばっているかを表す。
        - $\mathrm{{SSE}} > \mathrm{{SST}}$ なら、平均値で予測するより誤差が大きいため、$R^2 < 0$ になる。

        現在のテストデータでは以下である。

        - SSE（誤差平方和）: **{sse_lr:.3f}**
        - SST（全平方和）: **{sst_lr:.3f}**
        - SSE/SST: **{sse_sst_ratio:.3f}**
        - 式から再計算した R²: **{r2_from_sse_sst:.4f}**
        - scikit-learn のテスト R²: **{r2_lr:.4f}**
        - 訓練 R²: **{r2_train_lr:.4f}**
        - 5-fold CV の各R²: `{np.round(cv_scores_lr, 4).tolist()}`
        - 5-fold CV 平均R²: **{cv_scores_lr.mean():.4f}**
        - 訓練データ平均値を使った単純予測のR²: **{r2_baseline:.4f}**

        したがって、現在の表示でR²が負になる場合、それは計算ミスではない。
        とくに鉄鋼データでは、組成特徴量だけで `yield strength` を予測しているため、
        分割のされ方によって線形回帰のテストR²が負になることがある。
        """)
        if dataset_choice == "鉄鋼（構造材料）" and all(c in df_reg.columns for c in ["tensile strength", "elongation"]):
            leaky_cols = [
                c for c in df_reg.columns
                if c != target_col and is_numeric_dtype(df_reg[c])
            ]
            X_leaky = df_reg[leaky_cols].values
            Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(X_leaky, y, test_size=0.2, random_state=42)
            leaky_model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
            leaky_model.fit(Xl_tr, yl_tr)
            r2_leaky = r2_score(yl_te, leaky_model.predict(Xl_te))
            st.info(f"参考: `tensile strength` と `elongation` も特徴量に入れると、この分割では R² = {r2_leaky:.4f} になる。ただし、これらは同時測定された別の応答変数であり、実際の予測では使えない情報を使うため、教材では除外する。")

    if r2_lr < 0:
        st.warning(f"""
        **R² が負値である。** 線形回帰モデルが平均値予測より悪い結果であることを示す。
        現在の特徴量だけでは {target_col} の線形予測が難しい、またはテスト分割が厳しいことを意味する。
        SVR や Random Forest などの非線形モデルと比較すると、改善の有無を確認できる。
        """)

    fig_lr = px.scatter(x=y_test, y=y_pred_lr,
                        labels={"x": f"実測値 ({target_col})", "y": "予測値"},
                        title=f"線形回帰: 予測 vs 実測 (R² = {r2_lr:.4f})")
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    fig_lr.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                     line=dict(dash="dash", color="red"))
    fig_lr.update_layout(height=650)
    _plotly_chart(fig_lr)

    st.subheader("線形回帰の x-y プロット（1特徴量で見る）")
    st.markdown("多変量線形回帰は多数の特徴量を同時に使うが、まずは1つの特徴量 x と目的変数 y の関係を散布図で見ると理解しやすい。ここでは選んだ1特徴量だけで単回帰直線も重ねて表示する。")
    xy_feature = st.selectbox("x軸にする特徴量", feature_cols, key="linear_xy_feature")
    xy_idx = feature_cols.index(xy_feature)
    single_lr = LinearRegression()
    single_lr.fit(X_train_raw[:, [xy_idx]], y_train)
    x_grid = np.linspace(df_reg[xy_feature].min(), df_reg[xy_feature].max(), 200).reshape(-1, 1)
    y_grid = single_lr.predict(x_grid)
    df_xy_plot = pd.DataFrame({
        xy_feature: np.concatenate([X_train_raw[:, xy_idx], X_test_raw[:, xy_idx]]),
        target_col: np.concatenate([y_train, y_test]),
        "データ種別": ["訓練"] * len(y_train) + ["テスト"] * len(y_test),
    })
    fig_xy = px.scatter(df_xy_plot, x=xy_feature, y=target_col, color="データ種別", title=f"{xy_feature} と {target_col} の関係")
    fig_xy.add_scatter(x=x_grid.ravel(), y=y_grid, mode="lines", name="単回帰直線")
    fig_xy.update_layout(height=650)
    _plotly_chart(fig_xy)

    coef_df = pd.DataFrame({
        "特徴量": feature_cols,
        "係数 β": lr.coef_,
        "|β|": np.abs(lr.coef_),
    }).sort_values("|β|", ascending=False)
    fig_coef = px.bar(coef_df, x="特徴量", y="係数 β", title="回帰係数（標準化後）",
                      color="係数 β", color_continuous_scale="RdBu_r")
    fig_coef.update_layout(height=460)
    _plotly_chart(fig_coef)

    # 4.2 多項式回帰と過学習
    st.header("4.2 多項式回帰と過学習")
    st.markdown(r"""
    **多項式回帰**は特徴量の高次の項を追加して非線形性を表現する：

    $$
    \hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d
    $$

    ### バイアス-バリアンス分解
    $$
    E\left[(y - \hat{f}(\mathbf{x}))^2\right] = \underbrace{\text{Bias}^2[\hat{f}]}_{\text{未学習}} + \underbrace{\text{Var}[\hat{f}]}_{\text{過学習}} + \underbrace{\sigma^2}_{\text{ノイズ}}
    $$
    """)

    demo_feature = st.selectbox("特徴量を選択", feature_cols, key="poly_feat")
    max_degree = st.slider("多項式の最大次数", 1, 15, 10)

    X_demo = df_reg[demo_feature].values.reshape(-1, 1)
    y_demo = y.copy()
    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_demo, y_demo, test_size=0.2, random_state=42)

    train_errors = []
    test_errors = []
    degrees = list(range(1, max_degree + 1))
    for d in degrees:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        pipe.fit(X_tr_d, y_tr_d)
        train_errors.append(mean_squared_error(y_tr_d, pipe.predict(X_tr_d)))
        test_errors.append(mean_squared_error(y_te_d, pipe.predict(X_te_d)))

    default_degrees = sorted({d for d in [1, 3, max_degree] if d in degrees})
    selected_degrees = st.multiselect("表示する次数", degrees, default=default_degrees)
    x_plot = np.linspace(X_demo.min(), X_demo.max(), 300).reshape(-1, 1)
    fig_fit = go.Figure()
    fig_fit.add_scatter(x=X_tr_d.ravel(), y=y_tr_d, mode="markers",
                       name="訓練データ", marker=dict(size=5, opacity=0.55))
    fig_fit.add_scatter(x=X_te_d.ravel(), y=y_te_d, mode="markers",
                       name="テストデータ", marker=dict(size=7, opacity=0.65, symbol="x"))
    colors = px.colors.qualitative.Set1
    for i, d in enumerate(selected_degrees):
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        pipe.fit(X_tr_d, y_tr_d)
        y_plot = pipe.predict(x_plot)
        fig_fit.add_scatter(x=x_plot.ravel(), y=y_plot, mode="lines",
                           name=f"次数 {d}", line=dict(color=colors[i % len(colors)], width=3))
    fig_fit.update_layout(
        title="多項式フィッティング（過学習観察用・大きめ表示）",
        xaxis_title=demo_feature, yaxis_title=target_col, height=860,
    )
    _plotly_chart(fig_fit)

    fig_overfit = go.Figure()
    fig_overfit.add_scatter(x=degrees, y=train_errors, mode="lines+markers", name="訓練誤差 (MSE)")
    fig_overfit.add_scatter(x=degrees, y=test_errors, mode="lines+markers", name="テスト誤差 (MSE)")
    fig_overfit.update_layout(
        title="過学習の可視化: 次数 vs 誤差（大きめ表示）",
        xaxis_title="多項式の次数", yaxis_title="MSE", height=760,
    )
    _plotly_chart(fig_overfit)
    st.caption("次数を上げると訓練誤差は下がりやすい。一方、テスト誤差が増え始める場合は、訓練データの偶然のばらつきまで学習している状態、すなわち過学習である。")

    best_deg = degrees[np.argmin(test_errors)]
    st.success(f"テスト誤差が最小の次数: **{best_deg}** (MSE = {min(test_errors):.2f})")

    # 4.3 SVR
    st.header("4.3 サポートベクター回帰 (SVR)")
    st.markdown(r"""
    **サポートベクター回帰（SVR）** は、**サポートベクターマシン（SVM）** を回帰分析に応用した機械学習手法である。
    「誤差を許容するマージン（ε-チューブ）内にできるだけ多くのデータ点を収めながら、そのチューブ幅を広く取る」ことを目指して回帰関数を決定する。

    予測値と実測値の差が $\varepsilon$ 以内なら損失を0とする **ε-不感損失関数** を使う：

    $$
    \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, |y_i - \hat{y}_i| - \varepsilon)
    $$

    RBF カーネル: $K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$
    """)

    st.info("SVR は目的変数のスケールに敏感である。ここでは既定で y も標準化して学習し、予測時に元の単位へ戻す。これにより、材料強度のように値のスケールが大きいデータでも精度が出やすくなる。")
    col1, col2, col3 = st.columns(3)
    with col1:
        svr_kernel = st.selectbox("カーネル", ["rbf", "linear", "poly"])
        svr_C = st.slider("C（誤差ペナルティ）", 0.1, 100.0, 10.0, key="svr_c")
    with col2:
        svr_epsilon = st.slider("ε (標準化 y 上の不感帯)", 0.01, 0.50, 0.05, key="svr_eps")
        svr_gamma_choice = st.selectbox("gamma", ["scale", "auto", "0.1", "0.03", "0.01"], index=0)
    with col3:
        svr_scale_y = st.checkbox("目的変数 y も標準化する", value=True)

    svr_gamma = float(svr_gamma_choice) if svr_gamma_choice not in ["scale", "auto"] else svr_gamma_choice
    svr_base = SVR(kernel=svr_kernel, C=svr_C, epsilon=svr_epsilon, gamma=svr_gamma)
    if svr_scale_y:
        svr = TransformedTargetRegressor(regressor=svr_base, transformer=StandardScaler())
    else:
        svr = svr_base
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    r2_svr = r2_score(y_test, y_pred_svr)
    rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))

    col1, col2, col3 = st.columns(3)
    col1.metric("SVR R²", f"{r2_svr:.4f}")
    col2.metric("SVR RMSE", f"{rmse_svr:.2f}")
    col3.metric("y標準化", "あり" if svr_scale_y else "なし")

    fig_svr = px.scatter(x=y_test, y=y_pred_svr,
                         labels={"x": f"実測値 ({target_col})", "y": "予測値"},
                         title=f"SVR ({svr_kernel}): R² = {r2_svr:.4f}")
    min_val = min(y_test.min(), y_pred_svr.min())
    max_val = max(y_test.max(), y_pred_svr.max())
    fig_svr.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                      line=dict(dash="dash", color="red"))
    fig_svr.update_layout(height=740)
    _plotly_chart(fig_svr)

    with st.expander("SVR の精度が悪い場合の確認", expanded=(r2_svr < r2_lr)):
        st.markdown("""
        SVR は特徴量と目的変数のスケールに敏感である。まず以下を確認する。

        - カーネルは `rbf` を基準にする。
        - 「目的変数 y も標準化する」を有効にする。
        - `C=10`, `epsilon=0.05`, `gamma=scale` を基準設定とし、そこから少しずつ動かす。
        - 線形回帰より悪い場合は、データ分割や外れ値の影響も疑う。
        """)

    # 4.4 Random Forest
    st.header("4.4 ランダムフォレスト回帰と特徴量重要度")
    st.markdown(r"""
    **ランダムフォレスト** はバギング + 決定木のアンサンブル学習である：

    $$
    \hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})
    $$

    **特徴量重要度 (MDI)**: 各特徴量で分岐した時の不純度の減少量の合計
    """)

    st.markdown("Random Forest は木の本数、深さ、葉に必要なサンプル数などで性能が変わる。ここでは講義時間内で動くよう、候補を絞った簡易チューニングを行う。")
    rf_tune = st.checkbox("RFの簡易ハイパーパラメータチューニングを実施する", value=True)

    if rf_tune:
        rf_grid_size = st.selectbox("チューニング範囲", ["軽量（講義向け）", "標準"], index=0)
        with st.spinner("Random Forest の簡易チューニングを実行中..."):
            best_params_rf, best_score_rf = _cached_rf_tuning(X_train, y_train, rf_grid_size)
        rf = RandomForestRegressor(**best_params_rf, random_state=42, n_jobs=N_JOBS)
        rf.fit(X_train, y_train)
        st.info(f"最良パラメータ: {best_params_rf} / 3-fold CV R² = {best_score_rf:.4f}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            rf_n_estimators = st.slider("木の本数", 10, 500, 100, step=10)
        with col2:
            rf_max_depth = st.slider("最大深さ", 2, 30, 10)
        rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                    random_state=42, n_jobs=N_JOBS)
        rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    col1, col2 = st.columns(2)
    col1.metric("RF R²", f"{r2_rf:.4f}")
    col2.metric("RF RMSE", f"{rmse_rf:.2f}")

    importance_df = pd.DataFrame({
        "特徴量": feature_cols,
        "重要度 (MDI)": rf.feature_importances_,
    }).sort_values("重要度 (MDI)", ascending=True)

    fig_rf = px.scatter(x=y_test, y=y_pred_rf,
                        labels={"x": f"実測値 ({target_col})", "y": "予測値"},
                        title=f"Random Forest パリティプロット（大きめ表示）: R² = {r2_rf:.4f}")
    min_v = min(y_test.min(), y_pred_rf.min())
    max_v = max(y_test.max(), y_pred_rf.max())
    fig_rf.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                     line=dict(dash="dash", color="red"))
    fig_rf.update_layout(height=780)
    _plotly_chart(fig_rf)

    fig_imp = px.bar(importance_df, x="重要度 (MDI)", y="特徴量",
                     orientation="h", title="特徴量重要度 (MDI)",
                     color="重要度 (MDI)", color_continuous_scale="Viridis")
    fig_imp.update_layout(height=500)
    _plotly_chart(fig_imp)

    # モデル比較
    st.header("4.5 回帰モデルの比較")
    compare_df = pd.DataFrame({
        "モデル": ["線形回帰", "SVR", "Random Forest"],
        "R²": [r2_lr, r2_svr, r2_rf],
        "RMSE": [rmse_lr, rmse_svr, rmse_rf],
        "MAE": [mae_lr, mean_absolute_error(y_test, y_pred_svr),
                mean_absolute_error(y_test, y_pred_rf)],
    })
    st.dataframe(compare_df.style.highlight_max(subset=["R²"], color="lightgreen")
                 .highlight_min(subset=["RMSE", "MAE"], color="lightgreen"),
                 use_container_width=True)


# =====================================================================
# セクション 5: 分類問題（Hume-Rothery則）
# =====================================================================
elif section_key == "classification":
    st.title("🏷️ 分類問題 — Hume-Rothery 則の再現")
    st.markdown("**使用データ**: 高エントロピー合金 (HEA) 相分類 — Zenodo ACHIEF (1,103件)")
    st.markdown("---")

    st.header("5.1 HEAデータの俯瞰")
    st.markdown(r"""
    **HEA (High-Entropy Alloy; 高エントロピー合金)** は、複数の主要元素を比較的高濃度で含む合金である。
    従来合金のように「Fe基」「Ni基」と単一の主元素を決めるのではなく、多元素組成空間を探索する点が特徴である。

    このデータでは、合金組成から計算された VEC、原子半径差 `r_delta` ($\delta$)、混合エンタルピーなどを使い、相の種類を分類する。

    | ラベル | 意味 | 授業での見方 |
    |:---|:---|:---|
    | SS | Solid Solution、固溶体相 | 単相または固溶体としてまとまりやすい相 |
    | IM | Intermetallic、金属間化合物相 | 規則構造・化合物形成の傾向が強い相 |
    | AM | Amorphous、非晶質相 | 長距離秩序を持たない相 |
    """)
    _cls_features = ["VEC", "r_delta", "D_elec_nega", "Hmix (kJ/mol)", "Tm (K)"]
    st.dataframe(df_hea[["Alloy", "S_Phase"] + _cls_features].head(20), use_container_width=True)
    st.download_button(
        "HEA相分類データをCSVとして保存",
        _csv_bytes(df_hea),
        file_name="HEA_phases.csv",
        mime="text/csv",
        key="download_hea_classification",
    )
    overview_cols = st.columns(3)
    overview_cols[0].metric("データ数", f"{len(df_hea)}")
    overview_cols[1].metric("相クラス数", f"{df_hea['S_Phase'].nunique()}")
    overview_cols[2].metric("分類特徴量数", f"{len(_cls_features)}")

    st.markdown("#### 分類に使う相クラスと特徴量")
    cls_detail_df = df_hea["S_Phase"].value_counts().rename_axis("相クラス").reset_index(name="データ数")
    st.dataframe(cls_detail_df, use_container_width=True, hide_index=True)
    feature_detail_df = pd.DataFrame({
        "分類特徴量": _cls_features,
        "意味": [
            "価電子濃度。BCC/FCC傾向の経験的指標である。",
            "原子半径差 δ。原子サイズ不一致と格子ひずみの指標である。",
            "電気陰性度差。化合物形成傾向の目安である。",
            "混合エンタルピー。負に大きいほど化合物化しやすい。",
            "組成平均の融点。熱的安定性の粗い指標である。",
        ],
    })
    st.dataframe(feature_detail_df, use_container_width=True, hide_index=True)

    cls_counts_overview = df_hea["S_Phase"].value_counts()
    fig_cls_overview = px.bar(
        x=cls_counts_overview.index, y=cls_counts_overview.values,
        title="HEA相ラベルの分布", labels={"x": "相ラベル", "y": "データ数"},
        color=cls_counts_overview.index,
    )
    fig_cls_overview.update_layout(height=420)
    _plotly_chart(fig_cls_overview)

    st.caption(f"ペアプロットには分類特徴量 {_cls_features} の {len(_cls_features)} 個をすべて表示する。")
    fig_hea_pair = px.scatter_matrix(
        df_hea[["S_Phase"] + _cls_features],
        dimensions=_cls_features,
        color="S_Phase",
        title="HEA特徴量のペアプロット（分類特徴量をすべて表示）",
        height=1120,
        width=1120,
    )
    fig_hea_pair.update_traces(diagonal_visible=True, marker=dict(size=3))
    fig_hea_pair.update_layout(autosize=False)
    _plotly_chart_fixed(fig_hea_pair)

    # イメージ図
    st.header("5.2 分類問題のイメージ")
    st.markdown(r"""
    ### 分類問題とは
    入力 $\mathbf{x}$ を離散的なカテゴリ $y \in \{C_1, C_2, \dots, C_K\}$ に分類する問題である。
    """)

    # SVM のイメージ図
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### SVM（サポートベクターマシン）のイメージ")
        # 2Dデモデータで決定境界を可視化
        rng = np.random.default_rng(42)
        demo_n = 60
        X_demo_a = rng.normal([2, 2], 0.8, (demo_n, 2))
        X_demo_b = rng.normal([5, 5], 0.8, (demo_n, 2))
        X_demo_svm = np.vstack([X_demo_a, X_demo_b])
        y_demo_svm = np.array([0]*demo_n + [1]*demo_n)

        svm_demo = SVC(kernel="linear", C=1)
        svm_demo.fit(X_demo_svm, y_demo_svm)

        x_line = np.linspace(0, 7, 200)
        w = svm_demo.coef_[0]
        b = svm_demo.intercept_[0]
        y_boundary = -(w[0] * x_line + b) / w[1]
        y_margin_pos = -(w[0] * x_line + b - 1) / w[1]
        y_margin_neg = -(w[0] * x_line + b + 1) / w[1]

        fig_svm_demo = go.Figure()
        fig_svm_demo.add_scatter(x=X_demo_a[:, 0], y=X_demo_a[:, 1], mode="markers",
                                  marker=dict(color="blue", size=6), name="クラス A")
        fig_svm_demo.add_scatter(x=X_demo_b[:, 0], y=X_demo_b[:, 1], mode="markers",
                                  marker=dict(color="red", size=6), name="クラス B")
        fig_svm_demo.add_scatter(x=x_line, y=y_boundary, mode="lines",
                                  line=dict(color="black", width=3), name="決定境界 f(x)=0")
        fig_svm_demo.add_scatter(x=x_line, y=y_margin_pos, mode="lines",
                                  line=dict(color="green", dash="dash"), name="マージン f(x)=+1")
        fig_svm_demo.add_scatter(x=x_line, y=y_margin_neg, mode="lines",
                                  line=dict(color="green", dash="dash"), name="マージン f(x)=-1")
        fig_svm_demo.update_layout(title="SVM: 決定境界とマージン",
                                    xaxis_title="特徴量1", yaxis_title="特徴量2",
                                    height=660, showlegend=True,
                                    xaxis=dict(range=[0, 7], scaleanchor="y", scaleratio=1),
                                    yaxis=dict(range=[0, 7]))
        fig_svm_demo.add_annotation(x=3.2, y=4.1, text="決定境界",
                                     showarrow=True, ax=-40, ay=-40, font=dict(size=12))
        fig_svm_demo.add_annotation(x=2.9, y=4.9, text="マージン境界",
                                     showarrow=True, ax=-50, ay=-10, font=dict(size=12, color="green"))
        _plotly_chart(fig_svm_demo)

    with col2:
        st.markdown("#### k-means クラスタリングのイメージ")
        # k-means デモ
        X_km_demo = np.vstack([
            rng.normal([1, 1], 0.5, (30, 2)),
            rng.normal([4, 1], 0.5, (30, 2)),
            rng.normal([2.5, 4], 0.5, (30, 2)),
        ])
        km_demo = KMeans(n_clusters=3, random_state=42, n_init=KMEANS_N_INIT)
        labels_demo = km_demo.fit_predict(X_km_demo)
        centers = km_demo.cluster_centers_

        fig_km_demo = px.scatter(x=X_km_demo[:, 0], y=X_km_demo[:, 1],
                                  color=[f"クラスタ {l}" for l in labels_demo],
                                  title="k-means: 最近傍クラスタへの割当")
        _add_centroid_markers(fig_km_demo, centers[:, 0], centers[:, 1], name="重心")
        fig_km_demo.update_layout(height=620, xaxis_title="特徴量1", yaxis_title="特徴量2")
        _plotly_chart(fig_km_demo)

    # Hume-Rothery則の説明
    st.header("5.3 Hume-Rothery 則と HEA")
    st.markdown(r"""
    **Hume-Rothery 則** は、合金が固溶体を形成する条件を経験的にまとめたものである。
    機械学習で再現することで、経験則の妥当性を定量的に検証できる。

    | パラメータ | 記号 | 固溶体形成条件 | 物理的意味 |
    |:---|:---|:---|:---|
    | 原子半径差 | `r_delta` ($\delta$) | $\delta < 6.6\%$（比率表記なら 0.066） | 格子歪みが小さい |
    | VEC | VEC | BCC: < 6.87, FCC: > 8.0 | 電子構造が相安定性を支配 |
    | 電気陰性度差 | $\Delta\chi$ | 小さいほど良い | 化合物形成傾向が低い |
    | 混合エンタルピー | $\Delta H_{mix}$ | $-11.6 < \Delta H < 3.2$ kJ/mol | 負に大きいと化合物化 |

    $$
    \delta = \sqrt{\sum_{i=1}^{n} c_i \left(1 - \frac{r_i}{\bar{r}}\right)^2} \times 100\%
    $$
    $c_i$ は元素 $i$ の組成比、$r_i$ は原子半径、$\bar{r}$ は平均原子半径である。アプリ内ではこの原子半径差を `r_delta` と表記する。`r_delta` が大きいほど原子サイズの不一致が大きく、格子ひずみが大きいと読む。なお、このCSVの `r_delta` は比率表記であり、0.066 は 6.6% に相当する。

    $$
    VEC = \sum_{i=1}^{n} c_i \cdot VEC_i
    $$
    VEC は価電子濃度の組成平均である。電子構造に関係する量であり、BCC/FCCの相安定性を考える手がかりになる。
    """)

    # データ表示
    st.header("5.4 VEC–r_delta(δ) 空間で見るHEAデータ")
    _cls_features = ["VEC", "r_delta", "D_elec_nega", "Hmix (kJ/mol)", "Tm (K)"]
    st.dataframe(df_hea[["Alloy", "S_Phase"] + _cls_features].head(15), use_container_width=True)

    cls_counts = df_hea["S_Phase"].value_counts()
    fig_cls_dist = px.bar(x=cls_counts.index, y=cls_counts.values,
                          title="相のクラス分布（実データ）",
                          labels={"x": "相", "y": "データ数"},
                          color=cls_counts.index)
    fig_cls_dist.update_layout(height=300)
    _plotly_chart(fig_cls_dist)

    # VEC vs Phase の可視化
    st.markdown("#### VEC–r_delta 空間で Hume-Rothery 則を見る手順")
    st.markdown("""
    ここで行っているのは、機械学習モデルの学習ではなく、**経験則を2次元の記述子空間に重ねて確認する手法**である。

    1. 各合金を `VEC` と `r_delta` の2つの特徴量で表す。  
    2. 点の色を実際の相ラベル `S_Phase` にする。  
    3. VECの経験的な境界値（例: 6.87, 8.0）を縦線で重ねる。  
    4. 相ラベルが境界や原子半径差の大小に沿って分かれるかを観察する。  

    つまり、この図は「Hume-Rothery則がこのHEAデータでも説明力を持つか」を視覚的に確認するための図である。
    """)
    fig_hume = px.scatter(df_hea, x="VEC", y="r_delta", color="S_Phase",
                           title="VEC vs 原子半径差 r_delta（δ, 色: 相）",
                           labels={"r_delta": "原子半径差 r_delta (δ)"})
    fig_hume.add_vline(x=6.87, line_dash="dash", line_color="gray",
                        annotation_text="VEC=6.87 (BCC/FCC境界)")
    fig_hume.add_vline(x=8.0, line_dash="dash", line_color="gray",
                        annotation_text="VEC=8.0")
    fig_hume.update_layout(height=640)
    _plotly_chart(fig_hume)

    # SVM 分類
    st.header("5.5 SVM による相分類")
    st.markdown(r"""
    **SVM** は、クラス間の**マージンを最大化**する超平面を見つける：
    $$
    \min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i
    \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i
    $$

    **カーネル関数の役割**は、元の特徴量空間で直線的に分けにくいデータを、より分けやすい高次元空間で比較することである。実際に高次元座標を全部作るのではなく、サンプル間の類似度 $K(\mathbf{x}, \mathbf{x}')$ を使って計算するため、**カーネルトリック**と呼ばれる。

    | カーネル | 境界のイメージ | 使いどころ |
    |:---|:---|:---|
    | linear | 直線・平面 | 単純で解釈しやすい基準モデル |
    | rbf | 滑らかな曲線 | 非線形な相境界を表したい場合 |
    | poly | 多項式曲面 | 特徴量間の相互作用を表したい場合 |
    """)

    X_cls = df_hea[_cls_features].values
    y_cls = np.array(df_hea["S_Phase"].tolist())
    X_tr_c_raw, X_te_c_raw, y_tr_c, y_te_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    scaler_cls = StandardScaler()
    X_tr_c = scaler_cls.fit_transform(X_tr_c_raw)
    X_te_c = scaler_cls.transform(X_te_c_raw)

    svm_kernel = st.selectbox("SVM カーネル", ["rbf", "linear", "poly"])
    svm_C = st.slider("C（誤分類ペナルティ）", 0.1, 100.0, 10.0, key="svm_cls_c")

    svc = SVC(kernel=svm_kernel, C=svm_C, random_state=42)
    svc.fit(X_tr_c, y_tr_c)
    y_pred_cls = svc.predict(X_te_c)
    acc = accuracy_score(y_te_c, y_pred_cls)
    st.metric("テスト正解率 (Accuracy)", f"{acc:.4f}")

    cm = confusion_matrix(y_te_c, y_pred_cls, labels=sorted(set(y_cls)))
    fig_cm = px.imshow(cm, x=sorted(set(y_cls)), y=sorted(set(y_cls)),
                       text_auto=True, title="混同行列",
                       labels={"x": "予測", "y": "実際"}, color_continuous_scale="Blues")
    fig_cm.update_layout(height=560)
    _plotly_chart(fig_cm)

    st.subheader("SVM分類境界の可視化（PCA 2次元）")
    st.markdown("5次元特徴量のままでは境界を直接描けないため、標準化した特徴量をPCAで2次元に落としてから、同じカーネルのSVMを学習し、分類境界を可視化する。これは説明用の近似図であり、上の正解率とは別計算である。")
    pca_svm_vis = PCA(n_components=2)
    X_cls_2d = pca_svm_vis.fit_transform(np.vstack([X_tr_c, X_te_c]))
    y_cls_all = np.concatenate([y_tr_c, y_te_c])
    svc_2d = SVC(kernel=svm_kernel, C=svm_C, random_state=42)
    svc_2d.fit(X_cls_2d, y_cls_all)
    x_min, x_max = X_cls_2d[:, 0].min() - 0.6, X_cls_2d[:, 0].max() + 0.6
    y_min, y_max = X_cls_2d[:, 1].min() - 0.6, X_cls_2d[:, 1].max() + 0.6
    xx_svm, yy_svm = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
    grid_pred = svc_2d.predict(np.c_[xx_svm.ravel(), yy_svm.ravel()]).reshape(xx_svm.shape)
    phase_order = sorted(set(y_cls_all))
    phase_to_int = {phase: i for i, phase in enumerate(phase_order)}
    z_int = np.vectorize(phase_to_int.get)(grid_pred)
    fig_svm_vis = go.Figure()
    fig_svm_vis.add_contour(
        x=np.linspace(x_min, x_max, 180),
        y=np.linspace(y_min, y_max, 180),
        z=z_int,
        showscale=False,
        opacity=0.28,
        contours=dict(showlines=False),
        name="分類領域",
    )
    for phase in phase_order:
        mask = y_cls_all == phase
        fig_svm_vis.add_scatter(
            x=X_cls_2d[mask, 0], y=X_cls_2d[mask, 1],
            mode="markers", name=phase,
            marker=dict(size=5, opacity=0.72),
        )
    fig_svm_vis.update_layout(
        title=f"SVM分類境界の可視化（PCA 2次元, kernel={svm_kernel}）",
        xaxis_title=f"PC1 ({pca_svm_vis.explained_variance_ratio_[0] * 100:.1f}%)",
        yaxis_title=f"PC2 ({pca_svm_vis.explained_variance_ratio_[1] * 100:.1f}%)",
        height=760,
    )
    _plotly_chart(fig_svm_vis)

    # k-means
    st.header("5.6 k-means クラスタリング")
    st.markdown(r"""
    **k-means** は教師なし学習で、データを $K$ 個のクラスタに分割する手法である。

    $$
    J = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2
    $$

    この式の $J$ は **WCSS (Within-Cluster Sum of Squares; クラスタ内平方和)** である。各点 $\mathbf{x}$ と、その点が属するクラスタ重心 $\boldsymbol{\mu}_k$ との距離の二乗和を表す。$J$ が小さいほど、各クラスタの中で点が重心近くにまとまっている。

    **エルボー法**では、クラスタ数 $K$ を増やしたときの WCSS の下がり方を見る。最初は大きく下がり、ある点から改善が小さくなる。その折れ曲がり、つまり「肘」に見える点を候補のクラスタ数とする。

    **シルエットスコア**は、クラスタリング結果の良さを評価する指標である。同じクラスタ内の近さと、他クラスタからの遠さを合わせて評価する。値はおよそ -1 から 1 の範囲で、1に近いほどよく分離している。

    ただし、k-meansは各クラスタを「重心の周りに丸くまとまった点群」として扱うため、細長いクラスタ、曲がったクラスタ、密度が大きく異なるクラスタには向かない。この教材では、k-meansの考え方を理解するための基本例として扱う。

    1つのデータ点について、同じクラスタ内の平均距離を $a$、最も近い別クラスタへの平均距離を $b$ とすると、
    $$
    s = \frac{b-a}{\max(a,b)}
    $$
    である。$s \approx 1$ ならよく分離、$s \approx 0$ なら境界付近、$s < 0$ なら別クラスタに入れた方がよい可能性を示す。
    """)

    phase_order_km = sorted(set(y_cls))
    default_km = min(max(2, len(phase_order_km)), 8)
    st.info(f"HEAデータの相ラベルは {phase_order_km} の {len(phase_order_km)} 種類である。まずはクラスタ数 k = {default_km} を基準にすると、教師なし分類が相ラベルにどの程度対応するかを見やすい。")
    n_clusters_km = st.slider("クラスタ数 k", 2, 8, default_km, key="km_cls")
    scaler_km = StandardScaler()
    X_cls_scaled = scaler_km.fit_transform(X_cls)

    pca_cls = PCA(n_components=2)
    X_cls_pca = pca_cls.fit_transform(X_cls_scaled)

    st.subheader("k-meansの数ステップ実演")
    km_steps = st.slider("表示する反復回数", 1, 8, 3, key="km_steps_demo")
    km_step_model = KMeans(
        n_clusters=n_clusters_km,
        random_state=42,
        n_init=1,
        init="k-means++",
        max_iter=km_steps,
    )
    km_step_labels = km_step_model.fit_predict(X_cls_scaled)
    centers_step_pca = pca_cls.transform(km_step_model.cluster_centers_)
    sil_step = silhouette_score(
        X_cls_scaled, km_step_labels,
        sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(X_cls_scaled)),
        random_state=42,
    )
    st.metric("表示ステップでのシルエットスコア", f"{sil_step:.4f}")

    fig_km_step = px.scatter(
        x=X_cls_pca[:, 0], y=X_cls_pca[:, 1],
        color=[f"Cluster {l}" for l in km_step_labels],
        title=f"k-means: {km_steps} 回の反復後の割当（PCA空間）",
        labels={"x": f"PC1 ({pca_cls.explained_variance_ratio_[0] * 100:.1f}%)",
                "y": f"PC2 ({pca_cls.explained_variance_ratio_[1] * 100:.1f}%)"},
    )
    _add_centroid_markers(fig_km_step, centers_step_pca[:, 0], centers_step_pca[:, 1], name="クラスタ重心")
    fig_km_step.update_layout(height=760, xaxis=dict(scaleanchor="y", scaleratio=1))
    _plotly_chart(fig_km_step)
    st.caption("PCA空間の点群自体が正方形になるわけではない。図の縦横比をそろえることで距離感を読みやすくしているだけである。")

    km = KMeans(n_clusters=n_clusters_km, random_state=42, n_init=KMEANS_N_INIT)
    km_labels = km.fit_predict(X_cls_scaled)
    sil = silhouette_score(
        X_cls_scaled, km_labels,
        sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(X_cls_scaled)),
        random_state=42
    )
    st.metric("収束後のシルエットスコア", f"{sil:.4f}")

    fig_km = px.scatter(x=X_cls_pca[:, 0], y=X_cls_pca[:, 1],
                        color=[f"Cluster {l}" for l in km_labels],
                        title="k-means クラスタリング結果（収束後・PCA空間）",
                        labels={"x": f"PC1 ({pca_cls.explained_variance_ratio_[0] * 100:.1f}%)",
                                "y": f"PC2 ({pca_cls.explained_variance_ratio_[1] * 100:.1f}%)"})
    centers_pca = pca_cls.transform(km.cluster_centers_)
    _add_centroid_markers(fig_km, centers_pca[:, 0], centers_pca[:, 1], name="重心")
    fig_km.update_layout(height=760, xaxis=dict(scaleanchor="y", scaleratio=1))
    _plotly_chart(fig_km)

    crosstab_km = pd.crosstab(pd.Series(df_hea["S_Phase"], name="実際の相"),
                              pd.Series(km_labels, name="クラスタ"))
    st.markdown("#### 実際の相ラベルとk-meansクラスタの対応")
    st.dataframe(crosstab_km, use_container_width=True)

    # エルボー法
    K_range = range(2, 11)
    wcss = []
    sil_scores = []
    for k in K_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=KMEANS_N_INIT)
        km_temp.fit(X_cls_scaled)
        wcss.append(km_temp.inertia_)
        sil_scores.append(silhouette_score(
            X_cls_scaled, km_temp.labels_,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(X_cls_scaled)),
            random_state=42
        ))

    fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
    fig_elbow.add_scatter(x=list(K_range), y=wcss, mode="lines+markers",
                         name="WCSS（小さいほどまとまる）", secondary_y=False)
    fig_elbow.add_scatter(x=list(K_range), y=sil_scores, mode="lines+markers",
                         name="シルエットスコア（大きいほど分離）", secondary_y=True)
    fig_elbow.update_layout(title="エルボー法 + シルエットスコア", height=700)
    fig_elbow.update_xaxes(title_text="クラスタ数 k")
    fig_elbow.update_yaxes(title_text="WCSS", secondary_y=False)
    fig_elbow.update_yaxes(title_text="シルエットスコア", secondary_y=True)
    _plotly_chart(fig_elbow)
    st.caption("WCSSは Within-Cluster Sum of Squares の略であり、クラスタ内のばらつきの総量を表す。WCSSはクラスタ数を増やすほど必ず小さくなるため、単に最小値を見るのではなく、改善が鈍くなる点を探す。シルエットスコアはクラスタリング結果の良さを評価する指標であり、クラスタ数選択の別の目安になる。ただし、k-meansは細長いクラスタには向かない点にも注意が必要である。")


# =====================================================================
# セクション 6: 交差検証・汎化性能
# =====================================================================
elif section_key == "cv_generalization":
    st.title("🔄 交差検証と汎化性能評価")
    st.markdown(f"**使用データ**: {dataset_desc}")
    st.markdown("---")

    # 交差検証のイメージ図
    st.header("6.1 交差検証のイメージ")
    st.markdown(r"""
    ### k-fold 交差検証の概念図
    """)

    # k-fold visualization
    fig_cv_img = go.Figure()
    k_demo = 5
    colors_fold = px.colors.qualitative.Set2
    for fold_i in range(k_demo):
        for j in range(k_demo):
            color = "red" if j == fold_i else "steelblue"
            label = "テスト" if j == fold_i else "訓練"
            fig_cv_img.add_shape(type="rect",
                                 x0=j, y0=k_demo - fold_i - 1, x1=j + 0.9, y1=k_demo - fold_i - 0.1,
                                 fillcolor=color, opacity=0.7, line=dict(width=1))
            fig_cv_img.add_annotation(x=j + 0.45, y=k_demo - fold_i - 0.5,
                                       text=label, showarrow=False,
                                       font=dict(size=10, color="white"))
    fig_cv_img.update_layout(
        title="5-fold 交差検証: 5つの分割をすべて表示",
        xaxis=dict(title="データブロック", tickvals=[i + 0.45 for i in range(k_demo)],
                   ticktext=[f"Block {i+1}" for i in range(k_demo)], range=[-0.05, 5.05]),
        yaxis=dict(title="フォールド", tickvals=[i + 0.5 for i in range(k_demo)],
                   ticktext=[f"Fold {i+1}" for i in range(k_demo)], range=[-0.05, 5.05]),
        height=560, width=880, autosize=False, showlegend=False,
        margin=dict(l=80, r=40, t=80, b=80),
    )
    _plotly_chart_fixed(fig_cv_img)
    st.caption("5-foldでは5回の学習・評価を行う。図は5行で、各行の赤いブロックがその回のテストデータ、残り4ブロックが訓練データである。")

    st.markdown(r"""
    ### なぜ必要か？
    テストデータに対する性能は「たまたま」良い/悪い可能性がある。
    CVはデータの分割を複数回行い、性能の安定性を評価する。

    ### k-fold 交差検証
    $$
    \overline{\text{Score}} = \frac{1}{k}\sum_{i=1}^{k}\text{Score}_i
    \quad \pm \quad s = \sqrt{\frac{1}{k-1}\sum_{i=1}^{k}(\text{Score}_i - \overline{\text{Score}})^2}
    $$

    ### LOOCV (Leave-One-Out)
    $k = n$ の極端なケース。バイアスが小さいが計算コスト大。
    """)

    st.header("6.2 交差検証の実行")
    X = df_main[feature_cols].values
    y = df_main[target_col].values

    cv_model_name = st.selectbox("モデル", ["線形回帰", "Ridge", "Lasso", "SVR", "Random Forest"])
    cv_method = st.selectbox("CV 方法", ["k-fold CV", "LOOCV"])

    if cv_model_name == "線形回帰":
        model_cv = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    elif cv_model_name == "Ridge":
        model_cv = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    elif cv_model_name == "Lasso":
        model_cv = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1, max_iter=10000))])
    elif cv_model_name == "SVR":
        model_cv = Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10))])
    else:
        model_cv = Pipeline([("model", RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=N_JOBS))])

    if cv_method == "k-fold CV":
        k = st.slider("フォールド数 k", 2, 20, 5)
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        scoring = "r2"
    else:
        cv = LeaveOneOut()
        scoring = "neg_mean_squared_error"
        st.warning(f"LOOCV: データ数 {len(X)} 回の学習を行う。")

    with st.spinner("交差検証を実行中..."):
        scores = cross_val_score(model_cv, X, y, cv=cv, scoring=scoring, n_jobs=N_JOBS)
        y_pred_cv = cross_val_predict(model_cv, X, y, cv=cv, n_jobs=N_JOBS)

    if cv_method == "LOOCV":
        mse_scores = -scores
        overall_r2 = r2_score(y, y_pred_cv)
        col1, col2, col3 = st.columns(3)
        col1.metric("総合 R²", f"{overall_r2:.4f}")
        col2.metric("平均 RMSE", f"{np.sqrt(mse_scores.mean()):.4f}")
        col3.metric("フォールド数", f"{len(scores)}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("平均 R²", f"{scores.mean():.4f}")
        col2.metric("標準偏差", f"{scores.std():.4f}")
        col3.metric("フォールド数", f"{len(scores)}")
        fig_cv = px.bar(x=[f"Fold {i+1}" for i in range(len(scores))], y=scores,
                        title="各フォールドの R² スコア",
                        labels={"x": "フォールド", "y": "R²"})
        fig_cv.add_hline(y=scores.mean(), line_dash="dash", line_color="red",
                        annotation_text=f"平均: {scores.mean():.4f}")
        fig_cv.update_layout(height=400)
        _plotly_chart(fig_cv)

    st.subheader("交差検証後のパリティプロット")
    cv_parity_r2 = r2_score(y, y_pred_cv)
    cv_parity_rmse = float(np.sqrt(mean_squared_error(y, y_pred_cv)))
    st.caption("各データ点が、自分を含まない訓練フォールドで学習したモデルから予測された値を使う。通常の訓練データ上の予測より、汎化性能の確認に近い図である。")
    parity_df_cv = pd.DataFrame({"実測値": y, "CV予測値": y_pred_cv})
    fig_cv_parity = px.scatter(
        parity_df_cv,
        x="実測値", y="CV予測値",
        title=f"交差検証後のパリティプロット（{cv_model_name}, R²={cv_parity_r2:.4f}, RMSE={cv_parity_rmse:.3f}）",
    )
    min_cv_parity = float(min(np.min(y), np.min(y_pred_cv)))
    max_cv_parity = float(max(np.max(y), np.max(y_pred_cv)))
    fig_cv_parity.add_shape(
        type="line",
        x0=min_cv_parity, y0=min_cv_parity, x1=max_cv_parity, y1=max_cv_parity,
        line=dict(dash="dash", color="red"),
    )
    fig_cv_parity.update_layout(height=700)
    _plotly_chart(fig_cv_parity)

    # 学習曲線
    st.header("6.3 学習曲線")
    st.markdown(r"""
    **学習曲線** はデータ数と性能の関係を可視化：
    - ギャップが大きい → 過学習（データ増加が有効）
    - 両方低い → 未学習（モデル複雑度を上げる）
    """)

    with st.spinner("学習曲線を計算中..."):
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model_cv, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="r2", n_jobs=N_JOBS
        )

    fig_lc = go.Figure()
    fig_lc.add_scatter(x=train_sizes_abs, y=train_scores.mean(axis=1),
                       mode="lines+markers", name="訓練スコア",
                       error_y=dict(type="data", array=train_scores.std(axis=1)))
    fig_lc.add_scatter(x=train_sizes_abs, y=test_scores.mean(axis=1),
                       mode="lines+markers", name="検証スコア",
                       error_y=dict(type="data", array=test_scores.std(axis=1)))
    fig_lc.update_layout(title="学習曲線", xaxis_title="訓練データ数",
                        yaxis_title="R² スコア", height=450)
    _plotly_chart(fig_lc)

    # 検証曲線
    st.header("6.4 検証曲線（ハイパーパラメータ vs 性能）")
    val_model = st.selectbox("モデル（検証曲線）", ["Ridge", "Lasso", "Random Forest (max_depth)"])

    with st.spinner("検証曲線を計算中..."):
        if val_model == "Ridge":
            param_range = np.logspace(-3, 3, 20)
            train_s, test_s = validation_curve(
                Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
                X, y, param_name="model__alpha", param_range=param_range, cv=5, scoring="r2", n_jobs=N_JOBS)
            x_vals = np.log10(param_range)
            param_label = "log₁₀(α)"
        elif val_model == "Lasso":
            param_range = np.logspace(-4, 1, 20)
            train_s, test_s = validation_curve(
                Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10000))]),
                X, y, param_name="model__alpha", param_range=param_range, cv=5, scoring="r2", n_jobs=N_JOBS)
            x_vals = np.log10(param_range)
            param_label = "log₁₀(α)"
        else:
            param_range = np.arange(2, 25)
            train_s, test_s = validation_curve(
                Pipeline([("model", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=N_JOBS))]),
                X, y, param_name="model__max_depth", param_range=param_range, cv=5, scoring="r2", n_jobs=N_JOBS)
            x_vals = param_range
            param_label = "max_depth"

    fig_vc = go.Figure()
    fig_vc.add_scatter(x=x_vals, y=train_s.mean(axis=1), mode="lines+markers", name="訓練スコア")
    fig_vc.add_scatter(x=x_vals, y=test_s.mean(axis=1), mode="lines+markers", name="検証スコア")
    fig_vc.update_layout(title=f"検証曲線: {val_model}",
                        xaxis_title=param_label, yaxis_title="R²", height=450)
    _plotly_chart(fig_vc)


# =====================================================================
# セクション 7: 正則化・モデル選択
# =====================================================================
elif section_key == "regularization":
    st.title("⚖️ 正則化とモデル選択")
    st.markdown(f"**使用データ**: {dataset_desc}")
    st.markdown("---")

    X = df_main[feature_cols].values
    y = df_main[target_col].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    st.header("7.1 正則化の理論")
    st.markdown(r"""
    正則化は、モデルの複雑度にペナルティを課して**過学習を防止**する手法である。

    ### Ridge 回帰（L2 正則化）
    $$
    \hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}}
    \left[ \sum_{i=1}^{n}(y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2
    + \alpha \sum_{j=1}^{p} \beta_j^2 \right]
    $$
    - 閉形式の解: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
    - 係数を**縮小**するが0にはしない

    ### Lasso 回帰（L1 正則化）
    $$
    \hat{\boldsymbol{\beta}}_{\text{Lasso}} = \arg\min_{\boldsymbol{\beta}}
    \left[ \sum_{i=1}^{n}(y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2
    + \alpha \sum_{j=1}^{p} |\beta_j| \right]
    $$
    - 係数を**完全に0にする**（スパース解 → 特徴量選択）

    ### α の意味
    - $\alpha = 0$: 正則化なし = 通常の最小二乗法
    - $\alpha \to \infty$: すべての係数が0（定数モデル）
    """)

    st.header("7.2 Ridge vs Lasso の比較")
    alpha_val = st.slider("正則化パラメータ α", 0.001, 100.0, 1.0, step=0.1)

    ridge = Ridge(alpha=alpha_val)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    lasso = Lasso(alpha=alpha_val, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    r2_lasso = r2_score(y_test, y_pred_lasso)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ridge 回帰")
        st.metric("R²", f"{r2_ridge:.4f}")
        coef_ridge = pd.DataFrame({"特徴量": feature_cols, "係数": ridge.coef_})
        fig_r = px.bar(coef_ridge, x="特徴量", y="係数", title="Ridge 係数",
                       color="係数", color_continuous_scale="RdBu_r")
        fig_r.update_layout(height=350)
        _plotly_chart(fig_r)

    with col2:
        st.subheader("Lasso 回帰")
        st.metric("R²", f"{r2_lasso:.4f}")
        n_zero = (np.abs(lasso.coef_) < 1e-10).sum()
        st.markdown(f"**0 になった係数**: {n_zero} / {len(feature_cols)}")
        coef_lasso = pd.DataFrame({"特徴量": feature_cols, "係数": lasso.coef_})
        fig_l = px.bar(coef_lasso, x="特徴量", y="係数", title="Lasso 係数",
                       color="係数", color_continuous_scale="RdBu_r")
        fig_l.update_layout(height=350)
        _plotly_chart(fig_l)

    st.header("7.3 正則化パス（α vs 係数）")
    alphas = np.logspace(-2, 2, 40)
    ridge_coefs, lasso_coefs = _cached_regularization_paths(X_train, y_train, alphas)

    # 7.2 で選んだαがパス上のどこに当たるかを、破線で示す。
    # パス曲線そのものはαスライダーに依存しないが、この破線はスライダーに追従して動く。
    # αがパスの範囲外のときは、軸が伸びないよう端にそろえる。
    log_alpha_val = float(np.clip(np.log10(alpha_val),
                                  np.log10(alphas.min()), np.log10(alphas.max())))

    col1, col2 = st.columns(2)
    with col1:
        fig_rp = go.Figure()
        for j, name in enumerate(feature_cols):
            fig_rp.add_scatter(x=np.log10(alphas), y=ridge_coefs[:, j], mode="lines", name=name)
        fig_rp.add_vline(x=log_alpha_val, line_dash="dash", line_color="gray",
                         annotation_text="7.2で選んだα", annotation_position="top")
        fig_rp.update_layout(title="Ridge 正則化パス", xaxis_title="log₁₀(α)",
                            yaxis_title="係数", height=400)
        _plotly_chart(fig_rp)

    with col2:
        fig_lp = go.Figure()
        for j, name in enumerate(feature_cols):
            fig_lp.add_scatter(x=np.log10(alphas), y=lasso_coefs[:, j], mode="lines", name=name)
        fig_lp.add_vline(x=log_alpha_val, line_dash="dash", line_color="gray",
                         annotation_text="7.2で選んだα", annotation_position="top")
        fig_lp.update_layout(title="Lasso 正則化パス", xaxis_title="log₁₀(α)",
                            yaxis_title="係数", height=400)
        _plotly_chart(fig_lp)

    st.info(r"""
    **正則化パスの読み方**
    - **曲線そのものは α スライダーには依存しない**。各特徴量の係数が α 全体でどう変わるかを一度に描いたものである。
    - 灰色の破線は 7.2 で選んだ α の位置を示す。スライダーを動かすと破線が左右に動き、その位置での係数が上の 7.2 の棒グラフに対応する。
    - **Ridge**: α が大きくなると全係数が徐々に 0 に近づく（縮小）
    - **Lasso**: α が大きくなると係数が順次 **完全に 0** になる（特徴量選択）
    - Lasso で最後まで残る特徴量が最も重要な特徴量
    """)


# =====================================================================
# セクション 8: データ増強
# =====================================================================
elif section_key == "data_augmentation":
    st.title("📈 データ増強 (Data Augmentation)")
    st.markdown(f"**使用データ**: {dataset_desc}")
    st.markdown("---")

    st.header("8.1 データ増強とは")
    st.markdown(r"""
    **データ増強** は、既存のデータから新しいサンプルを生成してデータセットを拡大する手法である。
    ただし、材料データでは安易な増強が必ず精度向上につながるわけではない。実験していない点を人工的に増やしているだけなので、評価の信用性そのものを高める方法ではない。

    ### なぜデータ増強が必要か？
    - 材料実験は時間・コストがかかる（1サンプル数日〜数週間）
    - 少量データでは過学習しやすい
    - クラス不均衡（例: 超伝導体 vs 非超伝導体）の解消

    ### 材料データに適した増強手法

    | 手法 | 種類 | 説明 | 増強量の目安 |
    |:---|:---|:---|:---|
    | **ガウスノイズ付加** | 回帰 | 特徴量に微小ノイズ $\mathcal{N}(0, \sigma^2)$ を加える | 2〜5倍 |
    | **SMOTE** | 分類 | 少数クラスの近傍間で線形補間 | 少数クラスを多数クラスと同数に |
    | **Mixup** | 両方 | 2サンプル間の凸結合 $\tilde{x} = \lambda x_i + (1-\lambda)x_j$ | 2〜10倍 |
    | **ブートストラップ** | 両方 | 復元抽出による再サンプリング | 任意 |
    | **物理制約付き生成** | 材料特有 | 質量保存則・電荷中性等を満たす組成を生成 | 任意 |
    """)

    st.header("8.2 ガウスノイズ付加の実演")
    st.markdown(r"""
    最も単純な増強法。各特徴量 $x_j$ に対して：
    $$
    \tilde{x}_j = x_j + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
    $$
    $\sigma$ はデータの標準偏差の一定割合（通常 1〜10%）に設定。

    この実演では、測定済みの目的変数 $y$ は変えず、入力特徴量だけを少し揺らす。
    目的変数まで人工的に揺らすと、学生には「正解ラベルも作ってよい」と誤解されやすいためである。

    **重要:** ノイズレベルや増強倍率を変えてもR²がほとんど変わらない、あるいは悪化する場合がある。その結果は失敗ではなく、「この単純な増強は信用性の向上に効いていない」という重要な観察である。
    """)
    st.warning("この節はデータ補強の効果を保証する実演ではない。むしろ、安易なデータ補強を過信しないための確認用である。")

    X = df_main[feature_cols].values
    y = df_main[target_col].values

    noise_level = st.slider("ノイズレベル σ（標準偏差の割合 %）", 1, 20, 5)
    aug_multiplier = st.slider("増強倍率", 1, 10, 3)

    # まず train/test 分割（テストデータは増強しない — データリーク防止）
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # 訓練データのみにノイズ付加で増強
    rng = np.random.default_rng(42)
    X_aug_list = [X_tr]
    y_aug_list = [y_tr]
    for _ in range(aug_multiplier - 1):
        noise = rng.normal(0, noise_level / 100.0, X_tr.shape) * X_tr.std(axis=0)
        # 組成・密度・温度など負値が物理的に不自然な特徴量が多いため、下限0でクリップする。
        X_noisy = np.clip(X_tr + noise, a_min=0, a_max=None)
        X_aug_list.append(X_noisy)
        y_aug_list.append(y_tr.copy())

    X_tr_aug = np.vstack(X_aug_list)
    y_tr_aug = np.concatenate(y_aug_list)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("訓練データ数（元）", f"{len(X_tr)}")
    with col2:
        st.metric("訓練データ数（増強後）", f"{len(X_tr_aug)}")

    # 増強前後のモデル性能比較
    st.header("8.3 増強前後のモデル性能比較")

    # 元データ
    pipe_orig = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipe_orig.fit(X_tr, y_tr)
    r2_orig = r2_score(y_te, pipe_orig.predict(X_te))

    # 増強データ（テストは同じ X_te, y_te で評価）
    pipe_aug = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipe_aug.fit(X_tr_aug, y_tr_aug)
    r2_aug = r2_score(y_te, pipe_aug.predict(X_te))

    fig_compare = go.Figure()
    fig_compare.add_bar(x=["元データ", "増強後"], y=[r2_orig, r2_aug],
                        marker_color=["steelblue", "coral"],
                        text=[f"{r2_orig:.4f}", f"{r2_aug:.4f}"], textposition="auto")
    fig_compare.update_layout(title="データ増強前後の R² 比較 (Ridge回帰)",
                              yaxis_title="R²", height=420)
    _plotly_chart(fig_compare)

    delta_r2_aug = r2_aug - r2_orig
    st.metric("R²の変化量", f"{delta_r2_aug:+.4f}")
    if delta_r2_aug > 0.02:
        st.success("この条件ではR²が少し改善している。ただし、別の分割や別データで再確認する必要がある。")
    elif delta_r2_aug < -0.02:
        st.warning("この条件ではR²が悪化している。ノイズ付加が物理的な意味を壊している可能性がある。")
    else:
        st.info("R²はほとんど変化していない。この場合、単純なガウスノイズ付加によって予測の信用性が上がったとは言いにくい。")

    st.subheader("ノイズレベルを変えた感度確認")
    sensitivity_levels = [0, 1, 3, 5, 10, 20]
    sensitivity_rows = []
    for nl in sensitivity_levels:
        if nl == 0:
            X_tr_tmp, y_tr_tmp = X_tr, y_tr
        else:
            rng_tmp = np.random.default_rng(42)
            X_tmp_list = [X_tr]
            y_tmp_list = [y_tr]
            for _ in range(max(1, aug_multiplier) - 1):
                noise_tmp = rng_tmp.normal(0, nl / 100.0, X_tr.shape) * X_tr.std(axis=0)
                X_tmp_list.append(np.clip(X_tr + noise_tmp, a_min=0, a_max=None))
                y_tmp_list.append(y_tr.copy())
            X_tr_tmp = np.vstack(X_tmp_list)
            y_tr_tmp = np.concatenate(y_tmp_list)
        pipe_tmp = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        pipe_tmp.fit(X_tr_tmp, y_tr_tmp)
        sensitivity_rows.append({
            "ノイズレベル(%)": nl,
            "訓練データ数": len(X_tr_tmp),
            "R²": r2_score(y_te, pipe_tmp.predict(X_te)),
        })
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    fig_sens = px.line(sensitivity_df, x="ノイズレベル(%)", y="R²", markers=True, title="ノイズレベルに対するR²の感度")
    fig_sens.update_layout(height=520)
    _plotly_chart(fig_sens)
    st.dataframe(sensitivity_df.round(4), use_container_width=True)

    st.header("8.4 SMOTE（分類問題用）")
    st.markdown(r"""
    **SMOTE (Synthetic Minority Over-sampling Technique)** は、少数クラスのサンプル間で
    線形補間し、新しいサンプルを生成する：

    $$
    \tilde{\mathbf{x}} = \mathbf{x}_i + \lambda (\mathbf{x}_{nn} - \mathbf{x}_i), \quad \lambda \sim U(0, 1)
    $$

    - $\mathbf{x}_i$: 少数クラスのサンプル
    - $\mathbf{x}_{nn}$: $\mathbf{x}_i$ の k-近傍から選んだサンプル

    **注意点**:
    - 過度な増強は少数クラスの情報を薄める（overfitting to interpolation）
    - 材料データでは物理的に不合理な組成が生成される可能性がある
    - 後処理で物理制約をチェックすることが望ましい
    """)

    st.header("8.5 Mixup（組成補間）")
    st.markdown(r"""
    **Mixup** は2つのサンプルの凸結合を新しいサンプルとする手法：

    $$
    \tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda) \mathbf{x}_j
    $$
    $$
    \tilde{y} = \lambda y_i + (1-\lambda) y_j
    $$

    - $\lambda \sim \text{Beta}(\alpha, \alpha)$、通常 $\alpha = 0.2$
    - 材料の**組成空間での補間**に対応（合金 A と合金 B の中間組成）
    - 物理的意味：Vegard 則（格子定数の線形則）に対応

    **材料データでの利点**:
    - 組成空間の探索範囲を拡大
    - 凸結合なので質量保存を自動的に満たす（wt%の場合）
    """)

    st.info("""
    **まとめ: データ増強の指針**
    - 回帰問題 → ガウスノイズ付加 or Mixup が安全
    - 分類問題（不均衡） → SMOTE が効果的
    - 増強倍率: 2〜5倍が一般的。過度な増強は逆効果
    - 必ず**テストデータは増強しない**（データリーク防止）
    - 材料データ特有: 物理制約（質量保存、電荷中性）をチェック
    """)


# =====================================================================
# セクション 9: 特徴量生成
# =====================================================================
elif section_key == "feature_generation":
    st.title("🧪 特徴量生成 — matminer / MAGPIE")
    st.markdown("---")

    st.header("9.1 特徴量生成とは")
    st.markdown(r"""
    **特徴量生成** は、化学式や組成から、機械学習に入力できる数値列を作る操作である。
    材料分野では、元素の原子量、原子半径、電気陰性度、融点などを組成平均・範囲・偏差としてまとめることが多い。

    代表例が **MAGPIE特徴量** である。MAGPIEでは、各元素の性質をデータベースから取り出し、
    組成比で重み付けした平均、ばらつき、最大最小差などを計算する。

    例として、元素性質 $p_i$ に対して次のような特徴量を作れる。

    **組成加重平均:**
    $$
    \overline{p} = \sum_i c_i p_i
    $$

    **範囲:**
    $$
    \mathrm{range}(p) = \max_i(p_i) - \min_i(p_i)
    $$

    **平均絶対偏差:**
    $$
    \mathrm{avg\_dev}(p) = \sum_i c_i |p_i - \overline{p}|
    $$

    **原子半径差:**
    $$
    r_{\delta} = \sqrt{\sum_i c_i \left(1 - \frac{r_i}{\bar{r}}\right)^2} \times 100
    $$

    ここで、$c_i$ は元素 $i$ の組成比、$p_i$ は原子量・原子半径・電気陰性度・融点などの元素物性である。
    化学式をそのまま機械学習に入れるのではなく、こうした数値特徴量に変換する点が重要である。
    """)

    st.info("講義時間内で動くことを優先し、既定では生成する特徴量と対象行数を少量に絞っている。matminerが入っていない環境でも、内蔵の軽量MAGPIE風特徴量で実演できる。")

    st.header("9.2 化学式から特徴量を生成する")
    st.markdown(
        "化学式（組成）から、機械学習に使える数値特徴量を生成する。"
        "入力は手入力・組み込みのHEA例・自分のCSVから選べる。"
        "HEAに限らず、合金・酸化物など任意の材料系の化学式を対象にできる。"
    )

    feature_source = st.radio(
        "化学式の入力方法",
        ["手入力", "HEAデータ（例）", "CSVから読み込む"],
        horizontal=True,
    )

    # 回帰用の目的変数テーブル（formula列 + 数値目的変数列）。無ければ None のまま。
    target_table = None
    # 再学習用に出力へ組み入れる元データ全体（formula列 + アップロード/元データの全列）。
    source_table = None

    if feature_source == "HEAデータ（例）":
        max_formulas = st.slider("処理する化学式数", 5, 50, 12,
                                 help="matminerを使う場合、処理数が多いと講義時間内に終わらない可能性がある。")
        formulas = df_hea["Alloy"].astype(str).head(max_formulas).tolist()
        st.dataframe(pd.DataFrame({"formula": formulas}), use_container_width=True)
        _hea_target_cols = [
            c for c in df_hea.columns
            if c not in ("Alloy", "S_Phase") and is_numeric_dtype(df_hea[c])
        ]
        _hea_head = df_hea.assign(Alloy=df_hea["Alloy"].astype(str)).head(max_formulas)
        target_table = _hea_head[["Alloy"] + _hea_target_cols].rename(columns={"Alloy": "formula"})
        # 元データ（HEA）の全列を、再学習用の結合出力のために保持する。
        source_table = _hea_head.rename(columns={"Alloy": "formula"})

    elif feature_source == "CSVから読み込む":
        fg_csv = st.file_uploader(
            "化学式の列を含むCSV", type=["csv", "tsv", "txt"], key="fg_csv_uploader",
            help="化学式（組成）の列を1つ含むCSV。数値の目的変数列があれば回帰にも使える。",
        )
        if fg_csv is None:
            st.info("化学式の列を含むCSVをアップロードする。目的変数の数値列があれば、下の回帰にも使える。")
            formulas = []
        else:
            try:
                df_up = _parse_uploaded_csv(fg_csv.getvalue(), fg_csv.name)
            except Exception as exc:
                st.error(f"CSVの読み込みに失敗した: {exc}")
                df_up = None

            if df_up is None or len(df_up.columns) == 0:
                formulas = []
            else:
                df_up.columns = [str(c) for c in df_up.columns]
                # 化学式列の推定: formula/composition/alloy/組成 などの名前を優先する。
                _lower = [c.lower() for c in df_up.columns]
                _guess = 0
                for _kw in ("formula", "composition", "alloy", "化学式", "組成"):
                    _hits = [i for i, name in enumerate(_lower) if _kw in name]
                    if _hits:
                        _guess = _hits[0]
                        break
                formula_col = st.selectbox(
                    "化学式の列", df_up.columns.tolist(), index=_guess, key="fg_csv_formula_col"
                )
                _max_rows = int(min(200, len(df_up)))
                max_formulas = st.slider(
                    "処理する化学式数", 5, max(5, _max_rows), min(20, _max_rows),
                    key="fg_csv_max_formulas",
                )
                numeric_up = [c for c in df_up.columns if c != formula_col and is_numeric_dtype(df_up[c])]
                target_up_cols = st.multiselect(
                    "目的変数に使う数値列（回帰用・任意）", numeric_up,
                    default=numeric_up[:1], key="fg_csv_target_cols",
                )
                sub = df_up.head(max_formulas).copy()
                sub[formula_col] = sub[formula_col].astype(str)
                formulas = sub[formula_col].tolist()
                st.dataframe(sub[[formula_col] + target_up_cols].head(20), use_container_width=True)
                if target_up_cols:
                    target_table = sub[[formula_col] + target_up_cols].rename(columns={formula_col: "formula"})
                # アップロードした全列を保持する（再学習用の結合出力に使う）。
                source_table = sub.rename(columns={formula_col: "formula"})
                source_table = source_table[
                    ["formula"] + [c for c in source_table.columns if c != "formula"]
                ]

    else:  # 手入力
        formula_text = st.text_area(
            "化学式を1行に1つずつ入力",
            value="FeNi\nAlCoCrFeNi\nCoCrFeMnNi\nAg2Al\nAg5Cd8",
            height=160,
        )
        formulas = [line.strip() for line in formula_text.splitlines() if line.strip()]

    use_matminer = st.checkbox("matminer の MAGPIE 特徴量を試す（インストール済み環境のみ）", value=False)
    max_generated_features = st.slider("生成する特徴量数の上限", 6, 20, 12, help="計算量と表示の見やすさを考え、少数に絞る。")

    if st.button("特徴量を生成する", type="primary"):
        if not formulas:
            st.error("化学式が入力されていない。")
            st.session_state.pop("fg_result", None)
        else:
            if use_matminer:
                result, err = _matminer_magpie_features(formulas, max_features=max_generated_features)
                if err is not None:
                    st.warning(err)
                    st.info("授業中はこのまま内蔵の軽量MAGPIE風特徴量を使えばよい。matminer/pymatgen が必要なPCだけ、後でインストールする運用が安全である。")
                    with st.expander("matminer/pymatgen を使いたい場合のインストール例", expanded=False):
                        st.code("python -m pip install matminer pymatgen", language="bash")
                    features_generated, skipped = _fallback_magpie_features(formulas)
                    method_note = "内蔵の軽量MAGPIE風特徴量"
                else:
                    features_generated, skipped = result
                    method_note = "matminer の MAGPIE 特徴量"
                    st.success("matminer の MAGPIE 特徴量を生成した。")
            else:
                features_generated, skipped = _fallback_magpie_features(formulas)
                method_note = "内蔵の軽量MAGPIE風特徴量"
                st.success("内蔵の軽量MAGPIE風特徴量を生成した。")

            # 生成結果を保持し、後続の回帰でも使えるようにする。
            # （Streamlitではボタンのifブロックは押した瞬間しか実行されないため、
            #   目的変数やモデルを選ぶたびに消えないよう session_state に置く。）
            st.session_state["fg_result"] = {
                "features": features_generated,
                "skipped": skipped,
                "source": feature_source,
                "method": method_note,
                "target_table": target_table,
                "source_table": source_table,
            }

    fg_result = st.session_state.get("fg_result")
    if fg_result is not None:
        features_generated = fg_result["features"]
        skipped = fg_result["skipped"]
        numeric_generated = [
            c for c in features_generated.columns
            if c != "formula" and is_numeric_dtype(features_generated[c])
        ]

        if len(features_generated) == 0:
            st.error("特徴量を生成できる化学式がなかった。元素記号や組成表記を確認すること。")
        else:
            st.subheader("生成された特徴量")
            st.caption(f"生成方法: {fg_result['method']}")
            st.dataframe(features_generated.round(4), use_container_width=True)

            # 元データ（アップロードしたCSV/HEA例）の全列に生成特徴量を組み入れ、
            # そのまま改めて機械学習にかけられる「分析用データセット」を作る。
            source_table_dl = fg_result.get("source_table")
            combined_dataset = None
            if source_table_dl is not None and "formula" in source_table_dl.columns:
                _src = (
                    source_table_dl.assign(formula=source_table_dl["formula"].astype(str))
                    .drop_duplicates(subset="formula")
                )
                _gen = features_generated.assign(formula=features_generated["formula"].astype(str))
                # 生成特徴量名が元データの列と衝突する場合は、生成側に "_gen" を付けて元データ列を保つ。
                combined_dataset = _src.merge(_gen, on="formula", how="right", suffixes=("", "_gen"))
                # 化学式を先頭列にする（元データ列 → 生成特徴量 の並び）。
                combined_dataset = combined_dataset[
                    ["formula"] + [c for c in combined_dataset.columns if c != "formula"]
                ]

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "生成特徴量のみをCSVとして保存",
                    _csv_bytes(features_generated),
                    file_name="generated_features.csv",
                    mime="text/csv",
                    key="download_generated_features",
                    help="化学式と生成特徴量だけのCSV。",
                )
            with dl_col2:
                if combined_dataset is not None:
                    st.download_button(
                        "元データ＋生成特徴量のCSVを保存",
                        _csv_bytes(combined_dataset),
                        file_name="dataset_with_features.csv",
                        mime="text/csv",
                        key="download_combined_dataset",
                        help="アップロードした元データ（全列）に生成特徴量を組み入れた、そのまま機械学習にかけられるデータセット。",
                    )
                else:
                    st.caption(
                        "目的変数を含むCSVを読み込むか「HEAデータ（例）」を選ぶと、"
                        "元データ＋生成特徴量を結合したCSVも保存できる。"
                    )

            if combined_dataset is not None:
                with st.expander("結合後データセットのプレビュー", expanded=False):
                    st.dataframe(combined_dataset.head(20).round(4), use_container_width=True)
                    st.caption(
                        f"列数 {combined_dataset.shape[1]}（元データの全列 + 生成特徴量）。"
                        "このCSVは「4. 回帰問題」などの外部CSV読み込み機能にそのまま渡せる。"
                    )

            if len(numeric_generated) >= 2:
                display_generated = numeric_generated[:min(6, len(numeric_generated))]
                fig_feat = px.scatter_matrix(
                    features_generated[["formula"] + display_generated],
                    dimensions=display_generated,
                    hover_name="formula",
                    title=f"生成特徴量のペアプロット（先頭{len(display_generated)}特徴量）",
                    height=max(820, 190 * len(display_generated)),
                    width=max(820, 190 * len(display_generated)),
                )
                fig_feat.update_traces(diagonal_visible=True, marker=dict(size=6))
                fig_feat.update_layout(autosize=False)
                _plotly_chart_fixed(fig_feat)

        if skipped is not None and len(skipped) > 0:
            with st.expander("生成できなかった化学式", expanded=False):
                st.dataframe(skipped, use_container_width=True)

        # -------------------------------------------------------------
        # 生成した特徴量を入力に使って回帰する
        # -------------------------------------------------------------
        if len(features_generated) > 0:
            st.markdown("---")
            st.subheader("生成した特徴量で回帰する")
            st.markdown(
                "生成した特徴量を入力（説明変数）として、材料物性を予測する回帰を試す。"
                "目的変数には、組み込みHEA例の物性、または読み込んだCSVの数値列を使える。"
                "**化学式 → 特徴量生成 → 回帰** という一連の流れを、この画面だけで体験できる。"
            )

            reg_target_table = fg_result.get("target_table")
            _avail_target_cols = (
                [c for c in reg_target_table.columns
                 if c != "formula" and is_numeric_dtype(reg_target_table[c])]
                if reg_target_table is not None else []
            )

            if not _avail_target_cols:
                st.info(
                    "回帰には、各化学式に対応する目的変数（物性値）が必要である。"
                    "「HEAデータ（例）」を選ぶか、目的変数の数値列を含むCSVを読み込んで特徴量を生成すること。"
                    "手入力の化学式には物性値が紐づかないため、回帰はできない。"
                )
            elif len(numeric_generated) < 1:
                st.warning("数値の生成特徴量が無いため、回帰を実行できない。")
            else:
                # 目的変数列が生成特徴量と同名だと merge で衝突するため、内部名を退避する。
                _gen_cols = set(c for c in features_generated.columns if c != "formula")
                _safe_map = {
                    c: (f"__tgt__{c}" if c in _gen_cols else c) for c in _avail_target_cols
                }
                tt = reg_target_table[["formula"] + _avail_target_cols].rename(columns=_safe_map)
                tt = tt.assign(formula=tt["formula"].astype(str)).drop_duplicates(subset="formula")
                merged = (
                    features_generated.assign(formula=features_generated["formula"].astype(str))
                    .merge(tt, on="formula", how="inner")
                )

                if len(merged) < 5:
                    st.warning(
                        f"回帰に使える対応データが {len(merged)} 件と少ない。"
                        "「処理する化学式数」を増やす、または目的変数列を含むCSVを使うこと（20件以上を推奨）。"
                    )
                else:
                    ctrl1, ctrl2 = st.columns(2)
                    with ctrl1:
                        target_display = st.selectbox(
                            "目的変数",
                            _avail_target_cols,
                            index=len(_avail_target_cols) - 1,
                            key="fg_reg_target",
                        )
                        target_gen = _safe_map[target_display]
                    with ctrl2:
                        model_gen_name = st.radio(
                            "回帰モデル",
                            ["Ridge（線形）", "Random Forest"],
                            horizontal=True,
                            key="fg_reg_model",
                        )

                    X_gen_df = merged[numeric_generated].apply(pd.to_numeric, errors="coerce")
                    valid_mask = X_gen_df.notna().all(axis=1) & merged[target_gen].notna()
                    X_gen = X_gen_df[valid_mask].values
                    y_gen = merged.loc[valid_mask, target_gen].values
                    formulas_gen = merged.loc[valid_mask, "formula"].values
                    n_gen = len(y_gen)

                    if n_gen < 5:
                        st.warning(
                            f"欠損を除くと {n_gen} 件しか残らない。化学式数を増やすこと。"
                        )
                    else:
                        if model_gen_name.startswith("Ridge"):
                            model_gen = Pipeline([
                                ("scaler", StandardScaler()),
                                ("model", Ridge(alpha=1.0)),
                            ])
                        else:
                            model_gen = RandomForestRegressor(
                                n_estimators=200, random_state=42, n_jobs=N_JOBS
                            )

                        # サンプルが少ないので、交差検証で誠実に汎化性能を測る。
                        n_splits = min(5, n_gen)
                        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                        y_cv_pred = cross_val_predict(model_gen, X_gen, y_gen, cv=kf, n_jobs=N_JOBS)
                        r2_gen = r2_score(y_gen, y_cv_pred)
                        rmse_gen = float(np.sqrt(mean_squared_error(y_gen, y_cv_pred)))

                        mcol = st.columns(3)
                        mcol[0].metric("サンプル数", f"{n_gen}")
                        mcol[1].metric(f"CV R²（{n_splits}-fold）", f"{r2_gen:.3f}")
                        mcol[2].metric("CV RMSE", f"{rmse_gen:.3g}")

                        parity_df = pd.DataFrame({
                            "実測値": y_gen,
                            "予測値（CV）": y_cv_pred,
                            "formula": formulas_gen,
                        })
                        fig_parity = px.scatter(
                            parity_df, x="実測値", y="予測値（CV）", hover_name="formula",
                            title=f"{target_display} のパリティプロット（交差検証予測）",
                        )
                        lo = float(min(y_gen.min(), y_cv_pred.min()))
                        hi = float(max(y_gen.max(), y_cv_pred.max()))
                        fig_parity.add_scatter(
                            x=[lo, hi], y=[lo, hi], mode="lines",
                            line=dict(dash="dash", color="gray"), name="y = x",
                        )
                        fig_parity.update_layout(height=460)
                        _plotly_chart(fig_parity)

                        if not model_gen_name.startswith("Ridge"):
                            rf_fit = RandomForestRegressor(
                                n_estimators=200, random_state=42, n_jobs=N_JOBS
                            ).fit(X_gen, y_gen)
                            imp_df = pd.DataFrame({
                                "特徴量": numeric_generated,
                                "重要度": rf_fit.feature_importances_,
                            }).sort_values("重要度", ascending=False)
                            fig_imp = px.bar(
                                imp_df, x="重要度", y="特徴量", orientation="h",
                                title="Random Forest 特徴量重要度",
                            )
                            fig_imp.update_layout(
                                height=max(320, 26 * len(numeric_generated)),
                                yaxis=dict(autorange="reversed"),
                            )
                            _plotly_chart(fig_imp)

                        st.caption(
                            "注意: サンプル数が少ないと評価は不安定になる。"
                            "また、組成から直接決まる量（HEA例の電気陰性度差やVECなど）は"
                            "生成特徴量からほぼ再現できるため R² が高く出やすい。"
                            "実測物性（強度・臨界温度・バンドギャップなど）を目的変数にすると、"
                            "特徴量生成の効果をより正しく確かめられる。"
                        )

    st.header("9.3 matminerを使う場合の注意")
    st.markdown("""
    matminerは便利だが、授業PCではインストールや計算に時間がかかる場合がある。
    講義内では、次の方針が安全である。

    1. 処理する化学式数を 10〜20 件程度に絞る。
    2. 生成する特徴量も 10 個前後に絞る。
    3. 特徴量生成済みCSVを保存し、以後の解析ではそのCSVを使う。
    4. うまく動かないPCでは、内蔵の軽量MAGPIE風特徴量を使う。
    """)


# =====================================================================
# セクション 10: まとめ + レポート課題
# =====================================================================
elif section_key == "summary_assignments":
    st.title("📝 まとめとレポート課題")
    st.markdown("---")

    st.header("10.1 本日のまとめ")
    st.markdown(r"""
    ### マテリアルズ・インフォマティクスの一連のワークフローを体験した

    | ステップ | 手法 | キーポイント |
    |:---|:---|:---|
    | **データ探索** | 要約統計量、ペアプロット、外れ値検出 | データの質と特性を把握することが最重要 |
    | **次元削減** | PCA (Biplot) | 高次元データの可視化と冗長性の除去 |
    | **回帰** | 線形回帰、多項式回帰、SVR、Random Forest | 物性値の定量予測 |
    | **分類** | SVM / k-means | 材料のカテゴリ分類（Hume-Rothery則） |
    | **交差検証** | k-fold CV / LOOCV / 学習曲線 | 汎化性能の信頼性評価 |
    | **データ増強** | ノイズ付加 / SMOTE / Mixup | 少量データへの対処と限界の確認 |
    | **特徴量生成** | matminer / MAGPIE | 化学式・組成を機械学習用の数値特徴量へ変換 |
    """)

    st.header("10.2 使用したデータセット")
    st.markdown("""
    | データ | 出典 | 件数 | 用途 |
    |:---|:---|:---|:---|
    | 鉄鋼 降伏強度 | Citrine Informatics / Matbench | 312 | 回帰（構造材料） |
    | 超伝導体 臨界温度 | NIMS SuperCon / UCI | 500 | 回帰（機能材料） |
    | HEA 相分類 | Zenodo ACHIEF | 1,103 | 分類（Hume-Rothery則） |
    """)

    st.markdown("---")
    st.header("📋 レポート課題")
    st.error("以下の課題から **2つ** を選択し、レポートとして提出すること。")
    st.info("同じ課題内容は、配布フォルダ内の `report_assignment.html` でも確認できる。")

    st.markdown(r"""
    ---

    ### 課題 1: 回帰分析による物性予測（基礎）

    **テーマ**: 鉄鋼または超伝導体データを使い、物性値を予測するモデルを構築せよ。

    **要件**:
    1. データの探索的分析（要約統計量、相関行列、PCA biplot）を行い、データの特徴を説明せよ
    2. 外れ値検出を行い、結果を材料科学的に考察せよ
    3. 線形回帰、SVR、Random Forest を適用し、R²・RMSE・パリティプロットを比較せよ
    4. 線形回帰の R² が負になった場合、平均値予測との比較を用いて理由を説明せよ
    5. 5-fold CV の結果を報告し、モデルの安定性を議論せよ

    ---

    ### 課題 2: 過学習の理解と可視化（基礎〜応用）

    **テーマ**: 多項式回帰を用いて過学習現象を可視化・分析せよ。

    **要件**:
    1. 1変数の特徴量を選び、多項式の次数 $d = 1, 3, 5, 10, 15$ でフィッティングを行え
    2. 訓練誤差とテスト誤差をグラフ化し、過学習が起きる次数を特定せよ
    3. バイアス-バリアンス分解の考え方を用いて結果を説明せよ
    4. CSV編集画面で一部の値を変え、外れ値や測定値の変化が回帰結果へ与える影響を確認せよ
    5. 学習曲線を作成し、データ数が増えた場合の効果を議論せよ

    ---

    ### 課題 3: Hume-Rothery則の機械学習による再現（応用）

    **テーマ**: HEAデータからHume-Rothery則を機械学習で再現し、材料設計指針を議論せよ。

    **要件**:
    1. HEAデータの探索的分析を行え（VEC, δ, ΔH_mix の分布と相との関係）
    2. SVM（kernel: linear, rbf, poly）で SS/IM/AM を分類し、正解率と混同行列を比較せよ
    3. k-means クラスタリングで教師なし分離が可能か議論せよ
    4. VEC と δ の閾値を調べ、Hume-Rothery則との整合性を議論せよ
    5. 新しい合金を設計する場合、どの特徴量をどう制御すべきか提案せよ

    ---

    ### 課題 4: 特徴量重要度と材料科学的解釈（応用）

    **テーマ**: Random Forest の特徴量重要度と相関・PCAを組み合わせ、重要な説明変数を解釈せよ。

    **要件**:
    1. 鉄鋼または超伝導体データを用い、Random Forest の特徴量重要度を算出せよ
    2. 重要度上位3特徴量を示し、それぞれの物理的意味を説明せよ
    3. 相関行列と PCA biplot を用い、重要特徴量が他の特徴量とどのような関係にあるか説明せよ
    4. 重要特徴量だけで材料設計指針を作る場合の利点と限界を考察せよ
    5. 機械学習のスコアだけでなく、材料科学的に妥当な説明になっているか検討せよ

    ---

    ### 課題 5: MI ワークフローの総合実践（発展）

    **テーマ**: 授業配布データまたは外部データを用いて、MIの一連のワークフローを実践せよ。

    **要件**:
    1. データ取得またはデータ選択、前処理を行え
    2. 要約統計量、相関行列、PCA biplot、外れ値検出を行え
    3. 回帰または分類モデルを使い、予測・分類結果を評価せよ
    4. 交差検証により汎化性能を評価せよ
    5. 得られた結果から、材料設計指針を提案せよ

    **注意**: レポート課題では、結果の解釈、可視化、材料科学的考察を重視する。

    ---
    """)

    st.header("10.3 レポート作成のガイドライン")
    st.markdown("""
    ### 構成
    1. **目的**: 何を予測/分類し、なぜそれが重要か
    2. **データ**: 使用データの概要、出典、前処理
    3. **手法**: 使用した手法とその数式
    4. **結果**: 図表を用いた定量的な結果
    5. **考察**: 結果の材料科学的解釈
    6. **結論**: 得られた知見のまとめ

    ### 提出期限
    講義日から **2週間後** の講義開始前まで
    """)

    st.markdown("---")
    st.markdown("""
    ### 参考文献
    1. Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
    2. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An Introduction to Statistical Learning*. Springer.
    3. Ramprasad, R., et al. (2017). "Machine learning in materials informatics." *npj Computational Materials*, 3, 54.
    4. Ward, L., et al. (2016). "A general-purpose machine learning framework for predicting properties of inorganic materials." *npj Computational Materials*, 2, 16028.
    5. Rickman, J.M., et al. (2019). "Materials informatics for the screening of multi-principal elements and high-entropy alloys." *Nature Communications*, 10, 2618.
    """)
