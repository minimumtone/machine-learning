"""
マテリアルズ・インフォマティクス (MI) 講義用アプリケーション
============================================================
対象: 大学3回生 材料工学専攻 初心者向け
想定: 講義1コマで一連のMIワークフローを体験

セクション構成:
  1. MIとは
  2. データ探索
  3. 次元削減 PCA (Biplot)
  4. 回帰問題
  5. 分類問題（Hume-Rothery則）
  6. 交差検証・汎化性能評価
  7. 正則化・モデル選択
  8. データ増強
  9. まとめ + レポート課題

データセット（全て実データ）:
  - 鉄鋼: matminer steel_strength (Citrine Informatics, 312件)
  - 超伝導体: UCI/NIMS SuperCon (21,263件 → 500件抽出)
  - HEA相分類: Zenodo ACHIEF project (1,103件)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict, KFold, LeaveOneOut,
    learning_curve, validation_curve
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
    classification_report, confusion_matrix, silhouette_score,
    accuracy_score
)
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MI講義: マテリアルズ・インフォマティクス入門",
    page_icon="🔬",
    layout="wide",
)

# Matplotlib 日本語フォント設定
try:
    matplotlib.font_manager.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
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

# ---------------------------------------------------------------------------
# データ読み込み（実データ）
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_steel_data():
    """鉄鋼の機械的特性 — Citrine Informatics 実験データ (312件)
    出典: matminer steel_strength / Matbench"""
    df = pd.read_csv(DATA_DIR / "steel_strength.csv")
    return df


@st.cache_data
def load_superconductor_data():
    """超伝導体の臨界温度 — UCI/NIMS SuperCon (500件抽出)
    出典: NIMS supercon.nims.go.jp, UCI ML Repository"""
    df = pd.read_csv(DATA_DIR / "superconductor_500.csv")
    return df


@st.cache_data
def load_hea_data():
    """高エントロピー合金の相分類 — Zenodo ACHIEF (1,103件)
    出典: doi:10.5281/zenodo.5155150"""
    df = pd.read_csv(DATA_DIR / "HEA_phases.csv")
    return df


# ---------------------------------------------------------------------------
# サイドバー: セクション選択
# ---------------------------------------------------------------------------
st.sidebar.title("📚 MI 講義ナビゲーション")
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
    "9. まとめ＋レポート課題": "summary_assignments",
}

selected = st.sidebar.radio("セクションを選択", list(SECTIONS.keys()))
section_key = SECTIONS[selected]

st.sidebar.markdown("---")
st.sidebar.markdown("**使用データセット**")
dataset_choice = st.sidebar.selectbox(
    "回帰用データ",
    ["鉄鋼（構造材料）", "超伝導体（機能材料）"],
)

# データ読み込み
if dataset_choice == "鉄鋼（構造材料）":
    df_main = load_steel_data()
    target_col = "yield strength"
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
else:
    df_main = load_superconductor_data()
    target_col = "critical_temp"
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

feature_cols = [c for c in df_main.columns if c != target_col]
# Remove non-numeric columns and co-measured response variables from features
_response_cols = {"tensile strength", "elongation"}  # co-measured outputs, not input features
feature_cols = [c for c in feature_cols if df_main[c].dtype in ['float64', 'int64', 'float32', 'int32'] and c not in _response_cols]

df_hea = load_hea_data()


# ---------------------------------------------------------------------------
# ユーティリティ: データ要約表示
# ---------------------------------------------------------------------------
def show_data_summary(df, features, target, desc, detail):
    """回帰/分類セクション前にデータ要約を表示"""
    st.subheader("📋 データセットの概要")
    st.markdown(f"**{desc}**")
    with st.expander("データセットの詳細説明", expanded=False):
        st.markdown(detail)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 要約統計量")
        st.dataframe(df[features + [target]].describe().round(3), use_container_width=True)
    with col2:
        st.markdown("#### PCA Biplot（データ構造の概観）")
        X_sum = df[features].values
        scaler_sum = StandardScaler()
        X_sum_scaled = scaler_sum.fit_transform(X_sum)
        n_comp = min(2, len(features))
        pca_sum = PCA(n_components=n_comp)
        scores = pca_sum.fit_transform(X_sum_scaled)
        if n_comp >= 2:
            fig_bp = _create_biplot(scores, pca_sum, features,
                                    pca_sum.explained_variance_ratio_, target_values=df[target].values)
            fig_bp.update_layout(height=350, width=350)
            st.plotly_chart(fig_bp, use_container_width=True)
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
        height=500, width=500,
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
    材料科学にデータ科学・機械学習の手法を融合させた学際的分野です。

    従来の材料開発は **「経験と勘」** に頼る試行錯誤が中心でしたが、
    MIではデータ駆動型のアプローチにより、材料探索・設計・最適化を加速します。

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
        構築した第一原理計算データベースです。

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
    **本アプリでは以下の実データを使用します：**

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
    機械学習の目標は、入力 $\mathbf{x}$ から出力 $y$ への写像 $f$ を学習することです。

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
    R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
    $$

    $R^2 = 1$ で完全な予測、$R^2 = 0$ で平均値予測と同等。
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

    st.header("2.2 要約統計量")
    st.markdown(r"""
    **要約統計量** はデータの全体像を把握するための基本指標です。

    | 指標 | 数式 | 意味 |
    |:---|:---|:---|
    | 平均 | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$ | データの中心 |
    | 分散 | $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$ | データの散らばり |
    | 標準偏差 | $s = \sqrt{s^2}$ | 分散の平方根（元の単位） |
    | 中央値 | $\tilde{x}$ | データを並べた中央の値 |
    | 四分位範囲 | $\text{IQR} = Q_3 - Q_1$ | 外れ値検出の基準 |
    """)
    st.dataframe(df_main.describe().round(3), use_container_width=True)

    st.header("2.3 分布の可視化（ヒストグラム）")
    hist_col = st.selectbox("表示する変数", df_main.columns.tolist())
    fig_hist = px.histogram(df_main, x=hist_col, nbins=30, marginal="box",
                            title=f"{hist_col} の分布")
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.header("2.4 相関行列")
    st.markdown(r"""
    **ピアソン相関係数** は2変数間の線形関係の強さを表します：

    $$
    r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
    {\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2 \sum_{i=1}^{n}(y_i-\bar{y})^2}}
    $$
    """)
    corr = df_main[feature_cols + [target_col]].corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="相関行列ヒートマップ")
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.header("2.5 ペアプロット（散布図行列）")
    pair_cols = st.multiselect("表示する変数（2〜4個推奨）",
                               df_main.columns.tolist(),
                               default=feature_cols[:3] + [target_col])
    if len(pair_cols) >= 2:
        fig_pair = px.scatter_matrix(df_main[pair_cols], dimensions=pair_cols,
                                     height=600, title="ペアプロット")
        fig_pair.update_traces(diagonal_visible=True, marker=dict(size=3))
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.warning("2つ以上の変数を選択してください。")

    st.header("2.6 異常データの検出（外れ値検出）")
    st.markdown(r"""
    材料データには測定ミスや特殊条件のデータが含まれることがあります。

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
    outlier_col = st.selectbox("対象変数", df_main.columns.tolist(),
                               index=len(df_main.columns) - 1)

    df_outlier = df_main.copy()
    if outlier_method == "IQR法":
        Q1 = df_outlier[outlier_col].quantile(0.25)
        Q3 = df_outlier[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_outlier["外れ値"] = ((df_outlier[outlier_col] < lower) |
                              (df_outlier[outlier_col] > upper))
        st.markdown(f"IQR = {IQR:.2f}, 下限 = {lower:.2f}, 上限 = {upper:.2f}")
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
    st.plotly_chart(fig_out, use_container_width=True)


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
    射影して次元を削減する手法です。

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

    st.header("3.2 PCA の実行")
    X = df_main[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(len(feature_cols), X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    col1, col2 = st.columns(2)
    with col1:
        fig_var = go.Figure()
        fig_var.add_bar(x=[f"PC{i+1}" for i in range(len(explained))],
                        y=explained * 100, name="寄与率")
        fig_var.add_scatter(x=[f"PC{i+1}" for i in range(len(cumulative))],
                           y=cumulative * 100, name="累積寄与率",
                           mode="lines+markers")
        fig_var.update_layout(title="寄与率と累積寄与率", yaxis_title="寄与率 (%)",
                             height=400)
        st.plotly_chart(fig_var, use_container_width=True)

    with col2:
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=feature_cols,
        )
        fig_load = px.imshow(loadings.iloc[:, :min(4, n_components)],
                             text_auto=".2f",
                             color_continuous_scale="RdBu_r",
                             title="主成分負荷量（各特徴量の寄与）")
        fig_load.update_layout(height=400)
        st.plotly_chart(fig_load, use_container_width=True)

    # Biplot（正方形）
    st.header("3.3 Biplot（スコア＋負荷量ベクトル）")
    if n_components >= 2:
        fig_biplot = _create_biplot(
            X_pca, pca, feature_cols, explained,
            target_values=df_main[target_col].values
        )
        st.plotly_chart(fig_biplot, use_container_width=True)

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

    # データ要約の表示
    show_data_summary(df_main, feature_cols, target_col, dataset_desc, dataset_detail)

    X = df_main[feature_cols].values
    y = df_main[target_col].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 4.1 線形回帰
    st.header("4.1 線形回帰 (Linear Regression)")
    st.markdown(r"""
    最も基本的な回帰モデルです。目的変数 $y$ を特徴量の**線形結合**で表現します：

    $$
    \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p = \mathbf{x}^T \boldsymbol{\beta}
    $$

    ### 最小二乗法 (OLS)
    $$
    \boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
    $$
    """)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{r2_lr:.4f}")
    col2.metric("RMSE", f"{rmse_lr:.2f}")
    col3.metric("MAE", f"{mae_lr:.2f}")

    fig_lr = px.scatter(x=y_test, y=y_pred_lr,
                        labels={"x": f"実測値 ({target_col})", "y": "予測値"},
                        title=f"線形回帰: 予測 vs 実測 (R² = {r2_lr:.4f})")
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    fig_lr.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                     line=dict(dash="dash", color="red"))
    fig_lr.update_layout(height=450)
    st.plotly_chart(fig_lr, use_container_width=True)

    coef_df = pd.DataFrame({
        "特徴量": feature_cols,
        "係数 β": lr.coef_,
        "|β|": np.abs(lr.coef_),
    }).sort_values("|β|", ascending=False)
    fig_coef = px.bar(coef_df, x="特徴量", y="係数 β", title="回帰係数（標準化後）",
                      color="係数 β", color_continuous_scale="RdBu_r")
    fig_coef.update_layout(height=350)
    st.plotly_chart(fig_coef, use_container_width=True)

    # 4.2 多項式回帰と過学習
    st.header("4.2 多項式回帰と過学習")
    st.markdown(r"""
    **多項式回帰**は特徴量の高次の項を追加して非線形性を表現します：

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

    X_demo = df_main[demo_feature].values.reshape(-1, 1)
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

    col1, col2 = st.columns(2)
    with col1:
        selected_degrees = st.multiselect("表示する次数", degrees, default=[1, 3, max_degree])
        x_plot = np.linspace(X_demo.min(), X_demo.max(), 200).reshape(-1, 1)
        fig_fit = go.Figure()
        fig_fit.add_scatter(x=X_tr_d.ravel(), y=y_tr_d, mode="markers",
                           name="訓練データ", marker=dict(size=4, opacity=0.5))
        fig_fit.add_scatter(x=X_te_d.ravel(), y=y_te_d, mode="markers",
                           name="テストデータ", marker=dict(size=4, opacity=0.5, symbol="x"))
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
                               name=f"次数 {d}", line=dict(color=colors[i % len(colors)]))
        fig_fit.update_layout(title="多項式フィッティング",
                             xaxis_title=demo_feature, yaxis_title=target_col, height=450)
        st.plotly_chart(fig_fit, use_container_width=True)

    with col2:
        fig_overfit = go.Figure()
        fig_overfit.add_scatter(x=degrees, y=train_errors, mode="lines+markers", name="訓練誤差 (MSE)")
        fig_overfit.add_scatter(x=degrees, y=test_errors, mode="lines+markers", name="テスト誤差 (MSE)")
        fig_overfit.update_layout(title="過学習の可視化: 次数 vs 誤差",
                                 xaxis_title="多項式の次数", yaxis_title="MSE", height=450)
        st.plotly_chart(fig_overfit, use_container_width=True)

    best_deg = degrees[np.argmin(test_errors)]
    st.success(f"テスト誤差が最小の次数: **{best_deg}** (MSE = {min(test_errors):.2f})")

    # 4.3 SVR
    st.header("4.3 サポートベクター回帰 (SVR)")
    st.markdown(r"""
    **SVR** は、SVM の回帰版。予測値と実測値の差が $\varepsilon$ 以内なら損失を0とする
    **ε-不感損失関数**を使います：

    $$
    \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, |y_i - \hat{y}_i| - \varepsilon)
    $$

    RBF カーネル: $K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$
    """)

    col1, col2 = st.columns(2)
    with col1:
        svr_kernel = st.selectbox("カーネル", ["rbf", "linear", "poly"])
        svr_C = st.slider("C (正則化)", 0.1, 100.0, 10.0, key="svr_c")
    with col2:
        svr_epsilon = st.slider("ε (不感帯)", 0.01, 1.0, 0.1, key="svr_eps")

    svr = SVR(kernel=svr_kernel, C=svr_C, epsilon=svr_epsilon)
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    r2_svr = r2_score(y_test, y_pred_svr)
    rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))

    col1, col2 = st.columns(2)
    col1.metric("SVR R²", f"{r2_svr:.4f}")
    col2.metric("SVR RMSE", f"{rmse_svr:.2f}")

    fig_svr = px.scatter(x=y_test, y=y_pred_svr,
                         labels={"x": f"実測値 ({target_col})", "y": "予測値"},
                         title=f"SVR ({svr_kernel}): R² = {r2_svr:.4f}")
    min_val = min(y_test.min(), y_pred_svr.min())
    max_val = max(y_test.max(), y_pred_svr.max())
    fig_svr.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                      line=dict(dash="dash", color="red"))
    fig_svr.update_layout(height=450)
    st.plotly_chart(fig_svr, use_container_width=True)

    # 4.4 Random Forest
    st.header("4.4 ランダムフォレスト回帰と特徴量重要度")
    st.markdown(r"""
    **ランダムフォレスト** はバギング + 決定木のアンサンブル学習：

    $$
    \hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})
    $$

    **特徴量重要度 (MDI)**: 各特徴量で分岐した時の不純度の減少量の合計
    """)

    col1, col2 = st.columns(2)
    with col1:
        rf_n_estimators = st.slider("木の本数", 10, 500, 100, step=10)
    with col2:
        rf_max_depth = st.slider("最大深さ", 2, 30, 10)

    rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                random_state=42, n_jobs=-1)
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

    col1, col2 = st.columns(2)
    with col1:
        fig_rf = px.scatter(x=y_test, y=y_pred_rf,
                            labels={"x": f"実測値 ({target_col})", "y": "予測値"},
                            title=f"Random Forest: R² = {r2_rf:.4f}")
        min_v = min(y_test.min(), y_pred_rf.min())
        max_v = max(y_test.max(), y_pred_rf.max())
        fig_rf.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                         line=dict(dash="dash", color="red"))
        fig_rf.update_layout(height=400)
        st.plotly_chart(fig_rf, use_container_width=True)
    with col2:
        fig_imp = px.bar(importance_df, x="重要度 (MDI)", y="特徴量",
                         orientation="h", title="特徴量重要度 (MDI)",
                         color="重要度 (MDI)", color_continuous_scale="Viridis")
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)

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

    # イメージ図
    st.header("5.1 分類問題のイメージ")
    st.markdown(r"""
    ### 分類問題とは
    入力 $\mathbf{x}$ を離散的なカテゴリ $y \in \{C_1, C_2, \dots, C_K\}$ に分類する問題です。
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

        xx, yy = np.meshgrid(np.linspace(0, 7, 100), np.linspace(0, 7, 100))
        Z = svm_demo.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig_svm_demo = go.Figure()
        fig_svm_demo.add_contour(x=np.linspace(0, 7, 100), y=np.linspace(0, 7, 100),
                                  z=Z, contours=dict(start=-1, end=1, size=1),
                                  colorscale=[[0, "blue"], [0.5, "white"], [1, "red"]],
                                  showscale=False, opacity=0.3)
        fig_svm_demo.add_scatter(x=X_demo_a[:, 0], y=X_demo_a[:, 1], mode="markers",
                                  marker=dict(color="blue", size=6), name="クラス A")
        fig_svm_demo.add_scatter(x=X_demo_b[:, 0], y=X_demo_b[:, 1], mode="markers",
                                  marker=dict(color="red", size=6), name="クラス B")
        fig_svm_demo.update_layout(title="SVM: マージン最大化",
                                    xaxis_title="特徴量1", yaxis_title="特徴量2",
                                    height=350, showlegend=True)
        fig_svm_demo.add_annotation(x=3.5, y=6.5, text="決定境界",
                                     showarrow=False, font=dict(size=12))
        fig_svm_demo.add_annotation(x=3.5, y=1, text="← マージン →",
                                     showarrow=False, font=dict(size=11, color="green"))
        st.plotly_chart(fig_svm_demo, use_container_width=True)

    with col2:
        st.markdown("#### k-means クラスタリングのイメージ")
        # k-means デモ
        X_km_demo = np.vstack([
            rng.normal([1, 1], 0.5, (30, 2)),
            rng.normal([4, 1], 0.5, (30, 2)),
            rng.normal([2.5, 4], 0.5, (30, 2)),
        ])
        km_demo = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_demo = km_demo.fit_predict(X_km_demo)
        centers = km_demo.cluster_centers_

        fig_km_demo = px.scatter(x=X_km_demo[:, 0], y=X_km_demo[:, 1],
                                  color=[f"クラスタ {l}" for l in labels_demo],
                                  title="k-means: 最近傍クラスタへの割当")
        fig_km_demo.add_scatter(x=centers[:, 0], y=centers[:, 1], mode="markers",
                                 marker=dict(size=15, color="black", symbol="x"),
                                 name="重心", showlegend=True)
        fig_km_demo.update_layout(height=350, xaxis_title="特徴量1", yaxis_title="特徴量2")
        st.plotly_chart(fig_km_demo, use_container_width=True)

    # Hume-Rothery則の説明
    st.header("5.2 Hume-Rothery 則と HEA")
    st.markdown(r"""
    **Hume-Rothery 則** は、合金が固溶体を形成する条件を経験的にまとめたものです。
    機械学習で再現することで、経験則の妥当性を定量的に検証できます。

    | パラメータ | 記号 | 固溶体形成条件 | 物理的意味 |
    |:---|:---|:---|:---|
    | 原子半径差 | $\delta$ | $\delta < 6.6\%$ | 格子歪みが小さい |
    | VEC | VEC | BCC: < 6.87, FCC: > 8.0 | 電子構造が相安定性を支配 |
    | 電気陰性度差 | $\Delta\chi$ | 小さいほど良い | 化合物形成傾向が低い |
    | 混合エンタルピー | $\Delta H_{mix}$ | $-11.6 < \Delta H < 3.2$ kJ/mol | 負に大きいと化合物化 |

    $$
    \delta = \sqrt{\sum_{i=1}^{n} c_i \left(1 - \frac{r_i}{\bar{r}}\right)^2} \times 100\%
    $$

    $$
    VEC = \sum_{i=1}^{n} c_i \cdot VEC_i
    $$
    """)

    # データ表示
    st.header("5.3 HEA データの概要")
    _cls_features = ["VEC", "delta", "D_elec_nega", "Hmix (kJ/mol)", "Tm (K)"]
    st.dataframe(df_hea[["Alloy", "S_Phase"] + _cls_features].head(15), use_container_width=True)

    cls_counts = df_hea["S_Phase"].value_counts()
    fig_cls_dist = px.bar(x=cls_counts.index, y=cls_counts.values,
                          title="相のクラス分布（実データ）",
                          labels={"x": "相", "y": "データ数"},
                          color=cls_counts.index)
    fig_cls_dist.update_layout(height=300)
    st.plotly_chart(fig_cls_dist, use_container_width=True)

    # VEC vs Phase の可視化
    st.markdown("#### VEC vs 相 — Hume-Rothery 則の確認")
    fig_hume = px.scatter(df_hea, x="VEC", y="delta", color="S_Phase",
                           title="VEC vs 原子半径差 δ（色: 相）",
                           labels={"delta": "原子半径差 δ"})
    fig_hume.add_vline(x=6.87, line_dash="dash", line_color="gray",
                        annotation_text="VEC=6.87 (BCC/FCC境界)")
    fig_hume.add_vline(x=8.0, line_dash="dash", line_color="gray",
                        annotation_text="VEC=8.0")
    fig_hume.update_layout(height=450)
    st.plotly_chart(fig_hume, use_container_width=True)

    # SVM 分類
    st.header("5.4 SVM による相分類")
    st.markdown(r"""
    **SVM** は、クラス間の**マージンを最大化**する超平面を見つけます：
    $$
    \min_{\mathbf{w},b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i
    \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i
    $$
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
    svm_C = st.slider("C (SVM正則化)", 0.1, 100.0, 10.0, key="svm_cls_c")

    svc = SVC(kernel=svm_kernel, C=svm_C, random_state=42)
    svc.fit(X_tr_c, y_tr_c)
    y_pred_cls = svc.predict(X_te_c)
    acc = accuracy_score(y_te_c, y_pred_cls)
    st.metric("テスト正解率 (Accuracy)", f"{acc:.4f}")

    cm = confusion_matrix(y_te_c, y_pred_cls, labels=sorted(set(y_cls)))
    fig_cm = px.imshow(cm, x=sorted(set(y_cls)), y=sorted(set(y_cls)),
                       text_auto=True, title="混同行列",
                       labels={"x": "予測", "y": "実際"}, color_continuous_scale="Blues")
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    # k-means
    st.header("5.5 k-means クラスタリング")
    st.markdown(r"""
    **k-means** は教師なし学習で、データを $k$ 個のクラスタに分割します：
    $$
    J = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2
    $$
    """)

    n_clusters_km = st.slider("クラスタ数 k", 2, 8, 4, key="km_cls")
    X_cls_scaled = scaler_cls.fit_transform(X_cls)
    km = KMeans(n_clusters=n_clusters_km, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_cls_scaled)
    sil = silhouette_score(X_cls_scaled, km_labels)
    st.metric("シルエットスコア", f"{sil:.4f}")

    pca_cls = PCA(n_components=2)
    X_cls_pca = pca_cls.fit_transform(X_cls_scaled)
    fig_km = px.scatter(x=X_cls_pca[:, 0], y=X_cls_pca[:, 1],
                        color=[f"Cluster {l}" for l in km_labels],
                        title="k-means クラスタリング結果（PCA空間）")
    fig_km.update_layout(height=400)
    st.plotly_chart(fig_km, use_container_width=True)

    # エルボー法
    K_range = range(2, 11)
    wcss = []
    sil_scores = []
    for k in K_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_temp.fit(X_cls_scaled)
        wcss.append(km_temp.inertia_)
        sil_scores.append(silhouette_score(X_cls_scaled, km_temp.labels_))

    fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
    fig_elbow.add_scatter(x=list(K_range), y=wcss, mode="lines+markers",
                         name="WCSS", secondary_y=False)
    fig_elbow.add_scatter(x=list(K_range), y=sil_scores, mode="lines+markers",
                         name="シルエット", secondary_y=True)
    fig_elbow.update_layout(title="エルボー法 + シルエットスコア", height=400)
    fig_elbow.update_xaxes(title_text="クラスタ数 k")
    fig_elbow.update_yaxes(title_text="WCSS", secondary_y=False)
    fig_elbow.update_yaxes(title_text="シルエットスコア", secondary_y=True)
    st.plotly_chart(fig_elbow, use_container_width=True)


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
        title="5-fold 交差検証: 各フォールドで異なるテストセットを使用",
        xaxis=dict(title="データブロック", tickvals=[i + 0.45 for i in range(k_demo)],
                   ticktext=[f"Block {i+1}" for i in range(k_demo)]),
        yaxis=dict(title="フォールド", tickvals=[i + 0.5 for i in range(k_demo)],
                   ticktext=[f"Fold {i+1}" for i in range(k_demo)]),
        height=350, showlegend=False
    )
    st.plotly_chart(fig_cv_img, use_container_width=True)

    st.markdown(r"""
    ### なぜ必要か？
    テストデータに対する性能は「たまたま」良い/悪い可能性があります。
    CVはデータの分割を複数回行い、性能の安定性を評価します。

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
                                          random_state=42, n_jobs=-1))])

    if cv_method == "k-fold CV":
        k = st.slider("フォールド数 k", 2, 20, 5)
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        scoring = "r2"
    else:
        cv = LeaveOneOut()
        scoring = "neg_mean_squared_error"
        st.warning(f"LOOCV: データ数 {len(X)} 回の学習を行います。")

    with st.spinner("交差検証を実行中..."):
        scores = cross_val_score(model_cv, X, y, cv=cv, scoring=scoring)

    if cv_method == "LOOCV":
        mse_scores = -scores
        y_pred_loocv = cross_val_predict(model_cv, X, y, cv=LeaveOneOut())
        overall_r2 = r2_score(y, y_pred_loocv)
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
        st.plotly_chart(fig_cv, use_container_width=True)

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
            scoring="r2", n_jobs=-1
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
    st.plotly_chart(fig_lc, use_container_width=True)

    # 検証曲線
    st.header("6.4 検証曲線（ハイパーパラメータ vs 性能）")
    val_model = st.selectbox("モデル（検証曲線）", ["Ridge", "Lasso", "Random Forest (max_depth)"])

    with st.spinner("検証曲線を計算中..."):
        if val_model == "Ridge":
            param_range = np.logspace(-3, 3, 20)
            train_s, test_s = validation_curve(
                Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
                X, y, param_name="model__alpha", param_range=param_range, cv=5, scoring="r2", n_jobs=-1)
            x_vals = np.log10(param_range)
            param_label = "log₁₀(α)"
        elif val_model == "Lasso":
            param_range = np.logspace(-4, 1, 20)
            train_s, test_s = validation_curve(
                Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10000))]),
                X, y, param_name="model__alpha", param_range=param_range, cv=5, scoring="r2", n_jobs=-1)
            x_vals = np.log10(param_range)
            param_label = "log₁₀(α)"
        else:
            param_range = np.arange(2, 25)
            train_s, test_s = validation_curve(
                Pipeline([("model", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))]),
                X, y, param_name="model__max_depth", param_range=param_range, cv=5, scoring="r2", n_jobs=-1)
            x_vals = param_range
            param_label = "max_depth"

    fig_vc = go.Figure()
    fig_vc.add_scatter(x=x_vals, y=train_s.mean(axis=1), mode="lines+markers", name="訓練スコア")
    fig_vc.add_scatter(x=x_vals, y=test_s.mean(axis=1), mode="lines+markers", name="検証スコア")
    fig_vc.update_layout(title=f"検証曲線: {val_model}",
                        xaxis_title=param_label, yaxis_title="R²", height=450)
    st.plotly_chart(fig_vc, use_container_width=True)


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
    正則化は、モデルの複雑度にペナルティを課して**過学習を防止**する手法です。

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
        st.plotly_chart(fig_r, use_container_width=True)

    with col2:
        st.subheader("Lasso 回帰")
        st.metric("R²", f"{r2_lasso:.4f}")
        n_zero = (np.abs(lasso.coef_) < 1e-10).sum()
        st.markdown(f"**0 になった係数**: {n_zero} / {len(feature_cols)}")
        coef_lasso = pd.DataFrame({"特徴量": feature_cols, "係数": lasso.coef_})
        fig_l = px.bar(coef_lasso, x="特徴量", y="係数", title="Lasso 係数",
                       color="係数", color_continuous_scale="RdBu_r")
        fig_l.update_layout(height=350)
        st.plotly_chart(fig_l, use_container_width=True)

    st.header("7.3 正則化パス（α vs 係数）")
    alphas = np.logspace(-3, 2, 100)
    ridge_coefs = []
    lasso_coefs = []
    for a in alphas:
        r = Ridge(alpha=a).fit(X_train, y_train)
        ridge_coefs.append(r.coef_)
        la = Lasso(alpha=a, max_iter=10000).fit(X_train, y_train)
        lasso_coefs.append(la.coef_)

    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    col1, col2 = st.columns(2)
    with col1:
        fig_rp = go.Figure()
        for j, name in enumerate(feature_cols):
            fig_rp.add_scatter(x=np.log10(alphas), y=ridge_coefs[:, j], mode="lines", name=name)
        fig_rp.update_layout(title="Ridge 正則化パス", xaxis_title="log₁₀(α)",
                            yaxis_title="係数", height=400)
        st.plotly_chart(fig_rp, use_container_width=True)

    with col2:
        fig_lp = go.Figure()
        for j, name in enumerate(feature_cols):
            fig_lp.add_scatter(x=np.log10(alphas), y=lasso_coefs[:, j], mode="lines", name=name)
        fig_lp.update_layout(title="Lasso 正則化パス", xaxis_title="log₁₀(α)",
                            yaxis_title="係数", height=400)
        st.plotly_chart(fig_lp, use_container_width=True)

    st.info(r"""
    **正則化パスの読み方**
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
    **データ増強** は、既存のデータから新しいサンプルを生成してデータセットを拡大する手法です。
    材料データは取得コストが高く、少量データでのモデル構築が求められるため、データ増強は重要です。

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
    """)

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
        X_aug_list.append(X_tr + noise)
        y_aug_list.append(y_tr + rng.normal(0, y_tr.std() * noise_level / 200.0, y_tr.shape))

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
                              yaxis_title="R²", height=350)
    st.plotly_chart(fig_compare, use_container_width=True)

    st.header("8.4 SMOTE（分類問題用）")
    st.markdown(r"""
    **SMOTE (Synthetic Minority Over-sampling Technique)** は、少数クラスのサンプル間で
    線形補間し、新しいサンプルを生成します：

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
# セクション 9: まとめ + レポート課題
# =====================================================================
elif section_key == "summary_assignments":
    st.title("📝 まとめとレポート課題")
    st.markdown("---")

    st.header("9.1 本日のまとめ")
    st.markdown(r"""
    ### マテリアルズ・インフォマティクスの一連のワークフローを体験しました

    | ステップ | 手法 | キーポイント |
    |:---|:---|:---|
    | **データ探索** | 要約統計量、ペアプロット、外れ値検出 | データの質と特性を把握することが最重要 |
    | **次元削減** | PCA (Biplot) | 高次元データの可視化と冗長性の除去 |
    | **回帰** | 線形回帰、多項式回帰、SVR、Random Forest | 物性値の定量予測 |
    | **分類** | SVM / k-means | 材料のカテゴリ分類（Hume-Rothery則） |
    | **交差検証** | k-fold CV / LOOCV / 学習曲線 | 汎化性能の信頼性評価 |
    | **正則化** | Lasso (L1) / Ridge (L2) | 過学習防止と特徴量選択 |
    | **データ増強** | ノイズ付加 / SMOTE / Mixup | 少量データへの対処 |
    """)

    st.header("9.2 使用したデータセット")
    st.markdown("""
    | データ | 出典 | 件数 | 用途 |
    |:---|:---|:---|:---|
    | 鉄鋼 降伏強度 | Citrine Informatics / Matbench | 312 | 回帰（構造材料） |
    | 超伝導体 臨界温度 | NIMS SuperCon / UCI | 500 | 回帰（機能材料） |
    | HEA 相分類 | Zenodo ACHIEF | 1,103 | 分類（Hume-Rothery則） |
    """)

    st.markdown("---")
    st.header("📋 レポート課題")
    st.error("以下の課題から **2つ** を選択し、レポートとして提出してください。")

    st.markdown(r"""
    ---

    ### 課題 1: 回帰分析による物性予測（基礎）

    **テーマ**: 鉄鋼または超伝導体データを使い、物性値を予測するモデルを構築せよ。

    **要件**:
    1. データの探索的分析（要約統計量、相関行列、PCA biplot）を行い、データの特徴を説明せよ
    2. 外れ値検出を行い、結果を考察せよ
    3. 線形回帰と Ridge 回帰を適用し、R²・RMSE を比較せよ
    4. 5-fold CV の結果を報告し、モデルの安定性を議論せよ
    5. 正則化パラメータ $\alpha$ を変化させた検証曲線を作成し、最適値を決定せよ

    ---

    ### 課題 2: 過学習の理解と可視化（基礎〜応用）

    **テーマ**: 多項式回帰を用いて過学習現象を可視化・分析せよ。

    **要件**:
    1. 1変数の特徴量を選び、多項式の次数 $d = 1, 3, 5, 10, 15$ でフィッティングを行え
    2. 訓練誤差とテスト誤差をグラフ化し、過学習が起きる次数を特定せよ
    3. バイアス-バリアンス分解の数式を用いて結果を説明せよ
    4. データ増強（ノイズ付加 3倍）を適用し、過学習が軽減されることを示せ
    5. 学習曲線を作成し、データ数が増えた場合の効果を議論せよ

    ---

    ### 課題 3: Hume-Rothery則の機械学習による再現（応用）

    **テーマ**: HEAデータからHume-Rothery則を機械学習で再現し、材料設計指針を議論せよ。

    **要件**:
    1. HEAデータの探索的分析を行え（VEC, δ, ΔH_mix の分布と相との関係）
    2. SVM（カーネル: linear, rbf, poly）で SS/IM/AM を分類し正解率を比較せよ
    3. 混同行列を作成し、どの相の分類が難しいか考察せよ
    4. k-means クラスタリングで教師なし分離が可能か議論せよ
    5. VEC と δ の閾値を特定し、**Hume-Rothery則との整合性**を議論せよ
    6. 新合金設計の指針を提案せよ

    ---

    ### 課題 4: 特徴量重要度とモデル選択（応用）

    **テーマ**: Random Forest と Lasso を用いた特徴量重要度評価とモデル選択を行え。

    **要件**:
    1. 鉄鋼データを用い、RF の MDI で特徴量重要度を算出せよ
    2. Lasso の正則化パスにおいて、係数が0になる順序を報告せよ
    3. RF の重要度上位3特徴量と Lasso で残る特徴量を比較せよ
    4. 重要特徴量のみを使ったモデルと全特徴量モデルの性能を CV で比較せよ
    5. **材料科学的観点**から重要特徴量が物理的に妥当か考察せよ

    ---

    ### 課題 5: MI ワークフローの総合実践（発展）

    **テーマ**: 以下のデータソースからデータを取得し、MIの一連のワークフローを実践せよ。

    **データソース（いずれか）**:
    - [Materials Project](https://materialsproject.org/)
    - [OQMD](https://oqmd.org/)
    - [Matminer](https://hackingmaterials.lbl.gov/matminer/) — `load_dataset()`
    - [Kaggle Superconductor](https://www.kaggle.com/datasets/munumbutt/superconductor-dataset)

    **要件**:
    1. データ取得・前処理
    2. PCA biplot + 外れ値検出
    3. 最低3種のモデル比較
    4. 交差検証による汎化性能評価
    5. データ増強の効果検証
    6. 材料設計指針の提案

    ---
    """)

    st.header("9.3 レポート作成のガイドライン")
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
