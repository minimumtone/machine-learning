"""
マテリアルズ・インフォマティクス (MI) 講義用アプリケーション
============================================================
対象: 大学3回生 材料工学専攻 初心者向け
想定: 講義1コマで一連のMIワークフローを体験

セクション構成:
  1. MIとは
  2. データ探索
  3. 次元削減 PCA
  4. 回帰問題
  5. 正則化・モデル選択
  6. 分類問題
  7. 交差検証・汎化性能評価
  8. まとめ + レポート課題
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

# Matplotlib 日本語フォント設定（IPAGothic が無い環境でも動作）
try:
    matplotlib.font_manager.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
    plt.rcParams["font.family"] = "IPAGothic"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

# Plotly 日本語フォント設定（中国語フォントへのフォールバックを防止）
_JP_FONT = "Yu Gothic, YuGothic, Meiryo, Hiragino Sans, Hiragino Kaku Gothic ProN, Noto Sans JP, sans-serif"
_plotly_template = pio.templates["plotly"]
_plotly_template.layout.font = dict(family=_JP_FONT)
pio.templates.default = "plotly"

# Streamlit の Plotly テーマが上書きするため CSS でも強制指定
st.markdown(f"""<style>
.js-plotly-plot text, .js-plotly-plot .gtitle, .js-plotly-plot .xtitle,
.js-plotly-plot .ytitle, .js-plotly-plot .legendtext {{
    font-family: {_JP_FONT} !important;
}}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 材料データセット生成
# ---------------------------------------------------------------------------

@st.cache_data
def generate_steel_data(n=200, seed=42):
    """構造材料: 鉄鋼の機械的特性データ（組成・プロセス → 降伏強度）
    参照: OQMD / Matbench steel データをモデルに合成"""
    rng = np.random.default_rng(seed)
    C = rng.uniform(0.05, 0.8, n)       # 炭素 (wt%)
    Mn = rng.uniform(0.3, 2.0, n)       # マンガン (wt%)
    Si = rng.uniform(0.1, 1.0, n)       # ケイ素 (wt%)
    Cr = rng.uniform(0.0, 18.0, n)      # クロム (wt%)
    Ni = rng.uniform(0.0, 10.0, n)      # ニッケル (wt%)
    temp = rng.uniform(800, 1200, n)     # 焼入れ温度 (°C)
    # 降伏強度 (MPa) — 経験式ベース
    YS = (300 + 800*C + 50*Mn + 30*Si + 15*Cr + 10*Ni
           - 0.2*temp + 5*C*Mn*100
           + rng.normal(0, 30, n))
    YS = np.clip(YS, 150, 2000)
    df = pd.DataFrame({
        "C (wt%)": np.round(C, 3),
        "Mn (wt%)": np.round(Mn, 3),
        "Si (wt%)": np.round(Si, 3),
        "Cr (wt%)": np.round(Cr, 3),
        "Ni (wt%)": np.round(Ni, 3),
        "焼入温度 (°C)": np.round(temp, 1),
        "降伏強度 (MPa)": np.round(YS, 1),
    })
    return df


@st.cache_data
def generate_thermoelectric_data(n=180, seed=123):
    """機能材料: 熱電材料の性能指数 ZT
    参照: Matbench / Materials Project の熱電特性"""
    rng = np.random.default_rng(seed)
    carrier = rng.uniform(1e18, 1e21, n)           # キャリア濃度 (cm^-3)
    bandgap = rng.uniform(0.05, 2.0, n)            # バンドギャップ (eV)
    eff_mass = rng.uniform(0.5, 5.0, n)            # 有効質量 (m_e)
    lattice_thermal = rng.uniform(0.5, 10.0, n)    # 格子熱伝導率 (W/mK)
    temp_K = rng.uniform(300, 900, n)              # 測定温度 (K)
    log_carrier = np.log10(carrier)
    # ZT のモデル
    S = (200 - 30*(log_carrier - 19)**2 + 20*bandgap)  # Seebeck 係数的
    sigma = 50*log_carrier - 800 + 10*eff_mass          # 電気伝導率的
    ZT = (np.abs(S) * sigma / (lattice_thermal * 1e4) * (temp_K / 600)
          + rng.normal(0, 0.05, n))
    ZT = np.clip(ZT, 0.01, 3.0)
    df = pd.DataFrame({
        "キャリア濃度 log(cm⁻³)": np.round(log_carrier, 2),
        "バンドギャップ (eV)": np.round(bandgap, 3),
        "有効質量 (m_e)": np.round(eff_mass, 2),
        "格子熱伝導率 (W/mK)": np.round(lattice_thermal, 2),
        "測定温度 (K)": np.round(temp_K, 1),
        "性能指数 ZT": np.round(ZT, 3),
    })
    return df


@st.cache_data
def generate_polymer_data(n=160, seed=456):
    """高分子材料: ガラス転移温度 Tg の予測
    参照: PolyInfo / Matminer polymer データ"""
    rng = np.random.default_rng(seed)
    MW = rng.uniform(1e3, 1e6, n)                  # 分子量
    logMW = np.log10(MW)
    flexibility = rng.uniform(0, 10, n)             # 主鎖柔軟性指標
    polarity = rng.uniform(0, 5, n)                 # 極性指標
    crosslink = rng.uniform(0, 3, n)                # 架橋密度
    crystallinity = rng.uniform(0, 80, n)           # 結晶化度 (%)
    Tg = (80 + 20*polarity + 15*crosslink - 5*flexibility
          + 10*logMW + 0.3*crystallinity
          + rng.normal(0, 8, n))
    Tg = np.clip(Tg, -120, 400)
    df = pd.DataFrame({
        "分子量 log(g/mol)": np.round(logMW, 2),
        "主鎖柔軟性": np.round(flexibility, 2),
        "極性指標": np.round(polarity, 2),
        "架橋密度": np.round(crosslink, 2),
        "結晶化度 (%)": np.round(crystallinity, 1),
        "ガラス転移温度 Tg (°C)": np.round(Tg, 1),
    })
    return df


@st.cache_data
def generate_classification_data(n=250, seed=789):
    """分類用: 合金の結晶構造分類 (BCC/FCC/HCP)
    参照: OQMD / Matminer の結晶構造データ"""
    rng = np.random.default_rng(seed)
    VEC = rng.uniform(3.0, 12.0, n)                 # 価電子濃度
    delta_r = rng.uniform(0, 15, n)                  # 原子半径差 (%)
    delta_Hmix = rng.uniform(-30, 10, n)             # 混合エンタルピー (kJ/mol)
    electronegativity = rng.uniform(1.0, 2.5, n)     # 電気陰性度差
    # ルールベースの結晶構造分類
    structure = []
    for i in range(n):
        if VEC[i] < 6.87 and delta_r[i] < 6.6:
            structure.append("BCC")
        elif VEC[i] >= 8.0 and delta_r[i] < 6.6:
            structure.append("FCC")
        else:
            structure.append("HCP")
    # ノイズ付与（10%をランダムに変更）
    noise_idx = rng.choice(n, size=n // 10, replace=False)
    labels = ["BCC", "FCC", "HCP"]
    for idx in noise_idx:
        structure[idx] = rng.choice(labels)
    df = pd.DataFrame({
        "VEC": np.round(VEC, 2),
        "原子半径差 δ (%)": np.round(delta_r, 2),
        "混合エンタルピー ΔH_mix (kJ/mol)": np.round(delta_Hmix, 2),
        "電気陰性度差": np.round(electronegativity, 3),
        "結晶構造": structure,
    })
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
    "5. 正則化・モデル選択": "regularization",
    "6. 分類問題": "classification",
    "7. 交差検証・汎化性能": "cv_generalization",
    "8. まとめ＋レポート課題": "summary_assignments",
}

selected = st.sidebar.radio("セクションを選択", list(SECTIONS.keys()))
section_key = SECTIONS[selected]

st.sidebar.markdown("---")
st.sidebar.markdown("**使用データセット**")
dataset_choice = st.sidebar.selectbox(
    "材料データ",
    ["鉄鋼（構造材料）", "熱電材料（機能材料）", "高分子材料"],
)

# データ読み込み
if dataset_choice == "鉄鋼（構造材料）":
    df_main = generate_steel_data()
    target_col = "降伏強度 (MPa)"
    dataset_desc = "鉄鋼合金の組成・プロセス条件から降伏強度を予測（OQMD参照）"
    dataset_detail = """
| 特徴量 | 説明 | 単位 | 範囲 |
|:---|:---|:---|:---|
| C (wt%) | 炭素含有量 — 強度に最も影響する元素 | wt% | 0.05–0.8 |
| Mn (wt%) | マンガン — 固溶強化・靭性向上 | wt% | 0.3–2.0 |
| Si (wt%) | ケイ素 — 脱酸・固溶強化 | wt% | 0.1–1.0 |
| Cr (wt%) | クロム — 耐食性・焼入性向上 | wt% | 0–18 |
| Ni (wt%) | ニッケル — 靭性・耐食性向上 | wt% | 0–10 |
| 焼入温度 (°C) | オーステナイト化温度 | °C | 800–1200 |

**目的変数**: 降伏強度 (MPa) — 材料が塑性変形を開始する応力値
"""
elif dataset_choice == "熱電材料（機能材料）":
    df_main = generate_thermoelectric_data()
    target_col = "性能指数 ZT"
    dataset_desc = "熱電材料の電子構造特徴から性能指数ZTを予測（Materials Project参照）"
    dataset_detail = """
| 特徴量 | 説明 | 単位 | 範囲 |
|:---|:---|:---|:---|
| キャリア濃度 log(cm⁻³) | 電荷キャリア密度の対数値 | log(cm⁻³) | 18–21 |
| バンドギャップ (eV) | 電子の禁制帯幅 — 熱電性能の最適値が存在 | eV | 0.05–2.0 |
| 有効質量 (m_e) | キャリアの有効質量 — ゼーベック係数に寄与 | m_e | 0.5–5.0 |
| 格子熱伝導率 (W/mK) | フォノン輸送による熱伝導 — 低いほど高性能 | W/mK | 0.5–10 |
| 測定温度 (K) | 熱電性能評価温度 | K | 300–900 |

**目的変数**: 性能指数 ZT — 熱電変換効率を表す無次元数（ZT = S²σT/κ）
"""
else:
    df_main = generate_polymer_data()
    target_col = "ガラス転移温度 Tg (°C)"
    dataset_desc = "高分子の分子特徴からガラス転移温度を予測（PolyInfo参照）"
    dataset_detail = """
| 特徴量 | 説明 | 単位 | 範囲 |
|:---|:---|:---|:---|
| 分子量 log(g/mol) | 重量平均分子量の対数値 | log(g/mol) | 3–6 |
| 主鎖柔軟性 | 高分子主鎖の回転しやすさ指標 | — | 0–10 |
| 極性指標 | 側鎖の極性の強さ — 分子間力に影響 | — | 0–5 |
| 架橋密度 | 架橋点の密度 — Tg上昇要因 | — | 0–3 |
| 結晶化度 (%) | 結晶領域の割合 | % | 0–80 |

**目的変数**: ガラス転移温度 Tg (°C) — 非晶質領域がガラス状態からゴム状態に転移する温度
"""

feature_cols = [c for c in df_main.columns if c != target_col]
df_cls = generate_classification_data()

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
    | **OQMD** | ~100万件の DFT 計算結果（形成エネルギー等） | oqmd.org |
    | **Materials Project** | 結晶構造・電子状態・弾性定数 | materialsproject.org |
    | **MatBench** | 標準化された ML ベンチマーク | matbench.materialsproject.org |
    | **AFLOW** | 結晶構造・熱力学特性 | aflow.org |
    | **Open Catalyst** | 触媒反応の DFT データ | opencatalystproject.org |
    | **PolyInfo** | 高分子物性データ | polymer.nims.go.jp |
    | **Matminer** | 材料特徴量の自動生成ライブラリ | hackingmaterials.lbl.gov/matminer |
    """)

    st.header("1.4 本講義で学ぶこと")
    st.info(r"""
    **本日の講義で以下を体験します：**

    1. **データ探索** — 要約統計量・ペアプロット・外れ値検出
    2. **次元削減** — 主成分分析 (PCA)
    3. **回帰問題** — 線形回帰 → 多項式近似 → 過学習 → SVR → Random Forest
    4. **正則化** — Lasso / Ridge による特徴量選択とモデル選択
    5. **分類問題** — SVM / k-means クラスタリング
    6. **交差検証** — k-fold CV / LOOCV / 学習曲線
    7. **汎化性能評価** — バイアス-バリアンス分解
    """)

    st.header("1.5 機械学習の基本的な枠組み")
    st.markdown(r"""
    機械学習の目標は、入力 $\mathbf{x}$ から出力 $y$ への写像 $f$ を学習することです。

    $$
    y = f(\mathbf{x}) + \varepsilon
    $$

    - $\mathbf{x} = (x_1, x_2, \dots, x_p)^T$：**特徴量**（説明変数）— 組成・プロセス条件など
    - $y$：**目的変数**（応答変数）— 物性値（強度、ZT、Tg など）
    - $\varepsilon$：ノイズ（測定誤差など）、$E[\varepsilon] = 0$

    **回帰問題**: $y$ が連続値（例：降伏強度 300 MPa）

    **分類問題**: $y$ がカテゴリ（例：結晶構造 = BCC / FCC / HCP）
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

    # 2.1 データの確認
    st.header("2.1 データの概要")
    st.markdown("#### データセットの説明")
    st.markdown(dataset_detail)
    st.dataframe(df_main.head(20), use_container_width=True)
    st.markdown(f"データ数: **{len(df_main)}** 件、特徴量数: **{len(feature_cols)}** 個、目的変数: **{target_col}**")

    # 2.2 要約統計量
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

    # 2.3 ヒストグラム
    st.header("2.3 分布の可視化（ヒストグラム）")
    hist_col = st.selectbox("表示する変数", df_main.columns.tolist())
    fig_hist = px.histogram(df_main, x=hist_col, nbins=30, marginal="box",
                            title=f"{hist_col} の分布")
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

    # 2.4 相関行列
    st.header("2.4 相関行列")
    st.markdown(r"""
    **ピアソン相関係数** は2変数間の線形関係の強さを表します：

    $$
    r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
    {\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2 \sum_{i=1}^{n}(y_i-\bar{y})^2}}
    $$

    - $r = 1$：完全な正の相関
    - $r = -1$：完全な負の相関
    - $r = 0$：線形相関なし
    """)
    corr = df_main.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="相関行列ヒートマップ")
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    # 2.5 ペアプロット
    st.header("2.5 ペアプロット（散布図行列）")
    st.markdown("特徴量間の関係を一覧で確認します。特徴量を選んでください。")
    pair_cols = st.multiselect("表示する変数（2〜4個推奨）",
                               df_main.columns.tolist(),
                               default=df_main.columns.tolist()[:3] + [target_col])
    if len(pair_cols) >= 2:
        fig_pair = px.scatter_matrix(df_main[pair_cols], dimensions=pair_cols,
                                     height=600, title="ペアプロット")
        fig_pair.update_traces(diagonal_visible=True, marker=dict(size=3))
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        st.warning("2つ以上の変数を選択してください。")

    # 2.6 異常データ（外れ値）の検出
    st.header("2.6 異常データの検出（外れ値検出）")
    st.markdown(r"""
    材料データには測定ミスや特殊条件のデータが含まれることがあります。
    外れ値を検出し、適切に対処することがモデル構築の前処理として重要です。

    ### IQR 法（四分位範囲法）
    $$
    \text{外れ値条件}: \quad x < Q_1 - 1.5 \times \text{IQR} \quad \text{or} \quad x > Q_3 + 1.5 \times \text{IQR}
    $$
    ここで $\text{IQR} = Q_3 - Q_1$（第3四分位 − 第1四分位）

    ### Isolation Forest
    ランダムな分割で**孤立しやすいデータ**を異常とみなす。
    正常データは多くの分割が必要だが、異常データは少ない分割で孤立する。

    ### LOF (Local Outlier Factor)
    局所的なデータ密度を比較し、**周囲より密度が低い**点を異常とみなす。

    $$
    \text{LOF}(x) = \frac{\text{近傍の平均密度}}{\text{点 } x \text{ の密度}}
    $$

    LOF > 1 なら周囲より密度が低い（異常の可能性あり）。
    """)

    outlier_method = st.selectbox("外れ値検出手法",
                                  ["IQR法", "Isolation Forest", "LOF"])
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
        X_num = df_main.select_dtypes(include=[np.number])
        preds = iso.fit_predict(X_num)
        df_outlier["外れ値"] = preds == -1
    else:  # LOF
        n_neighbors = st.slider("近傍数", 5, 50, 20)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        X_num = df_main.select_dtypes(include=[np.number])
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

    st.info("""
    **材料データにおける外れ値の扱い方**
    - 測定ミス → 除外が妥当
    - 特殊条件のデータ → 分析目的に応じて判断
    - 新規材料の発見 → 外れ値こそ重要な場合もある（例: 超高 ZT 材料）
    """)


# =====================================================================
# セクション 3: 次元削減 PCA
# =====================================================================
elif section_key == "pca":
    st.title("🔄 主成分分析 (PCA)")
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
       - $\lambda_k$: 第 $k$ 主成分の固有値（= 分散の大きさ）
       - $\mathbf{v}_k$: 第 $k$ 主成分の固有ベクトル（= 主成分の方向）
    4. 固有値の大きい順に主成分を選択

    ### 寄与率と累積寄与率
    第 $k$ 主成分の**寄与率**（説明した分散の割合）:

    $$
    \text{寄与率}_k = \frac{\lambda_k}{\sum_{j=1}^{p} \lambda_j}
    $$

    **累積寄与率** が 80〜90% 以上になる主成分数を採用するのが一般的です。

    ### 主成分スコア
    データ $\mathbf{x}_i$ の第 $k$ 主成分スコア:

    $$
    z_{ik} = \mathbf{v}_k^T \mathbf{x}_i = \sum_{j=1}^{p} v_{kj} x_{ij}
    $$
    """)

    st.header("3.2 PCA の実行")
    X = df_main[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(len(feature_cols), X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # 寄与率
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
        # 主成分負荷量
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

    # 2D 散布図
    st.header("3.3 主成分空間でのデータ分布")
    if n_components >= 2:
        df_pca_plot = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            target_col: df_main[target_col].values,
        })
        fig_pca = px.scatter(df_pca_plot, x="PC1", y="PC2", color=target_col,
                             color_continuous_scale="Viridis",
                             title="PC1 vs PC2（色: 目的変数）",
                             labels={"PC1": f"PC1 ({explained[0]*100:.1f}%)",
                                     "PC2": f"PC2 ({explained[1]*100:.1f}%)"})
        fig_pca.update_layout(height=500)
        st.plotly_chart(fig_pca, use_container_width=True)

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

    パラメータ $\boldsymbol{\beta}$ は**残差平方和 (RSS)** を最小化して求めます：

    $$
    \boldsymbol{\hat{\beta}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2
    $$

    行列表記では、**正規方程式**：

    $$
    \boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
    $$

    ### 幾何学的解釈
    OLS は $\mathbf{y}$ を列空間 $\text{Col}(\mathbf{X})$ に直交射影しています。
    残差ベクトル $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ は列空間と直交します。
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

    # 予測 vs 実測
    fig_lr = px.scatter(x=y_test, y=y_pred_lr,
                        labels={"x": f"実測値 ({target_col})", "y": "予測値"},
                        title=f"線形回帰: 予測 vs 実測 (R² = {r2_lr:.4f})")
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    fig_lr.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                     line=dict(dash="dash", color="red"))
    fig_lr.update_layout(height=450)
    st.plotly_chart(fig_lr, use_container_width=True)

    # 係数
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

    次数 $d$ を上げるとモデルの**表現力**（複雑さ）が増しますが、
    高すぎると**過学習 (overfitting)** を起こします。

    ### 過学習 (Overfitting) とは
    - 訓練データに**過度に適合**し、未知データに対する予測性能が低下すること
    - 訓練誤差は低いが、テスト誤差が高い状態
    - モデルがデータのノイズまで学習してしまう

    ### バイアス-バリアンス分解
    テスト誤差は以下の3つの成分に分解できます：

    $$
    E\left[(y - \hat{f}(\mathbf{x}))^2\right] = \underbrace{\text{Bias}^2[\hat{f}]}_{\text{未学習}} + \underbrace{\text{Var}[\hat{f}]}_{\text{過学習}} + \underbrace{\sigma^2}_{\text{ノイズ}}
    $$

    - **バイアス (Bias)**: モデルが単純すぎることによる誤差（未学習）
    - **バリアンス (Variance)**: 訓練データの違いによる予測のばらつき（過学習）
    - **ノイズ** $\sigma^2$: 削減不可能な誤差

    **最適なモデル複雑度** = バイアスとバリアンスのトレードオフ点
    """)

    # 1変数での多項式フィッティングデモ
    st.subheader("多項式近似のデモ（1変数）")
    st.markdown("特徴量1つを選び、多項式の次数を変えて当てはまりの変化を観察します。")
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
        # フィッティング曲線
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
                             xaxis_title=demo_feature, yaxis_title=target_col,
                             height=450)
        st.plotly_chart(fig_fit, use_container_width=True)

    with col2:
        # 学習曲線（過学習の可視化）
        fig_overfit = go.Figure()
        fig_overfit.add_scatter(x=degrees, y=train_errors, mode="lines+markers",
                               name="訓練誤差 (MSE)")
        fig_overfit.add_scatter(x=degrees, y=test_errors, mode="lines+markers",
                               name="テスト誤差 (MSE)")
        fig_overfit.update_layout(title="過学習の可視化: 次数 vs 誤差",
                                 xaxis_title="多項式の次数",
                                 yaxis_title="MSE",
                                 height=450)
        st.plotly_chart(fig_overfit, use_container_width=True)

    best_deg = degrees[np.argmin(test_errors)]
    st.success(f"テスト誤差が最小の次数: **{best_deg}** (MSE = {min(test_errors):.2f})")

    st.warning(r"""
    **過学習の見分け方**
    - 訓練誤差は低いのにテスト誤差が高い → 過学習
    - 次数が上がるほど訓練誤差は下がるが、テスト誤差はある点から上昇する
    - これが**バイアス-バリアンストレードオフ**の具体例
    """)

    # 4.3 SVR
    st.header("4.3 サポートベクター回帰 (SVR)")
    st.markdown(r"""
    **SVR (Support Vector Regression)** は、SVM の回帰版です。
    予測値と実測値の差が $\varepsilon$ 以内なら損失を0とする**ε-不感損失関数**を使います：

    $$
    L_\varepsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \varepsilon)
    $$

    最適化問題：
    $$
    \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} L_\varepsilon(y_i, \hat{y}_i)
    $$

    - $C$: 正則化パラメータ（大きいほどデータに忠実）
    - $\varepsilon$: 不感帯の幅
    - **カーネルトリック**: 非線形変換を暗黙的に適用（RBF, polynomial など）

    RBF カーネル:
    $$
    K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)
    $$
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
    **ランダムフォレスト (Random Forest)** はバギング + 決定木のアンサンブル学習です。

    ### アルゴリズム
    1. データの**ブートストラップサンプル**を $B$ 個生成
    2. 各サンプルで決定木を学習（各分岐で $m = \sqrt{p}$ 個の特徴量をランダムに選択）
    3. $B$ 個の木の予測を**平均**（回帰）または**多数決**（分類）

    $$
    \hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})
    $$

    ### 特徴量重要度
    **MDI (Mean Decrease Impurity)**: 各特徴量で分岐した時の不純度の減少量の合計

    $$
    \text{Importance}(x_j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b} \Delta I(t, x_j)
    $$

    - $\Delta I(t, x_j)$: ノード $t$ での特徴量 $x_j$ による不純度の減少
    - 回帰: MSE の減少、分類: ジニ不純度 or エントロピーの減少

    **Permutation Importance**: 特徴量の値をシャッフルして精度低下を測定
    — MDI よりもバイアスが少ないとされる
    """)

    col1, col2 = st.columns(2)
    with col1:
        rf_n_estimators = st.slider("木の本数 (n_estimators)", 10, 500, 100, step=10)
    with col2:
        rf_max_depth = st.slider("最大深さ (max_depth)", 2, 30, 10)

    rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    col1, col2 = st.columns(2)
    col1.metric("RF R²", f"{r2_rf:.4f}")
    col2.metric("RF RMSE", f"{rmse_rf:.2f}")

    # 特徴量重要度
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
# セクション 5: 正則化・モデル選択
# =====================================================================
elif section_key == "regularization":
    st.title("🎯 正則化とモデル選択")
    st.markdown(f"**使用データ**: {dataset_desc}")
    st.markdown("---")

    X = df_main[feature_cols].values
    y = df_main[target_col].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    st.header("5.1 正則化の理論")
    st.markdown(r"""
    **正則化 (Regularization)** は、モデルの複雑さにペナルティを課して
    過学習を防ぐ手法です。損失関数にペナルティ項を追加します。

    ### Ridge 回帰（L2 正則化）

    $$
    \hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}}
    \left[ \sum_{i=1}^{n}(y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2
    + \alpha \sum_{j=1}^{p} \beta_j^2 \right]
    $$

    - ペナルティ: $\alpha \|\boldsymbol{\beta}\|_2^2 = \alpha \sum_j \beta_j^2$
    - 係数を **0に近づける**（縮小するが0にはしない）
    - 閉形式の解: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X} + \alpha \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
    - 多重共線性（特徴量間の高い相関）に強い

    ### Lasso 回帰（L1 正則化）

    $$
    \hat{\boldsymbol{\beta}}_{\text{Lasso}} = \arg\min_{\boldsymbol{\beta}}
    \left[ \sum_{i=1}^{n}(y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2
    + \alpha \sum_{j=1}^{p} |\beta_j| \right]
    $$

    - ペナルティ: $\alpha \|\boldsymbol{\beta}\|_1 = \alpha \sum_j |\beta_j|$
    - 係数を **完全に0にする**（スパース解 → 特徴量選択）
    - 閉形式の解はない（座標降下法などで求解）

    ### 正則化パラメータ α の意味
    - $\alpha = 0$: 正則化なし = 通常の最小二乗法
    - $\alpha \to \infty$: すべての係数が0（定数モデル）
    - **最適な $\alpha$** は交差検証で決定する
    """)

    st.header("5.2 Ridge vs Lasso の比較")

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

    # α パス
    st.header("5.3 正則化パス（α vs 係数）")
    st.markdown("α を変化させた時の係数の推移を観察します。")

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
            fig_rp.add_scatter(x=np.log10(alphas), y=ridge_coefs[:, j],
                              mode="lines", name=name)
        fig_rp.update_layout(title="Ridge 正則化パス",
                            xaxis_title="log₁₀(α)", yaxis_title="係数",
                            height=400)
        st.plotly_chart(fig_rp, use_container_width=True)

    with col2:
        fig_lp = go.Figure()
        for j, name in enumerate(feature_cols):
            fig_lp.add_scatter(x=np.log10(alphas), y=lasso_coefs[:, j],
                              mode="lines", name=name)
        fig_lp.update_layout(title="Lasso 正則化パス",
                            xaxis_title="log₁₀(α)", yaxis_title="係数",
                            height=400)
        st.plotly_chart(fig_lp, use_container_width=True)

    st.info(r"""
    **正則化パスの読み方**
    - **Ridge**: α が大きくなると全係数が徐々に 0 に近づく（縮小）
    - **Lasso**: α が大きくなると係数が順次 **完全に 0** になる（特徴量選択）
    - Lasso で最後まで残る特徴量が最も重要な特徴量
    """)


# =====================================================================
# セクション 6: 分類問題
# =====================================================================
elif section_key == "classification":
    st.title("🏷️ 分類問題")
    st.markdown("**使用データ**: 合金の結晶構造分類 (BCC / FCC / HCP)")
    st.markdown("---")

    st.header("6.1 分類問題とは")
    st.markdown(r"""
    分類問題は、入力 $\mathbf{x}$ を離散的なカテゴリ $y \in \{C_1, C_2, \dots, C_K\}$ に
    分類する問題です。

    材料科学の例：
    - 結晶構造の予測（BCC / FCC / HCP）
    - アモルファス形成能の判定（形成する / しない）
    - 超伝導体の分類
    """)

    _cls_features = ["VEC", "原子半径差 δ (%)", "混合エンタルピー ΔH_mix (kJ/mol)", "電気陰性度差"]
    X_cls = df_cls[_cls_features].values
    y_cls = np.array(df_cls["結晶構造"].tolist())
    X_tr_c_raw, X_te_c_raw, y_tr_c, y_te_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    scaler_cls = StandardScaler()
    X_tr_c = scaler_cls.fit_transform(X_tr_c_raw)
    X_te_c = scaler_cls.transform(X_te_c_raw)

    st.dataframe(df_cls.head(10), use_container_width=True)

    # クラス分布
    cls_counts = df_cls["結晶構造"].value_counts()
    fig_cls_dist = px.bar(x=cls_counts.index, y=cls_counts.values,
                          title="結晶構造のクラス分布",
                          labels={"x": "結晶構造", "y": "データ数"})
    fig_cls_dist.update_layout(height=300)
    st.plotly_chart(fig_cls_dist, use_container_width=True)

    # 6.2 SVM
    st.header("6.2 サポートベクターマシン (SVM)")
    st.markdown(r"""
    **SVM (Support Vector Machine)** は、クラス間の**マージン（余白）を最大化**する
    超平面を求める分類手法です。

    ### 線形 SVM

    2クラス分類の場合、決定境界は超平面 $\mathbf{w}^T \mathbf{x} + b = 0$ で表されます。

    最適化問題（ハードマージン）：
    $$
    \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad
    \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i
    $$

    **マージン**: $\frac{2}{\|\mathbf{w}\|}$（大きいほど汎化性能が高い）

    ソフトマージン（ノイズを許容）：
    $$
    \min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i
    \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
    $$

    - $C$: 正則化パラメータ（大 → ハードマージンに近い、小 → 誤分類を許容）
    - $\xi_i$: スラック変数（マージン違反の程度）

    ### カーネル SVM
    非線形分離にはカーネルトリックを使用します：

    $$
    K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')
    $$

    - RBF カーネル: $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$
    """)

    col1, col2 = st.columns(2)
    with col1:
        svm_kernel = st.selectbox("SVM カーネル", ["rbf", "linear", "poly"])
        svm_C = st.slider("C", 0.1, 100.0, 1.0, key="svm_c")

    svc = SVC(kernel=svm_kernel, C=svm_C, random_state=42)
    svc.fit(X_tr_c, y_tr_c)
    y_pred_svc = svc.predict(X_te_c)
    acc_svc = accuracy_score(y_te_c, y_pred_svc)

    st.metric("SVM 正解率 (Accuracy)", f"{acc_svc:.4f}")

    # 混同行列
    cm = confusion_matrix(y_te_c, y_pred_svc, labels=["BCC", "FCC", "HCP"])
    fig_cm = px.imshow(cm, x=["BCC", "FCC", "HCP"], y=["BCC", "FCC", "HCP"],
                       text_auto=True, color_continuous_scale="Blues",
                       labels={"x": "予測", "y": "実際"},
                       title="SVM 混同行列")
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("**分類レポート:**")
    report = classification_report(y_te_c, y_pred_svc, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    # 決定境界の可視化（2D PCA空間）
    st.subheader("決定境界の可視化（PCA 2D 射影）")
    pca_cls = PCA(n_components=2)
    X_cls_scaled_all = scaler_cls.transform(X_cls)
    X_cls_2d = pca_cls.fit_transform(X_cls_scaled_all)
    X_tr_2d, X_te_2d, y_tr_2d, y_te_2d = train_test_split(
        X_cls_2d, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    svc_2d = SVC(kernel=svm_kernel, C=svm_C, random_state=42)
    svc_2d.fit(X_tr_2d, y_tr_2d)

    h = 0.2
    x_min, x_max = X_cls_2d[:, 0].min() - 1, X_cls_2d[:, 0].max() + 1
    y_min, y_max = X_cls_2d[:, 1].min() - 1, X_cls_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc_2d.predict(np.c_[xx.ravel(), yy.ravel()])

    label_map = {"BCC": 0, "FCC": 1, "HCP": 2}
    Z_num = np.array([label_map[z] for z in Z]).reshape(xx.shape)

    fig_db = go.Figure()
    fig_db.add_contour(x=np.arange(x_min, x_max, h), y=np.arange(y_min, y_max, h),
                       z=Z_num, showscale=False, opacity=0.3,
                       colorscale=["blue", "green", "red"],
                       contours=dict(coloring="heatmap"))

    for label, color in zip(["BCC", "FCC", "HCP"], ["blue", "green", "red"]):
        mask = y_cls == label
        fig_db.add_scatter(x=X_cls_2d[mask, 0], y=X_cls_2d[mask, 1],
                          mode="markers", name=label,
                          marker=dict(color=color, size=6))

    fig_db.update_layout(title="SVM 決定境界 (PCA 2D)",
                        xaxis_title="PC1", yaxis_title="PC2", height=500)
    st.plotly_chart(fig_db, use_container_width=True)

    # 6.3 k-means
    st.header("6.3 k-means クラスタリング")
    st.markdown(r"""
    **k-means** は教師なし学習の代表的なクラスタリング手法です。

    ### アルゴリズム
    1. $k$ 個のクラスタ中心をランダムに初期化
    2. 各データを最も近いクラスタ中心に割り当て
    3. クラスタ中心を再計算（所属データの平均）
    4. 収束するまで 2-3 を繰り返す

    ### 目的関数
    クラスタ内分散の最小化（Within-Cluster Sum of Squares; WCSS）:

    $$
    J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
    $$

    - $C_k$: 第 $k$ クラスタに属するデータ集合
    - $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$: クラスタ中心

    ### 最適な k の決定
    - **エルボー法**: WCSS の減少率が急激に鈍る点（肘）を選ぶ
    - **シルエットスコア**: クラスタの密集度と分離度のバランスを評価

    $$
    s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
    $$
    - $a(i)$: 同クラスタ内の平均距離
    - $b(i)$: 最近隣クラスタとの平均距離
    - $s(i) \in [-1, 1]$: 1に近いほど良い分離
    """)

    n_clusters = st.slider("クラスタ数 k", 2, 10, 3)

    X_cls_num_scaled = scaler_cls.transform(X_cls)
    pca_km = PCA(n_components=2)
    X_km_2d = pca_km.fit_transform(X_cls_num_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_cls_num_scaled)
    sil = silhouette_score(X_cls_num_scaled, km_labels)

    st.metric("シルエットスコア", f"{sil:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        df_km = pd.DataFrame({"PC1": X_km_2d[:, 0], "PC2": X_km_2d[:, 1],
                              "クラスタ": km_labels.astype(str),
                              "実際の構造": y_cls})
        fig_km = px.scatter(df_km, x="PC1", y="PC2", color="クラスタ",
                           symbol="実際の構造",
                           title=f"k-means (k={n_clusters}) — PCA 2D",
                           height=450)
        st.plotly_chart(fig_km, use_container_width=True)

    with col2:
        # エルボー法
        wcss = []
        sil_scores = []
        K_range = range(2, 11)
        for k in K_range:
            km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_temp.fit(X_cls_num_scaled)
            wcss.append(km_temp.inertia_)
            sil_scores.append(silhouette_score(X_cls_num_scaled, km_temp.labels_))

        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_scatter(x=list(K_range), y=wcss, mode="lines+markers",
                             name="WCSS", secondary_y=False)
        fig_elbow.add_scatter(x=list(K_range), y=sil_scores, mode="lines+markers",
                             name="シルエット", secondary_y=True)
        fig_elbow.update_layout(title="エルボー法 + シルエットスコア", height=450)
        fig_elbow.update_xaxes(title_text="クラスタ数 k")
        fig_elbow.update_yaxes(title_text="WCSS", secondary_y=False)
        fig_elbow.update_yaxes(title_text="シルエットスコア", secondary_y=True)
        st.plotly_chart(fig_elbow, use_container_width=True)


# =====================================================================
# セクション 7: 交差検証・汎化性能
# =====================================================================
elif section_key == "cv_generalization":
    st.title("🔄 交差検証と汎化性能評価")
    st.markdown(f"**使用データ**: {dataset_desc}")
    st.markdown("---")

    X = df_main[feature_cols].values
    y = df_main[target_col].values

    st.header("7.1 交差検証の理論")
    st.markdown(r"""
    **交差検証 (Cross-Validation; CV)** は、モデルの汎化性能を評価するための手法です。

    ### なぜ必要か？
    テストデータに対する性能は「たまたま」良い/悪い可能性があります。
    CVはデータの分割を複数回行い、性能の安定性を評価します。

    ### ホールドアウト法
    最も簡単な方法。データを訓練セットとテストセットに1回だけ分割。

    $$
    \text{Score} = \text{Metric}(y_{\text{test}}, \hat{y}_{\text{test}})
    $$

    **問題点**: 分割のランダム性に依存する（不安定）

    ---

    ### k-fold 交差検証

    データを $k$ 個のフォールド（部分集合）に分割し、$k$ 回学習・評価を繰り返す：

    1. データを $k$ 等分: $D_1, D_2, \dots, D_k$
    2. $i = 1, 2, \dots, k$ について:
       - $D_i$ をテスト、残り $D \setminus D_i$ を訓練に使用
       - 評価指標 $\text{Score}_i$ を計算
    3. 平均と標準偏差を算出:

    $$
    \overline{\text{Score}} = \frac{1}{k}\sum_{i=1}^{k}\text{Score}_i
    \quad \pm \quad s = \sqrt{\frac{1}{k-1}\sum_{i=1}^{k}(\text{Score}_i - \overline{\text{Score}})^2}
    $$

    **一般的な $k$ の値**: 5 または 10

    ---

    ### LOOCV (Leave-One-Out Cross-Validation)

    $k = n$（データ数）の極端なケース。1つのデータだけをテスト、残り全てで学習：

    $$
    \text{CV}_{\text{LOOCV}} = \frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{y}_{-i})
    $$

    - $\hat{y}_{-i}$: $i$ 番目のデータを除いて学習したモデルの予測
    - **利点**: バイアスが非常に小さい
    - **欠点**: 計算コストが大（$n$ 回学習が必要）、分散が大きい
    """)

    st.header("7.2 交差検証の実行")

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
        score_label = "R²"
    else:
        cv = LeaveOneOut()
        scoring = "neg_mean_squared_error"
        score_label = "負MSE"
        st.warning(f"LOOCV: データ数 {len(X)} 回の学習を行います。少し時間がかかります。")
        st.info("LOOCV では各フォールドが1サンプルのため R² は定義できません。代わりに MSE を使用し、全予測値から総合 R² を算出します。")

    with st.spinner("交差検証を実行中..."):
        scores = cross_val_score(model_cv, X, y, cv=cv, scoring=scoring)

    st.markdown(f"### 結果 ({cv_method})")
    if cv_method == "LOOCV":
        mse_scores = -scores  # neg_mean_squared_error → positive MSE
        y_pred_loocv = cross_val_predict(model_cv, X, y, cv=LeaveOneOut())
        overall_r2 = r2_score(y, y_pred_loocv)
        overall_rmse = np.sqrt(mse_scores.mean())
        col1, col2, col3 = st.columns(3)
        col1.metric("総合 R²（全予測から算出）", f"{overall_r2:.4f}")
        col2.metric("平均 RMSE", f"{overall_rmse:.4f}")
        col3.metric("フォールド数", f"{len(scores)}")
        fig_cv = px.histogram(x=mse_scores, nbins=30, title="LOOCV 各サンプルの二乗誤差分布",
                              labels={"x": "二乗誤差"})
        fig_cv.update_layout(height=400)
        st.plotly_chart(fig_cv, use_container_width=True)
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

    # 7.3 学習曲線
    st.header("7.3 学習曲線")
    st.markdown(r"""
    **学習曲線** はデータ数と性能の関係を可視化します。

    - **訓練スコア**: 常に高い（データが少ないほど完全にフィット）
    - **検証スコア**: データが増えると改善する
    - 2つの曲線の **ギャップ** = 過学習の度合い

    $$
    \text{ギャップが大きい} \Rightarrow \text{過学習（データ増加 or 正則化が有効）}
    $$
    $$
    \text{両方低い} \Rightarrow \text{未学習（モデルの複雑度を上げる）}
    $$
    """)

    with st.spinner("学習曲線を計算中..."):
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model_cv, X, y, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="r2", n_jobs=-1
        )

    fig_lc = go.Figure()
    fig_lc.add_scatter(x=train_sizes_abs, y=train_scores.mean(axis=1),
                       mode="lines+markers", name="訓練スコア",
                       error_y=dict(type="data", array=train_scores.std(axis=1)))
    fig_lc.add_scatter(x=train_sizes_abs, y=test_scores.mean(axis=1),
                       mode="lines+markers", name="検証スコア",
                       error_y=dict(type="data", array=test_scores.std(axis=1)))
    fig_lc.update_layout(title="学習曲線",
                        xaxis_title="訓練データ数",
                        yaxis_title="R² スコア", height=450)
    st.plotly_chart(fig_lc, use_container_width=True)

    # 7.4 検証曲線
    st.header("7.4 検証曲線（ハイパーパラメータ vs 性能）")
    st.markdown(r"""
    **検証曲線** は、あるハイパーパラメータを変化させた時の
    訓練スコアと検証スコアの推移を表示します。

    最適なハイパーパラメータ = 検証スコアが最大の点。
    """)

    val_model = st.selectbox("モデル（検証曲線）", ["Ridge", "Lasso", "Random Forest (max_depth)"])

    with st.spinner("検証曲線を計算中..."):
        if val_model == "Ridge":
            param_range = np.logspace(-3, 3, 20)
            train_s, test_s = validation_curve(
                Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
                X, y, param_name="model__alpha",
                param_range=param_range, cv=5, scoring="r2", n_jobs=-1
            )
            param_label = "α (log scale)"
            x_vals = np.log10(param_range)
        elif val_model == "Lasso":
            param_range = np.logspace(-4, 1, 20)
            train_s, test_s = validation_curve(
                Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10000))]),
                X, y, param_name="model__alpha",
                param_range=param_range, cv=5, scoring="r2", n_jobs=-1
            )
            param_label = "α (log scale)"
            x_vals = np.log10(param_range)
        else:
            param_range = np.arange(2, 25)
            train_s, test_s = validation_curve(
                Pipeline([("model", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))]),
                X, y, param_name="model__max_depth",
                param_range=param_range, cv=5, scoring="r2", n_jobs=-1
            )
            param_label = "max_depth"
            x_vals = param_range

    fig_vc = go.Figure()
    fig_vc.add_scatter(x=x_vals, y=train_s.mean(axis=1), mode="lines+markers",
                       name="訓練スコア",
                       error_y=dict(type="data", array=train_s.std(axis=1)))
    fig_vc.add_scatter(x=x_vals, y=test_s.mean(axis=1), mode="lines+markers",
                       name="検証スコア",
                       error_y=dict(type="data", array=test_s.std(axis=1)))
    fig_vc.update_layout(title=f"検証曲線: {val_model}",
                        xaxis_title=param_label, yaxis_title="R²", height=450)
    st.plotly_chart(fig_vc, use_container_width=True)

    best_idx = test_s.mean(axis=1).argmax()
    if val_model in ["Ridge", "Lasso"]:
        best_param = param_range[best_idx]
        st.success(f"最適な α ≈ {best_param:.4f}（検証 R² = {test_s.mean(axis=1)[best_idx]:.4f}）")
    else:
        st.success(f"最適な max_depth ≈ {param_range[best_idx]}（検証 R² = {test_s.mean(axis=1)[best_idx]:.4f}）")

    st.header("7.5 汎化性能のまとめ")
    st.markdown(r"""
    ### バイアス-バリアンストレードオフ（再掲）

    $$
    \text{テスト誤差} = \text{Bias}^2 + \text{Variance} + \text{Noise}
    $$

    | 状態 | 訓練誤差 | テスト誤差 | 学習曲線の特徴 | 対策 |
    |:---|:---|:---|:---|:---|
    | **未学習** | 高 | 高 | 両方低い、ギャップ小 | モデル複雑度↑、特徴量追加 |
    | **適切** | やや低 | やや低 | 適度なギャップ | — |
    | **過学習** | 非常に低 | 高 | ギャップ大 | 正則化、データ増加、特徴量削減 |
    """)


# =====================================================================
# セクション 8: まとめ + レポート課題
# =====================================================================
elif section_key == "summary_assignments":
    st.title("📝 まとめとレポート課題")
    st.markdown("---")

    st.header("8.1 本日のまとめ")
    st.markdown(r"""
    ### マテリアルズ・インフォマティクスの一連のワークフローを体験しました

    | ステップ | 手法 | キーポイント |
    |:---|:---|:---|
    | **データ探索** | 要約統計量、ペアプロット、外れ値検出 | データの質と特性を把握することが最重要 |
    | **次元削減** | PCA | 高次元データの可視化と冗長性の除去 |
    | **回帰** | 線形回帰、多項式回帰、SVR、Random Forest | 物性値の定量予測 |
    | **過学習** | バイアス-バリアンス分解 | モデルの複雑度と汎化性能のトレードオフ |
    | **正則化** | Lasso (L1) / Ridge (L2) | 過学習防止と特徴量選択 |
    | **分類** | SVM / k-means | 材料のカテゴリ分類とクラスタリング |
    | **モデル評価** | k-fold CV / LOOCV / 学習曲線 | 汎化性能の信頼性評価 |
    """)

    st.header("8.2 重要公式のまとめ")
    st.markdown(r"""
    #### 回帰
    - **線形回帰**: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
    - **Ridge**: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
    - **Lasso**: $\min \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1$

    #### 評価指標
    - **MSE**: $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$
    - **$R^2$**: $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

    #### PCA
    - 共分散行列の固有値問題: $\mathbf{C}\mathbf{v} = \lambda\mathbf{v}$
    - 寄与率: $\lambda_k / \sum \lambda_j$

    #### SVM
    - マージン最大化: $\min \frac{1}{2}\|\mathbf{w}\|^2 + C\sum\xi_i$

    #### k-means
    - WCSS: $J = \sum_k \sum_{\mathbf{x} \in C_k}\|\mathbf{x} - \boldsymbol{\mu}_k\|^2$

    #### 交差検証
    - k-fold: $\overline{\text{Score}} = \frac{1}{k}\sum_{i=1}^{k}\text{Score}_i$

    #### バイアス-バリアンス分解
    - $E[(y-\hat{f})^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$
    """)

    st.markdown("---")
    st.header("📋 レポート課題")
    st.error("以下の課題から **2つ** を選択し、レポートとして提出してください。")

    st.markdown(r"""
    ---

    ### 課題 1: 回帰分析による物性予測（基礎）

    **テーマ**: 本アプリの材料データセットの1つを使い、物性値を予測するモデルを構築せよ。

    **要件**:
    1. データの探索的分析（要約統計量、相関行列、ペアプロット）を行い、データの特徴を説明せよ
    2. 外れ値検出を行い、結果を考察せよ
    3. 線形回帰と Ridge 回帰を適用し、以下を比較せよ：
       - $R^2$, RMSE の値
       - 回帰係数の違い
    4. 5-fold CV の結果を報告し、モデルの安定性を議論せよ
    5. 正則化パラメータ $\alpha$ を変化させた検証曲線を作成し、最適値を決定せよ

    **提出物**: レポート（A4 4〜6ページ）+ 使用コード

    ---

    ### 課題 2: 過学習の理解と可視化（基礎〜応用）

    **テーマ**: 多項式回帰を用いて過学習現象を可視化・分析せよ。

    **要件**:
    1. 1変数の特徴量を選び、多項式の次数 $d = 1, 3, 5, 10, 15$ でフィッティングを行え
    2. 各次数における訓練誤差とテスト誤差をグラフ化し、過学習が起きる次数を特定せよ
    3. バイアス-バリアンス分解の数式を用いて、結果を理論的に説明せよ
    4. Ridge 正則化（$\alpha = 0.01, 0.1, 1, 10, 100$）を適用し、過学習が抑制されることを示せ
    5. 学習曲線を作成し、データ数が増えた場合の効果を議論せよ

    **提出物**: レポート（A4 4〜6ページ）+ 使用コード

    ---

    ### 課題 3: 分類問題と材料設計（応用）

    **テーマ**: 合金の結晶構造（BCC/FCC/HCP）を分類するモデルを構築し、材料設計指針を議論せよ。

    **要件**:
    1. 結晶構造分類データの探索的分析を行え
    2. SVM（カーネル: linear, rbf, poly）を適用し、各カーネルの正解率を比較せよ
    3. 混同行列を作成し、どのクラスの分類が難しいか考察せよ
    4. k-means クラスタリングを行い、教師なし学習で結晶構造の分離が可能か議論せよ
    5. エルボー法とシルエットスコアにより最適なクラスタ数を決定せよ
    6. 以上の結果から、**新しい合金を設計する際のVECや原子半径差の指針**を提案せよ

    **提出物**: レポート（A4 5〜7ページ）+ 使用コード

    ---

    ### 課題 4: 特徴量重要度とモデル選択（応用）

    **テーマ**: Random Forest と Lasso を用いた特徴量の重要度評価とモデル選択を行え。

    **要件**:
    1. 鉄鋼データセットを用い、Random Forest の MDI（Mean Decrease Impurity）で特徴量重要度を算出せよ
    2. Lasso の正則化パスにおいて、係数が0になる順序を報告せよ
    3. RF の重要度上位3特徴量と Lasso で残る特徴量を比較し、一致・不一致を議論せよ
    4. 重要特徴量のみ（上位3個）を使ったモデルと全特徴量モデルの性能を CV で比較せよ
    5. **材料科学的な観点**から、重要とされた特徴量が物理的に妥当かどうか考察せよ
       （例: 炭素量が鉄鋼の強度に支配的なのは冶金学的に正しいか？）

    **提出物**: レポート（A4 5〜7ページ）+ 使用コード

    ---

    ### 課題 5: MI ワークフローの総合実践（発展）

    **テーマ**: 以下のいずれかの材料データベースから自分でデータを取得し、MIの一連のワークフローを実践せよ。

    **データソース（いずれか1つ）**:
    - [Materials Project](https://materialsproject.org/) — API でデータ取得
    - [OQMD](https://oqmd.org/) — 形成エネルギーの予測
    - [Matminer](https://hackingmaterials.lbl.gov/matminer/) — `matminer.datasets.load_dataset()` を使用
    - [MatBench](https://matbench.materialsproject.org/) — 標準ベンチマーク

    **要件**:
    1. データ取得・前処理（欠損値処理、特徴量エンジニアリング）
    2. 探索的データ分析（PCA、ペアプロット、外れ値検出）
    3. 回帰 or 分類モデルの構築（最低3種類のモデルを比較）
    4. 交差検証による汎化性能評価
    5. 特徴量重要度の分析と材料科学的解釈
    6. 結果に基づく**材料設計指針の提案**

    **提出物**: レポート（A4 6〜8ページ）+ 使用コード + データ取得スクリプト

    ---
    """)

    st.header("8.3 レポート作成のガイドライン")
    st.markdown("""
    ### 構成
    1. **目的**: 何を予測/分類し、なぜそれが重要か
    2. **データ**: 使用データの概要、出典、前処理
    3. **手法**: 使用した手法とその数式（本アプリの数式を参照可）
    4. **結果**: 図表を用いた定量的な結果
    5. **考察**: 結果の材料科学的解釈、モデルの限界
    6. **結論**: 得られた知見のまとめ

    ### 注意事項
    - 図には必ず**軸ラベル**と**タイトル**をつけること
    - 数式は LaTeX 形式で正しく記述すること
    - コードは Python（scikit-learn 推奨）で再現可能な形式にすること
    - **他の学生のコードのコピーは不可**（データ分析の結果は必ず異なるはず）

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
    5. Dunn, A., et al. (2020). "Benchmarking materials property prediction methods: the Matbench test suite." *npj Computational Materials*, 6, 138.
    """)
