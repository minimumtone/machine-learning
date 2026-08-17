# B2-NiAl 不定比相解析と 4SL/8SL 副格子モデル接続：統合レポート

この文書は、B2-NiAl の MLIP 再現検証レポートと、得られた点欠陥エネルギーを 4SL/8SL CALPHAD 副格子モデルに接続する設計書を統合したものです。B2 不定比解析（Part I）で得られた欠陥形成エネルギーが、Part II の 4SL/8SL パラメータ化の出発データとして用いられます。

---

## Part I: B2-NiAl 不定比・凸包・欠陥化学 検証

## 検証1: B2不定比Ni-Al組成の格子定数・平均原子体積のMLIP再現

対象: T. Yamanouchi and S. Miura, *Mater. Trans.* **59**, 546–555 (2018), Fig. 6(a) のB2-NiAl不定比枝（開丸：実験の平均原子体積）。DOI: 10.2320/matertrans.MJ201604。

熱処理温度は **1473 K**（168 h → 水焼入れ）だが、測定された格子定数・平均原子体積は **室温** であることに注意。

### 主要結果（要約）

| 項目 | 結果 |
|---|---|
| 完全 B2 | $\bar V=11.975$ Å³/atom, $a=2.882$ Å |
| Fig. 6(a) 再現 ($0.45\le x_{\rm Al}\le0.60$) | RMSE = **0.158 Å³/atom**, MAPE = **0.99 %** ($n=12$) |
| B2 単相域の上限 $x_{\max}$ (0 K 全相凸包) | 反サイトのみ 0.492；最安定 B2 枝 0.520：実験 ~0.60 を 0 K では再現しない。$\sim0.08$ の差は有限温度効果 |
| B2 単相域の上限 $x_{\max}$ (1473 K 凸包) | **0.660** (3–20 meV tol)；$x=0.67$ の次点で凸包から +0.06 eV と乖離するため上限が括られており、実験の 0.60 を挟む |
| 格子定数傾き $da/dx$ (Al-rich) | MACE $-0.39$ Å/$x_{\rm Al}$；Taylor & Doyle (1972) $-0.14$ Å/$x_{\rm Al}$ → **約 2.9 倍の過大評価** |
| 欠陥支配性 | Ni 過剰：Ni 反サイト；Al 過剰：Ni 構造空孔（構成欠陥 / 三重欠陥傾向） |
| 4SL/8SL 秩序化強度 | $V\approx -0.10$ 〜 $-0.15$ eV/bond |

### 1. 計算条件

- **MLIP**: MACE-MP-0 medium, float64, CPU
- **緩和**: `LBFGS(FrechetCellFilter)`, $f_{\max}=0.02$ eV/Å, max 500 ステップ
- **スーパーセル**: B2 4×4×4 慣用超胞（128 サイト = 64 B2 単位胞）
- **欠陥様式**:
  - Ni 過剰側 ($x_{\mathrm{Al}}<0.5$): Ni が Al サイトに入る **反サイト**、または Al サイトに **空孔**
  - Al 過剰側 ($x_{\mathrm{Al}}>0.5$): Ni サイトに **空孔**、または Ni サイトに Al が入る **反サイト**
- **サンプリング**: 各組成・各枝で乱数配置を 3 シード作成。追加で x=0.2–0.4 / 0.6–0.8 の代表組成、x=0.42–0.58 の競合、x=0.61–0.79 の Al-rich 密サンプリング（0.02 刻み）を実施。
- **定義の明確化**:
  - 体積は **占有原子あたり** $\bar V=V/N_{\rm atom}$ と **サイトあたり** $\bar V_{\rm site}=V/N_{\rm site}$ の両方を出力。B2 空孔枝では両者は $1-c_{\rm vac}$ だけずれる。
  - 元素化学ポテンシャル（MACE-MP-0 fcc 基準）：$\mu_{\rm Ni}=-5.7319$ eV/atom, $\mu_{\rm Al}=-3.7284$ eV/atom。
  - 自由エネルギーは **占有原子あたりの Helmholtz 自由エネルギー** $G_i$ を用いる。$G_i = (E_i - \mu_{\rm Ni}N_{\rm Ni} - \mu_{\rm Al}N_{\rm Al})/N_{\rm atom} - k_{\rm B}T \ln g_i/N_{\rm atom}$、$g_i=C(64,n_{\rm defect})$。
  - 全系の半巨視正準ポテンシャルは $\Omega_i = N_{\rm atom}G_i$ なので、固定組成での枝比較には $G_i$ を使えば十分。

実験基準: Fig. 6(a) 開丸を PDF から Hough 円検出＋軸校正でデジタイズ（`analysis/fig6a_digitized_circles.csv`、読取精度 ±0.05 Å³）。$x>0.60$ の点は Ni$_2$Al$_3$/NiAl$_3$ に相当するため、B2 単相枝の再現度評価から除外した。

### 2. 点欠陥形成エネルギー

完全 B2 から欠陥 1 個あたりのエネルギー（`analysis/b2_defect_energies.csv`）：

| 欠陥種 ($x$ 領域) | 平均 ΔE (eV/欠陥) | 標準偏差 | 支配性 |
|---|---|---|---|
| Ni 反サイト on Al サイト（Ni 過剰） | 0.79 | 0.20 | Ni 過剰側で最低エネルギー |
| Al 空孔（Ni 過剰） | 1.78 | 0.15 | 常に Ni 反サイトより高い |
| Ni 空孔（Al 過剰） | 1.17 | 0.08 | Al 過剰側で最低エネルギー |
| Al 反サイト on Ni サイト（Al 過剰） | 1.59 | 0.10 | 常に Ni 空孔より高い |

これは B2-NiAl の **構成欠陥（constitutional defect）パターン**（Ni 過剰：Ni 反サイト；Al 過剰：Ni 構造空孔）である。熱平衡で言えばこれらが集まり **三重欠陥錯体** $2V_{\rm Ni}+{\rm Ni}_{\rm Al}$ として記述されることもある（ここでの表は構成欠陥パターン）。

**重要**: 個々の点欠陥形成エネルギーは元素化学ポテンシャルの取り方に敏感である。B2 単相で
$$\mu_{\rm Ni}+\mu_{\rm Al}=E_{\rm B2}/{\rm formula}=-10.844\ {\rm eV}$$
という拘束のもと，$\Delta\mu=\mu_{\rm Al}-\mu_{\rm Ni}$ には約 **2.768 eV** の自由度がある。$\mu_{\rm Ni}$ の可動域が 1.384 eVで $d(\Delta\mu)=-2\,d\mu_{\rm Ni}$ なので、合計 2.768 eV 動く。したがって「Ni 過剰=反サイト，Al 過剰=空孔」という結論は特定の $\Delta\mu$ に依存しており，$E_f$ vs $\Delta\mu$ プロットと Korzhavyi et al. (Phys. Rev. B **61**, 6003) との比較が必要（次ステップ）。

### 3. Fig. 6(a) B2 枝の再現（拘束 B2 単相）

**この節は B2 相を強制した拘束計算（実験も焼入れ・準安定試料）に対応する。§4 では無拘束の相安定性（凸包）を扱う。**

- 完全 B2: $\bar V=11.975$ Å³/atom（実験 ~12.0）, $a=2.882$ Å（Taylor & Doyle 2.887）
- 比較は B2 の均一域 **$0.45\le x_{\rm Al}\le0.60$** に限定。$x>0.60$ のデジタイズ点は Ni$_2$Al$_3$/NiAl$_3$ 等の中間化合物に相当するため，B2 枝の再現度評価から除外した。
- 各枝の混合は **占有原子あたりの Helmholtz 自由エネルギー $G_i$**（半巨視正準 + 解析的配置エントロピー）を用いて評価：

$$G_i = \frac{E_i - \mu_{\rm Ni}N_{\rm Ni} - \mu_{\rm Al}N_{\rm Al}}{N_{\rm atom}} - k_{\rm B}T\frac{\ln g_i}{N_{\rm atom}},\qquad g_i=C(64,n_{\rm defect})$$

- 各組成で $G_i$ が低い枝を採用。これにより Al 過剰側では Ni 構造空孔枝が選ばれ、体積の急速な増大が再現される。

| 指標 | 値 |
|---|---|
| 比較点数 | 12 |
| RMSE（体積） | 0.158 Å³/atom |
| MAPE | 0.99 % |

残留誤差 0.158 Å³/atom は、完全 B2 体積 12 Å³ に対して約 1.3 %。これは §7 の MACE 格子定数誤差（体積で 0.5–1.5 %）と同じオーダーであり、**既に MACE-MP-0 の体積精度の床に到達している**。重み付けモデルを更に磨いても大きな改善は困難で、より高精度が必要なら NiAl 専用の MLIP へ移るのが妥当である。

### 4. B2 単相域の上限 $x_{\max}$（0 K・1473 K 凸包）

**この節は無拘束の相安定性を扱う。§3 とは問いが異なることに注意。**

`analyze_b2_hull_xmax.py` で、純元素・fcc-SQS・L1$_2$-Ni$_3$Al・Ni$_3$Al$_4$・Ni$_5$Al$_3$・Ni$_2$Al$_3$・NiAl$_3$ を含む **0 K 全相凸包** と、1473 K では Ni$_3$Al$_4$ / Ni$_5$Al$_3$ を除いた **高温凸包**（Ni, Ni$_3$Al, NiAl, Ni$_2$Al$_3$, NiAl$_3$）を作成した。B2 枝には $-TS_{\rm conf}(x)$ を追加。

$x_{\max}$ は「凸包からの乖離が許容値以下で最も Al-rich な組成」と定義した。判定許容値を 3 / 5 / 10 / 20 meV/atom で振った感度を以下に示す。

| 凸包・枝 | 3 meV | 5 meV | 10 meV | 20 meV |
|---|---|---|---|---|
| 0 K 全相 / 最安定 B2 枝 (Ef) | 0.512 | 0.520 | 0.529 | 0.561 |
| 0 K 全相 / 反サイトのみ (Ef) | 0.492 | 0.492 | 0.508 | 0.508 |
| 1473 K / 最安定 B2 枝 (G) | 0.660 | 0.660 | 0.660 | 0.660 |
| 実験（Ellner / Yamanouchi Fig. 6） | ~0.60 | ~0.60 | ~0.60 | ~0.60 |

**解釈**: 0 K 全相凸包では、最安定 B2 枝（空孔を含む）が $x_{\max}\approx0.52$ で止まり、実験 0.60 をかなり見誤る。Ni$_3$Al$_4$ / Ni$_5$Al$_3$ は 700 °C 以下の低温相なので、1473 K 比較では除外すべき。これらを除き、B2 枝に 1473 K の理想的配置エントロピーを加えると、$x_{\max}\approx0.66$ まで広がる。$x=0.67$ の次点で最安定 B2 枝は凸包から +0.06 eV と離れるので、上限は 0.66 近傍で括られており、**実験の 0.60 を挟む**。残る差は、中間化合物の熱膨張・振動エントロピー、MACE の中間化合物エネルギー精度、統計誤差に依存する。

Al-rich 反サイト単相は 0 K 凸包から約 0.09 eV/atom、空孔単相は約 0.03–0.10 eV/atom と凸包から乖離する。これは「二相分離」ではなく **B2 単相域の上限** を示す。空孔を含めた最安定 B2 枝は $x_{\rm Al}\approx0.52$ で上限に達し、$T=1473$ K の配置エントロピーで 0.66 付近まで広がる。

参照: `analysis/b2_branch_finiteT_hull.csv`, `analysis/b2_xmax_sensitivity.csv`, `figures/fig_b2_hull_finiteT.png`。

### 5. 格子定数 $a(x)$ の異常

Al 過剰側では $\bar V$ が急増する一方、**実効格子定数 $a$ は緩やかに減少**する。これは Ni サイトに構造空孔が入ることによる密度欠損の特徴で、構造空孔の「指紋」である。

Bradley & Taylor (1937) のデータは小数 2 桁で精度が主張と釣り合わないため、一次出典である **Taylor & Doyle, J. Appl. Cryst. 5 (1972) 201** の線形フィットを直接使用した。原論文の Abstract から、B2-NiAl の格子定数は：

- Ni-rich 側: 2.8870 Å (50 at.% Ni) → 2.8618 Å (66 at.% Ni)
- Al-rich 側: 2.8870 Å (50 at.% Ni) → 2.8652 Å (34 at.% Ni)

と線形に変化する。これを $x_{\rm Al}=1-x_{\rm Ni}$ に変換して MACE の安定枝と比較した。主な比較は傾き $da/dx$ と $d\bar V/dx$ で行う。点ごとの差は系統的オフセットに敏感であるが、傾きは空孔の有無で符号が変わるため、物理的に判別力が高い。

| $x_{\rm Al}$ | $a_{\rm T\&D}$ (Å) | $a_{\rm MACE}$ (Å) | $\Delta a$ (Å) |
|---|---|---|---|
| 0.45 | 2.879 | 2.873 | -0.006 |
| 0.48 | 2.884 | 2.879 | -0.005 |
| 0.50 | 2.887 | 2.882 | -0.005 |
| 0.52 | 2.884 | 2.874 | -0.010 |
| 0.55 | 2.880 | 2.861 | -0.019 |
| 0.58 | 2.876 | 2.849 | -0.027 |
| 0.60 | 2.873 | 2.842 | -0.032 |
| 0.62 | 2.871 | 2.834 | -0.037 |

**傾き比較**

| 領域 | $da/dx$ MACE (Å/$x_{\rm Al}$) | $da/dx$ Taylor & Doyle (Å/$x_{\rm Al}$) | $d\bar V/dx$ MACE (Å³/atom/$x_{\rm Al}$) | $d\bar V/dx$ Taylor & Doyle (Å³/atom/$x_{\rm Al}$) |
|---|---|---|---|---|
| Ni-rich (0.42–0.50) | +0.20 | +0.16 | +2.47 | +1.96 |
| Al-rich (0.50–0.66) | **-0.39** | **-0.14** | **+17.69** | **+5.71** |

MACE は Ni-rich 側の傾きを約 $+0.20$ Å/$x_{\rm Al}$ で再現する（Taylor & Doyle: $+0.16$；+26 %）。一方、Al-rich 側では MACE $-0.39$ Å/$x_{\rm Al}$ に対し Taylor & Doyle $-0.14$ Å/$x_{\rm Al}$ と**約 2.9 倍に過大評価**する。$d\bar V/dx$ でも Al-rich MACE (+17.69) は実験値 (+5.71) の 3.1 倍である。これは MACE が Ni 空孔形成に伴う局所緩和を強く出しすぎており、格子定数の変化率は体積再現精度とは独立の、より厳しい検証指標であることを示している。

参照: `analysis/taylor_doyle_a_reconstructed.csv`, `analysis/a_comparison_taylor_doyle_mace.csv`, `analysis/taylor_doyle_mace_slopes.csv`, `figures/fig_b2_a_taylor_doyle_overlay.png`。

### 6. 凸包・中間化合物

`07_reports/fig_energy_diagram_nial_generator.py` により、純元素に加え L1$_2$-Ni$_3$Al，**Ni$_3$Al$_4$**，Ni$_5$Al$_3$，Ni$_2$Al$_3$，NiAl$_3$ の MACE-MP-0 緩和エネルギーを含めた **0 K 真の凸包** を作成した。Ni$_3$Al$_4$ ($x_{\rm Al}=0.571$, $E_f=-0.649$ eV/atom) は 0 K の B2 均一域の上限近傍を強く安定化する。ただし Ni$_3$Al$_4$ / Ni$_5$Al$_3$ は 700 °C 以下の低温相であり、1473 K の比較では除外する。

有限温度凸包（1473 K）では、これら低温相を除き、B2 枝の理想配置エントロピーを競合相に対して加算する。作成した図・CSV から、$x_{\max}(1473 \text{ K})$ は 0.66 付近に達し、実験の 0.60 を挟む（§4）。

参照: `analysis/mace_mp_ref_results.csv`, `mp_reference_structures.json`。

### 7. MLIP 誤差評価（MACE vs MP-DFT vs 実験）

`analysis/BENCHMARK_MACE_vs_MP_vs_EXP.md`:

| 構造 | $x_{\rm Al}$ | MACE $a$ (Å) | MP-PBE $a$ (Å) | 実験 $a$ (Å) | MACE 誤差 (%) |
|---|---|---|---|---|
| fcc-Ni | 0.000 | 3.5098 | 3.4751 | 3.524 | -0.40 |
| fcc-Al | 1.000 | 4.0602 | 4.0389 | 4.050 | +0.25 |
| B2-NiAl | 0.500 | 2.8819 | 2.8597 | 2.887 | -0.18 |
| L1$_2$-Ni$_3$Al | 0.250 | 3.5545 | 3.5231 | 3.572 | -0.49 |

| 構造 | $x_{\rm Al}$ | MACE $a$ (Å) | 実験 $a$ (Å) | MACE 誤差 (%) |
|---|---|---|---|---|
| Ni$_3$Al$_4$ | 0.571 | ? | 3.56 | ? |

MACE-MP-0 medium は格子定数を 0.5 % 未満で再現しており、Fig. 6 議論の 1–2 % 差異を扱う上で十分な精度がある。§3 の RMSE = 0.158 Å³/atom は、§7 の体積再現精度とほぼ同じ床（≈1 %）であり、同一 MLIP 内ではこれ以上の精密化には限界がある。

### 8. 4SL/8SL B2 副格子モデルへの接続

`4SL_B2_MODEL_DESIGN.md`、`b2_offstoich/extract_b2_defect_energies.py`、および `b2_offstoich/extract_4sl_b2_parameters.py` で，MLIP から組成依存な空孔/反サイト形成エネルギーを抽出し，4-sublattice / 8-sublattice CEF モデルのエンドメンバー・パラメータに変換する手順を設計。

- **副格子置換対称性**: 等価なサブラティスの入れ替えに対して Gibbs エネルギーが不変でなければならない（Ansara / Dupin / Sundman）。4SL/8SL モデルではこの対称性を課すことで，256 エンドメンバーの過剰決定・偽の秩序相出現を防ぐ。
- **秩序化強度**: 第一近接対相互作用の線形結合から得られる有効秩序化エネルギーは

  $$V = J_{\rm NiAl} - \frac{J_{\rm NiNi}+J_{\rm AlAl}}{2}$$

  の一つの値のみが物理的に意味を持つ。現状の簡易 Ising 外挿では $V \approx -0.1$ 〜 $-0.15$ eV/bond（Ni–Al 結合が他の結合より約 0.1 eV 強く負）。個別の $J_{\rm NiAl}, J_{\rm NiNi}, J_{\rm AlAl}$ の絶対値は A2 エンドメンバー・クラスター展開がないため信頼できない（詳細：`analysis/b2_pair_interactions.json`）。
- 元素化学ポテンシャルではなく **B2 単相の自由エネルギー曲面** を介して正しい欠陥優位を記述するための第一ステップである。

### 9. 限界と次ステップ

1. 0 K 静的計算（実験は室温 XRD）。熱膨張で ~0.5–1 %。
2. 欠陥形成エネルギーの $\Delta\mu$ 依存性を明示し，$E_f$ vs $\Delta\mu$ 図で Korzhavyi et al.（Phys. Rev. B **61**, 6003）と比較する。
3. Bradley & Taylor の密度測定から導かれる構造空孔濃度と MACE 予測を $x=0.55,0.60$ で直接比較する。
4. MACE-MP-0 に磁性・スピン自由度がない（Ni-rich 端）。
5. 各組成 3 配置のサンプリングは統計的に希薄。最低 10–20 配置，または `icet` クラスター展開に移行。
6. 1473 K の熱処理の振動エントロピー / MD 平衡化は未実装。
7. 4SL/8SL 副格子モデルのエンドメンバー対称性拘束を実装し，pycalphad/TDB 形式の原型を出力する。

### 10. 主要成果物

- `run_b2_offstoich*.py`, `make_figures.py`, `plot_a_taylor_doyle_overlay.py`, `extract_b2_defect_energies.py`
- `analyze_b2_hull_xmax.py`, `analyze_alrich_dense.py`, `extract_4sl_b2_parameters.py`, `add_ni3al4.py`
- `analysis/b2_defect_energies.csv`, `analysis/b2_branch_hull_xmax.csv`, `analysis/b2_xmax.json`, `analysis/b2_xmax_sensitivity.csv`, `analysis/b2_alrich_dense_phase_stability.csv`, `analysis/b2_branch_finiteT_hull.csv`
- `analysis/taylor_doyle_a_reconstructed.csv`, `analysis/a_comparison_taylor_doyle_mace.csv`, `analysis/taylor_doyle_mace_slopes.csv`
- `analysis/BENCHMARK_MACE_vs_MP_vs_EXP.md`（Ni$_3$Al$_4$ 含む）
- `analysis/TABLE4_MACE_ANALOGUE.md`, `analysis/C14_YAMANOUCHI_WEIGHTED_CHECK.md`
- `analysis/b2_pair_interactions.json`
- `4SL_B2_MODEL_DESIGN.md`
- `07_reports/fig_energy_diagram_nial.png`（真の Ni-Al 凸包図）
- `figures/fig_b2_offstoich_vbar*.png`, `fig_b2_offstoich_a.png`, `fig_b2_offstoich_eform.png`, `fig_b2_alrich_dense_hull.png`, `fig_b2_hull_xmax.png`, `fig_b2_hull_finiteT.png`, `fig_b2_a_taylor_doyle_overlay.png`

### 付録 A. Yamanouchi Table 4 / C14-Nb(Ni,Al)$_2$ 検証

- `analysis/TABLE4_MACE_ANALOGUE.md`: B2/fcc 双方から導いた原子直径（CN8）を CoAl/PdAl/RhAl/IrAl で比較。
- `analysis/C14_YAMANOUCHI_WEIGHTED_CHECK.md`: C14-Nb(Ni,Al)$_2$（$x=0.5$）の観測体積は純元素平均より **化合物由来の加重平均** に近いことを再確認。

**Table 4 検証について**: MACE は B2 由来と fcc 由来の $D_B$ の差の符号と桁を再現するが、**絶対値を約 2 倍過大評価**する（MACE 0.03–0.06 Å に対し実験 0.01–0.03 Å）。これは格子定数再現精度（§7、0.5 %以内）から伝播する ±0.02 Å の誤差帯と同程度であり、MACE-MP-0 の分解能の限界にある。ただし純元素平均を用いた場合の誤差（4–5 %）に比べれば 1 桁小さいため、Yamanouchi & Miura の定性的結論——$D_B$ は B2 の格子定数から取るべき——は MACE でも支持される。本稿の主線（B2 不定比の体積・格子定数・均一域幅再現）とは独立した副次検証である。


---

## Part II: MLIP 空孔・アンチサイトエネルギー → 4SL/8SL B2 副格子モデル パラメータ化検討

## MLIP 空孔・アンチサイトエネルギー → 4SL/8SL B2 副格子モデル パラメータ化検討

Yamanouchi & Miura（2018）の補助議論（Tsang スライド）を受け、MLIP 計算から B2-NiAl の点欠陥エネルギーを取り出し、CALPHAD 的 4-sublattice/8-sublattice B2 秩序相モデルのエンドメンバー・パラメータに接続する方法を整理する。

### 1. 現状の MLIP データ

MACE-MP-0 medium（0 K 静的緩和、FrechetCellFilter、128 サイト超胞）から得られた点欠陥形成エネルギー（完全 B2 からの差分、fcc 元素基準）：

| 欠陥種 | 平均 ΔE (eV/欠陥) | 標準偏差 | 備考 |
|---|---|---|---|
| Ni 反サイト on Al 副格子（Ni-rich） | 0.79 | 0.20 | 濃度依存あり、x≈0.46 で 0.68 eV、x≈0.20 で 1.17 eV |
| Al 反サイト on Ni 副格子（Al-rich） | 1.59 | 0.10 | x≈0.80 で 1.37 eV、x≈0.60 で 1.62 eV |
| Ni 空孔（Al-rich、Ni 副格子）     | 1.17 | 0.08 | x≈0.50 で 1.06 eV、x≈0.54 で 1.28 eV |
| Al 空孔（Ni-rich、Al 副格子）     | 1.78 | 0.15 | 常に Ni 反サイトより高エネルギー |

重要な知見：
- **Ni-rich 側**：Ni 反サイト （~0.7–0.9 eV） << Al 空孔 （~1.7–2.0 eV） → 反サイトが支配的。
- **Al-rich 側**：Ni 空孔 （~1.1 eV） < Al 反サイト （~1.4–1.6 eV） → 空孔が支配的。
- ΔE は欠陥濃度に依存するため、単一点欠陥近似ではなく **濃度依存な相互作用項**が必要。

### 2. 2 サブラティス CEF モデル（出発点）

最も簡単な B2 記述は

```
(Ni, Al, Va)_{0.5} (Al, Ni, Va)_{0.5}
```

エンドメンバー（完全秩序・主な点欠陥）を以下のように定める：

| エンドメンバー | 意味 | MLIP からの見積もり |
|---|---|---|
| G(Ni:Al) | 完全 B2-NiAl | MACE E(NiAl-B2)/formula = -10.844 eV |
| G(Al:Ni) | 反 B2-NiAl（Al/Ni サブラティス入れ替え） | = G(Ni:Al)（B2 は C.N.8 対称） |
| G(Ni:Ni) | 反サイトに近い Ni 過剰極限 | A2-Ni（bcc）への外挿 or x→0.5 Ni-rich 極限 |
| G(Al:Al) | Al 過剰極限 | A2-Al（bcc）への外挿 or x→0.5 Al-rich 極限 |
| G(Va:Al) | Ni 欠損 Al 過剰（Al-rich 空孔） | G(Ni:Al) + n_sites/2 · E_Ni_vac |
| G(Ni:Va) | Al 欠損 Ni 過剰（Ni-rich 空孔） | G(Ni:Al) + n_sites/2 · E_Al_vac |

注：現状では A2-Ni/Al の MLIP 計算がないため、x=0.5 付近の点欠陥データを用いてエンドメンバーを **x 依存として表現**し、相互作用パラメータに変換する方法を採用する。

### 3. 4SL/8SL B2 モデルへの一般化

4 サブラティスモデルは B2 の 2 つの簡単立方サブラティスをそれぞれ 2 つに分割：

```
(Ni,Al,Va)_{1/4}(Ni,Al,Va)_{1/4}(Ni,Al,Va)_{1/4}(Ni,Al,Va)_{1/4}
```

- (α1, α2) は元の Ni 副格子を 2 分割、
- (β1, β2) は元の Al 副格子を 2 分割。

利点：
1. 同じ副格子内でもローカル環境（例：Al@2a vs Al@6h のような Laves サイト分割思考）を表現可能。
2. B2/B32/DO3/A2 などのさまざまな秩序度を同一モデルで扱える。

8 サブラティスはさらに最近接対を細かく区別し、**第一近接 Ni–Al / Ni–Ni / Al–Al 対エネルギー**を別々のパラメータに結びつける。MLIP からは `icet` 等のクラスター展開でこれらの対相互作用を直接抽出できる。

**副格子置換対称性**: 4SL/8SL CEF では，等価なサブラティス（例：α1 と α2，β1 と β2）の置換に対してモデルの総 Gibbs エネルギーが不変でなければならない（Ansara / Dupin / Sundman）。この対称性を課さないと，エンドメンバー数が過剰決定となり，実在しない低対称性の偽秩序相が計算上出現しうる。TDB 化する際は，各等価サブラティス群に対して同じ Gibbs 関数を割り当てるか，対称性に基づくエンドメンバーの縮約が必要である。

### 4. MLIP → エンドメンバー・パラメータ変換の手順

1. **配置エネルギーの収集**：128 サイト B2 超胞で、全組成（x_Al=0.20–0.80）の反サイト・空孔配置を MACE で緩和。
2. **点欠陥形成エネルギーの定義**：式 (1)–(4) に従い、完全 B2 からの差分および元素化学ポテンシャルで正規化。
3. **クラスター展開（任意）**：icet/BOMD 等で Ni–Al, Ni–Ni, Al–Al 第一近接対の有効相互作用エネルギー J_ij を推定すれば、8SL パラメータとして直接使用可能。
4. **CEF エネルギー関数の構築**：
   
   $$ G_m = \sum_i y_i^1 y_j^2 y_k^3 y_l^4 \, G_{ijkl}^{\rm end} + RT\sum_s \sum_i y_i^s \ln y_i^s + G_{\rm excess} $$

   ただし $y_i^s$ はサブラティス $s$ の種 $i$ の占有位点分率、$G_{ijkl}^{\rm end}$ はエンドメンバー自由エネルギー。
5. **濃度依存 interaction の導入**：
   
   $$ G_{\rm excess} = \sum_s y_i^s y_j^s \, L_{ij}^s + y_k^t y_l^t \, L_{kl}^t + \dots $$

   ここで $L$ は Redlich–Kister 多項式 $L = L^0 + L^1 (y_i - y_j) + \dots$。
6. **パラメータフィット**：MLIP の $\Delta E_{\rm defect}(x)$ をターゲットとして、$G_{\rm end}$ と $L$ を最小二乗フィット。MACE の形成エネルギー図（凸包）も追加制約として含める。

### 5. 現時点での制約・未解決点

- **振動エントロピーなし**：1473 K の熱処理には phonon/MD によるエントロピー補正が必要。
- **磁性**：Ni のスピン分極効果は MACE-MP-0 にはない。
- **A2 端成分未計算**：完全 A2-Ni/A2-Al の MLIP データがないため、B2-A2 秩序変態エネルギーは外挿依存。
- **サンプリングの希薄さ**：各組成 3 配置では 4SL/8SL の全エンドメンバーをカバーできない。最低 10–20 配置、さらにクラスター展開用データが必要。

### 6. 次の実行計画

1. 現行の Al-rich 密サンプリング完了を待ち、`b2_defect_energies.csv` を更新。
2. A2-Ni/A2-Al（bcc）の参考計算を追加し、完全秩序化エネルギーを推定。
3. `icet` クラスター展開で第一近接対相互作用 J_{ij} を抽出し、8SL 対応エンドメンバー表を作成。
4. pycalphad/TDB 形式の原型を出力し、形成エネルギー図と整合するか検証。

### 7. 付録：第一近接対相互作用の見積もり（Ising 近似）

B2 構造の最近接対を三種類の有効相互作用で近似する：

$$ E = \sum_{\langle ij \rangle} J_{ij}, \qquad J_{\rm NiAl} < 0 \text{（安定化）} $$

慣用胞（2 原子）あたりの結合数は 8。完全 B2-NiAl では全結合が Ni–Al：

$$ E_{\rm B2} = 8 J_{\rm NiAl} $$

完全ランダム A2（各サイト占有率 0.5）では Ni–Ni:Al–Al:Ni–Al = 2:2:4：

$$ E_{\rm A2} = 2 J_{\rm NiNi} + 2 J_{\rm AlAl} + 4 J_{\rm NiAl} $$

したがって、**秩序化エネルギー**は

$$ \Delta E_{\rm order} = E_{\rm A2} - E_{\rm B2} = 2(J_{\rm NiNi} + J_{\rm AlAl}) - 4 J_{\rm NiAl} $$

MACE-MP-0 から $E_{\rm B2} \approx -10.844$ eV/formula、$E_{\rm A2}$ は $b2\_order\_param.csv$ の $\eta=0$ 平均から $E/N \approx -5.246$ eV/atom、すなわち $E_{\rm A2} \approx -10.492$ eV/formula、よって

$$ \Delta E_{\rm order} \approx +0.35\text{–}0.40 \text{ eV/formula} $$

重要なことは，個別の $J_{ij}$ の絶対値ではなく，Ni–Al 結合が他の結合よりどれだけ強い負かを表す **秩序化強度（effective ordering energy）**

$$ V = J_{\rm NiAl} - \frac{J_{\rm NiNi} + J_{\rm AlAl}}{2} $$

のみが物理的に意味を持つ。現状の簡易 Ising 外挿では

$$ V \approx -0.1 \text{ 〜 } -0.15 \ {\rm eV/bond} $$

（Ni–Al 結合が他より約 0.1 eV 強く負）という数量級が得られる。個別の $J_{\rm NiAl}, J_{\rm NiNi}, J_{\rm AlAl}$ については A2 端成分やクラスター展開なしには決定できないため，**3 つの $J$ 値を表にするのは避け，$V$ の値のみを報告する**。

定数対モデルには明らかな制限がある：$E_{\rm A2}$ の予測は観測より 0.25 eV/formula 高くなり，高濃度反サイト/空孔同士の相互作用を見落としている。これは 4SL/8SL モデルでは組成依存な $L$ パラメータで補正するか，`icet` クラスター展開で対相互作用を直接抽出する必要がある。

### 8. 成果物

- `b2_offstoich/analysis/b2_defect_energies.csv`（各配置の空孔/反サイト形成エネルギー）
- 本設計書 `4SL_B2_MODEL_DESIGN.md`
- `b2_offstoich/extract_4sl_b2_parameters.py`（秩序化強度 $V$ の簡易推定）
- `b2_offstoich/analysis/b2_pair_interactions.json`（$V$ の値のみを使用。個別 $J$ 表は非推奨）

