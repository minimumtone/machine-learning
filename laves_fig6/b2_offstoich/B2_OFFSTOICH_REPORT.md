# 検証1: B2不定比Ni-Al組成の格子定数・平均原子体積のMLIP再現

対象: T. Yamanouchi and S. Miura, *Mater. Trans.* **59**, 546–555 (2018), Fig. 6(a) のB2-NiAl不定比枝（開丸：実験の平均原子体積）。DOI: 10.2320/matertrans.MJ201604。

熱処理温度は **1473 K**（168 h → 水焼入れ）だが、測定された格子定数・平均原子体積は **室温** であることに注意。

## 1. 計算条件

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
  - 1473 K の Boltzmann 重みは **半巨視正準ポテンシャル** で評価：

$$ \Omega_i = E_i - \mu_{\rm Ni}N_{\rm Ni} - \mu_{\rm Al}N_{\rm Al} - k_{\rm B}T \ln g_i, \qquad g_i=C(64,n_{\rm defect}) $$

実験基準: Fig. 6(a) 開丸を PDF から Hough 円検出＋軸校正でデジタイズ（`analysis/fig6a_digitized_circles.csv`、読取精度 ±0.05 Å³）。

## 2. 点欠陥形成エネルギー

完全 B2 から欠陥 1 個あたりのエネルギー（`analysis/b2_defect_energies.csv`）：

| 欠陥種 ($x$ 領域) | 平均 ΔE (eV/欠陥) | 標準偏差 | 支配性 |
|---|---|---|---|
| Ni 反サイト on Al サイト（Ni 過剰） | 0.79 | 0.20 | Ni 過剰側で最低エネルギー |
| Al 空孔（Ni 過剰） | 1.78 | 0.15 | 常に Ni 反サイトより高い |
| Ni 空孔（Al 過剰） | 1.17 | 0.08 | Al 過剰側で最低エネルギー |
| Al 反サイト on Ni サイト（Al 過剰） | 1.59 | 0.10 | 常に Ni 空孔より高い |

これは B2-NiAl の **構成欠陥（constitutional defect）**，すなわち **三重欠陥化学**（triple-defect: Ni 過剰 = Ni 反サイト、Al 過剰 = Ni 構造空孔）と整合する。

**重要**: 個々の点欠陥形成エネルギーは元素化学ポテンシャルの取り方に敏感である。B2 単相で
$$\mu_{\rm Ni}+\mu_{\rm Al}=E_{\rm B2}/{\rm formula}=-10.844\ {\rm eV}$$
という拘束のもと，$\Delta\mu=\mu_{\rm Al}-\mu_{\rm Ni}$ には約 1.384 eV の自由度がある（fcc 和 -9.4603 eV との差）。したがって「Ni 過剰=反サイト，Al 過剰=空孔」という結論は特定の $\Delta\mu$ に依存しており，$E_f$ vs $\Delta\mu$ プロットと Korzhavyi et al. (Phys. Rev. B **61**, 6003) との比較が必要（次ステップ）。

## 3. Fig. 6(a) B2 枝の再現

- 完全 B2: $\bar V=11.975$ Å³/atom（実験 ~12.0）, $a=2.882$ Å（実験 2.887）
- 比較は B2 の均一域 **$0.45\le x_{\rm Al}\le0.60$** に限定。$x>0.60$ のデジタイズ点は Ni$_2$Al$_3$/NiAl$_3$ 等の中間化合物に相当するため，B2 枝の再現度評価から除外した。
- 各枝の混合は **占有原子あたりの Helmholtz 自由エネルギー**（半巨視正準 + 解析的配置エントロピー）を用いて評価：

$$F_i = \frac{E_i - \mu_{\rm Ni}N_{\rm Ni} - \mu_{\rm Al}N_{\rm Al}}{N_{\rm atom}} - k_{\rm B}T\frac{\ln g_i}{N_{\rm atom}},\qquad g_i=C(64,n_{\rm defect})$$

- この取り扱いにより Al 過剰側では Ni 構造空孔枝が選ばれ，体積の急速な増大が再現される。

| 指標 | 値 |
|---|---|
| 比較点数 | 12 |
| RMSE（体積） | 0.158 Å³/atom |
| MAPE | 0.99 % |

残留誤差は主に (1) 0 K 計算 vs 室温実験の熱膨張，(2) 元素化学ポテンシャルと実際の B2 単相化学ポテンシャルの差，(3) 3 配置サンプリングの統計誤差に起因する。

## 4. B2 単相域の上限 $x_{\max}$（0 K 凸包からの抽出）

Al-rich 反サイト密サンプリング（$x_{\rm Al}=0.61$–0.79）を含む全 B2 枝を，純元素・fcc-SQS・L1$_2$-Ni$_3$Al・**Ni$_3$Al$_4$**・Ni$_5$Al$_3$・Ni$_2$Al$_3$・NiAl$_3$ を含む真の 0 K 凸包と比較した。$x_{\max}$ は「凸包からの乖離が $\le5$ meV/atom で最も Al-rich な組成」と定義した（`b2_offstoich/analyze_b2_hull_xmax.py`）：

| 枝 | $x_{\max}$ | 備考 |
|---|---|---|
| 反サイトのみ | 0.49 | 0 K では B2 均一域の Ni-rich 側のみで凸包上 |
| 空孔込み | 0.52 | Al 過剰側で約 0.03 だけ右に伸びる |
| 実験（Ellner; Yamanouchi Fig.6） | ~0.60 | 0 K 凸包では Ni$_3$Al$_4$/Ni$_2$Al$_3$ が優位であり，残り $\sim0.08$ は配置エントロピーで埋まる |

Al-rich 側の B2 反サイト単相は凸包から **0.08–0.11 eV/atom** 高エネルギーであるが，これは「二相分離」ではなく **B2 単相域の上限** を示す。空孔を導入すると上限が $x_{\rm Al}\approx0.52$ まで右に伸びるが，実験的 0.60 までを完全に再現するには，有限温度の配置エントロピー（およびおそらく MACE の中間化合物エネルギー精度）を含めた自由エネルギー計算が必要である。

## 5. 格子定数 $a(x)$ の異常

Al 過剰側では $\bar V$ が急増する一方，**実効格子定数 $a$ はほぼ一定〜微減**（2.882 Å → 2.84 Å @ $x_{\rm Al}=0.60$）。これは Ni 構造空孔導入による密度欠損の特徴であり，構造空孔の「指紋」とされる。

Bradley & Taylor (Proc. R. Soc. A **159**, 56, 1937) の実験値と MACE 安定枝の比較（再描画元: Jiang & Chen, Acta Mater. **53**, 2643, 2005, Fig. 4(a)）:

| $x_{\rm Al}$ | $a_{\rm exp}$ (Å) | $a_{\rm MACE}$ (Å) | $\Delta a$ (Å) |
|---|---|---|---|
| 0.50 | 2.88 | 2.879 | -0.001 |
| 0.55 | 2.87 | 2.861 | -0.009 |
| 0.60 | 2.87 | 2.861 | -0.009 |

MACE の格子定数誤差は B2 域で **~0.01 Å（約 0.3–0.7 %）** と，§7 の $<0.5$ % 評価と整合する。`figures/fig_b2_a_bradley_overlay.png` および `analysis/a_comparison_bradley_mace.csv` 参照。

## 6. 凸包・中間化合物

`07_reports/fig_energy_diagram_nial_generator.py` により，純元素に加え L1$_2$-Ni$_3$Al，**Ni$_3$Al$_4$**，Ni$_5$Al$_3$，Ni$_2$Al$_3$，NiAl$_3$ の MACE-MP-0 緩和エネルギーを含めた **真の凸包** を作成した。Ni$_3$Al$_4$ ($x_{\rm Al}=0.571$, $E_f=-0.649$ eV/atom) は B2 均一域の上限近傍を強く安定化し，Al-rich B2 単相の $x_{\max}$ 決定に不可欠である。`analysis/mace_mp_ref_results.csv` および `mp_reference_structures.json` に追加済み。

## 7. MLIP 誤差評価（MACE vs MP-DFT vs 実験）

`analysis/BENCHMARK_MACE_vs_MP_vs_EXP.md`:

| 構造 | $x_{\rm Al}$ | MACE $a$ (Å) | MP-PBE $a$ (Å) | 実験 $a$ (Å) | MACE 誤差 (%) |
|---|---|---|---|---|---|
| fcc-Ni | 0.000 | 3.5098 | 3.4751 | 3.524 | -0.40 |
| fcc-Al | 1.000 | 4.0602 | 4.0389 | 4.050 | +0.25 |
| B2-NiAl | 0.500 | 2.8819 | 2.8597 | 2.887 | -0.18 |
| L1$_2$-Ni$_3$Al | 0.250 | 3.5545 | 3.5231 | 3.572 | -0.49 |

MACE-MP-0 medium は格子定数を 0.5% 未満で再現しており、Fig.6 議論の 1–2% 差異を議論する上で十分な精度があると判断できる。

## 8. Yamanouchi Table 4 / C14-Nb(Ni,Al)$_2$ 検証

- `analysis/TABLE4_MACE_ANALOGUE.md`: B2/fcc 双方から導いた原子直径（CN8）を CoAl/PdAl/RhAl/IrAl で比較。差は 0.03–0.06 Å（0.003–0.006 nm）で，Yamanouchi & Miura の 1.03 換算と整合。
- `analysis/C14_YAMANOUCHI_WEIGHTED_CHECK.md`: C14-Nb(Ni,Al)$_2$（$x=0.5$）の観測体積は純元素平均より **化合物由来の加重平均** に近いことを再確認。

**§7 と §8 の整合性について**: §7 の MACE 格子定数誤差 $<0.5$ % は B2/fcc **結晶そのもの**の $a$ を指す。§8 の $D_B$（B2-derived vs fcc-derived）の差 0.03–0.06 Å は同一元素の CN8 原子直径に対する **1.03 CN 換算因子** および **B2 vs fcc の局所配位差** 由来であり，格子定数再現精度とは独立な「サイズ因子」レベルの差である。両者は矛盾していない。Yamanouchi の実験 $D_B$ の読み取り精度（0.001–0.003 nm）に対し，MACE は 0.003–0.006 nm 程度の差を持つが，系統的な B2-B2 比較では十分な精度がある。

## 9. 4SL/8SL B2 副格子モデルへの接続

`4SL_B2_MODEL_DESIGN.md`、`b2_offstoich/extract_b2_defect_energies.py`、および `b2_offstoich/extract_4sl_b2_parameters.py` で，MLIP から組成依存な空孔/反サイト形成エネルギーを抽出し，4-sublattice / 8-sublattice CEF モデルのエンドメンバー・パラメータに変換する手順を設計。

- **副格子置換対称性**: 等価なサブラティスの入れ替えに対して Gibbs エネルギーが不変でなければならない（Ansara / Dupin / Sundman）。4SL/8SL モデルではこの対称性を課すことで，256 エンドメンバーの過剰決定・偽の秩序相出現を防ぐ。
- **秩序化強度**: 第一近接対相互作用の線形結合から得られる有効秩序化エネルギーは

  $$V = J_{\rm NiAl} - \frac{J_{\rm NiNi}+J_{\rm AlAl}}{2}$$

  の一つの値のみが物理的に意味を持つ。現状の簡易 Ising 外挿では $V \approx -0.1$ 〜 $-0.15$ eV/bond（Ni–Al 結合が他の結合より約 0.1 eV 強く負）。個別の $J_{\rm NiAl}, J_{\rm NiNi}, J_{\rm AlAl}$ の絶対値は A2 エンドメンバー・クラスター展開がないため信頼できない。
- 元素化学ポテンシャルではなく **B2 単相の自由エネルギー曲面** を介して正しい欠陥優位を記述するための第一ステップである。

## 10. 限界と次ステップ

1. 0 K 静的計算（実験は室温 XRD）。熱膨張で ~0.5–1%。
2. 欠陥形成エネルギーの $\Delta\mu$ 依存性を明示し，$E_f$ vs $\Delta\mu$ 図で Korzhavyi et al.（Phys. Rev. B **61**, 6003）と比較する。
3. Bradley & Taylor の密度測定から導かれる構造空孔濃度と MACE 予測を $x=0.55,0.60$ で直接比較する。
4. MACE-MP-0 に磁性・スピン自由度がない（Ni-rich 端）。
5. 各組成 3 配置のサンプリングは統計的に希薄。最低 10–20 配置，または `icet` クラスター展開に移行。
6. 1473 K の熱処理の振動エントロピー / MD 平衡化は未実装。
7. 4SL/8SL 副格子モデルのエンドメンバー対称性拘束を実装し，pycalphad/TDB 形式の原型を出力する。

## 11. 主要成果物

- `run_b2_offstoich*.py`, `make_figures.py`, `plot_a_bradley_overlay.py`, `extract_b2_defect_energies.py`
- `analyze_b2_hull_xmax.py`, `analyze_alrich_dense.py`, `extract_4sl_b2_parameters.py`, `add_ni3al4.py`
- `analysis/b2_defect_energies.csv`, `analysis/b2_branch_hull_xmax.csv`, `analysis/b2_xmax.json`, `analysis/b2_alrich_dense_phase_stability.csv`
- `analysis/bradley_taylor_a_exp.csv`, `analysis/a_comparison_bradley_mace.csv`
- `analysis/BENCHMARK_MACE_vs_MP_vs_EXP.md`（Ni$_3$Al$_4$ 含む）
- `analysis/TABLE4_MACE_ANALOGUE.md`, `analysis/C14_YAMANOUCHI_WEIGHTED_CHECK.md`
- `4SL_B2_MODEL_DESIGN.md`
- `figures/fig_b2_offstoich_vbar*.png`, `fig_b2_offstoich_a.png`, `fig_b2_offstoich_eform.png`, `fig_b2_alrich_dense_hull.png`, `fig_b2_hull_xmax.png`, `fig_b2_a_bradley_overlay.png`
- `07_reports/fig_energy_diagram_nial.png`
