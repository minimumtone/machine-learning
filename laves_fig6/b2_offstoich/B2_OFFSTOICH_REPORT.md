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

これは B2-NiAl の **三重欠陥（triple-defect）化学**（Ni 過剰=Ni 反サイト、Al 過剰=Ni 構造空孔）と整合する。

## 3. Fig. 6(a) B2 枝の再現

- 完全 B2: $\bar V=11.975$ Å³/atom（実験 ~12.0）, $a=2.882$ Å（実験 2.887）
- 半巨視正準 Boltzmann 混合（1473 K、解析的配置エントロピー込み）後のデジタイズ実験点との比較:

| 指標 | 値 |
|---|---|
| RMSE（体積） | 0.460 Å³/atom |
| MAPE | 2.8% |

現在の重み付けは **元素化学ポテンシャル** を使用しており、B2 単相としての化学ポテンシャルではないため、Al 過剰側で空孔枝の優位が過小評価される傾向がある。これは `make_figures.py` の混合曲線に反映される。より正確な取り扱いには B2 相の化学ポテンシャル、あるいは 4SL/8SL 副格子モデルが必要（下記）。

## 4. Al 過剰側の二相分離傾向（密サンプリング結果）

Al-rich 反サイト密サンプリング（$x_{\rm Al}=0.61$–0.79）を凸包と比較した結果（`b2_offstoich/analyze_alrich_dense.py`）：

| $x_{\rm Al}$ | $E_f$ (eV/atom) | 凸包上の $E_f$ | 凸包からの乖離 (eV/atom) |
|---|---|---|---|
| 0.61 | -0.514 | -0.618 | +0.103 |
| 0.63 | -0.486 | -0.590 | +0.104 |
| 0.65 | -0.465 | -0.571 | +0.106 |
| 0.67 | -0.441 | -0.544 | +0.103 |
| 0.69 | -0.430 | -0.525 | +0.095 |
| 0.71 | -0.397 | -0.497 | +0.100 |
| 0.73 | -0.377 | -0.479 | +0.102 |
| 0.75 | -0.350 | -0.451 | +0.101 |
| 0.77 | -0.330 | -0.409 | +0.079 |
| 0.79 | -0.300 | -0.381 | +0.080 |

Al 過剰側の B2 反サイト単相は凸包から **0.08–0.11 eV/atom** 高エネルギーであり、**$x_{\rm Al}\gtrsim0.6$ では B2-NiAl + NiAl$_3$ の二相分離が熱力学的に有利**となる。このため、空孔を考慮しない純粋反サイトモデルでは B2 単相を維持できない。実際には Ni 構造空孔（三重欠陥）が導入されることで、B2 相はより広い Al 過剰領域まで準安定に存在する。

## 5. 格子定数の異常

Al 過剰側では $\bar V$ が急増する一方、**実効格子定数 $a$ はほぼ一定〜微減**（2.88 Å → 2.84–2.82 Å）。これは構造空孔導入による密度欠損の特徴。`fig_b2_offstoich_a.png` 参照。

## 6. 凸包・中間化合物

`07_reports/fig_energy_diagram_nial_generator.py` により、純元素に加え L1$_2$-Ni$_3$Al、Ni$_5$Al$_3$、Ni$_2$Al$_3$、NiAl$_3$ の MACE-MP-0 緩和エネルギーを含めた **真の凸包** を作成した。これにより単純な Ni–B2–Al 三点タイラインではなく、熱力学的に正しい下側包絡線を描く。

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

- `analysis/TABLE4_MACE_ANALOGUE.md`: B2/fcc 双方から導いた原子直径（CN8）を CoAl/PdAl/RhAl/IrAl で比較。差は 0.03–0.06 Å（0.003–0.006 nm）で、Yamanouchi & Miura の 1.03 換算と整合。
- `analysis/C14_YAMANOUCHI_WEIGHTED_CHECK.md`: C14-Nb(Ni,Al)$_2$（$x=0.5$）の観測体積は純元素平均より **化合物由来の加重平均** に近いことを再確認。

## 9. 4SL/8SL B2 副格子モデルへの接続

`4SL_B2_MODEL_DESIGN.md`、`b2_offstoich/extract_b2_defect_energies.py`、および `b2_offstoich/extract_4sl_b2_parameters.py` で、MLIP から組成依存な空孔/反サイト形成エネルギーを抽出し、4-sublattice / 8-sublattice CEF モデルのエンドメンバー・パラメータに変換する手順を設計。

定数対相互作用の簡易推定値（`analysis/b2_pair_interactions.json`）：

| 相互作用 | 値 (eV/結合) |
|---|---|
| $J_{\rm NiAl}$ | -1.356 |
| $J_{\rm NiNi}$ | -1.257 |
| $J_{\rm AlAl}$ | -1.165 |

これは元素化学ポテンシャルではなく **B2 単相の自由エネルギー曲面** を介して正しい欠陥優位を記述するための第一ステップである。

## 10. 限界と次ステップ

1. 0 K 静的計算（実験は室温 XRD）。熱膨張で ~0.5–1%。
2. 半巨視正準重みに用いた元素化学ポテンシャルは B2 単相には近似。4SL/8SL パラメータ化で改善。
3. MACE-MP-0 に磁性・スピン自由度がない（Ni-rich 端）。
4. 各組成 3 配置のサンプリングは統計的に希薄。最低 10–20 配置、または `icet` クラスター展開に移行。
5. 1473 K の熱処理の振動エントロピー / MD 平衡化は未実装。

## 11. 主要成果物

- `run_b2_offstoich*.py`, `make_figures.py`, `extract_b2_defect_energies.py`
- `analyze_alrich_dense.py`, `extract_4sl_b2_parameters.py`
- `analysis/b2_defect_energies.csv`, `analysis/b2_pair_interactions.json`, `analysis/b2_alrich_dense_phase_stability.csv`
- `analysis/BENCHMARK_MACE_vs_MP_vs_EXP.md`
- `analysis/TABLE4_MACE_ANALOGUE.md`, `analysis/C14_YAMANOUCHI_WEIGHTED_CHECK.md`
- `4SL_B2_MODEL_DESIGN.md`
- `figures/fig_b2_offstoich_vbar*.png`, `fig_b2_offstoich_a.png`, `fig_b2_offstoich_eform.png`, `fig_b2_alrich_dense_hull.png`
- `07_reports/fig_energy_diagram_nial.png`
