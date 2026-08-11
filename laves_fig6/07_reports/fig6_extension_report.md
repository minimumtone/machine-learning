# Fig. 6(a)(b) 再現・拡張レポート（MLIP確率論的アプローチ第2弾）

対象: T. Yamanouchi and S. Miura, *Mater. Trans.* **59**, 546–555 (2018), Fig. 6。
計算: MACE-MP-0 medium (float64, CPU)、`LBFGS(FrechetCellFilter)`、fmax = 0.02 eV/Å。
Boltzmann混合の温度は論文の熱処理条件 **1473 K**（168 h → 水焼入れ）に更新。

## 1. C14-Nb(Ni$_{1-x}$Al$_x$)$_2$（Fig. 6(a) Laves枝＝黒三角）の確率論的掃引

- 2×2×1 supercell（48原子、B副格子32サイト）にNi/Alを乱数占有（×3配置）、
  $x = 1/16 \ldots 15/16$ の15組成を掃引（`laves_fig6/c14_prob/`）。
- Fig. 6(a) の黒三角（$\bar V_{\mathrm{Nb(Ni,Al)_2}}$実験点、原図ラスタから6点デジタイズ）と比較:
  **RMSE 0.073 Å$^3$/atom、MAPE 0.40%**。全域で単調・ほぼ線形に再現。
- 配置ばらつきは平均 0.011 Å$^3$/atom と小さく、Laves相の$\bar V$は原子配置に鈍感。

## 2. B2不定比枝の拡張（Fig. 6(a) 白丸）

- 空孔枝を広域化: Ni過剰側 $x_{\mathrm{Al}}$ = 0.30–0.49（Al空孔）、
  Al過剰側 0.52–0.66（Ni空孔）、乱数配置×3。
- Boltzmann混合（1473 K）でのデジタイズ実験12点との一致: RMSE 0.157 Å$^3$/atom、
  MAPE 0.98%（1273 K時と同値。重みが両温度でほぼ飽和しているため）。
- 注意: Ni過剰側の空孔濃度が大きい領域（$x<0.36$、$c_{\mathrm{vac}}>20$%）では
  緩和が不収束の配置が増え（局所的な構造崩壊）、この枝は実際には反サイト枝より
  高エネルギーで熱力学的に選択されない。

### 空孔導入量の別表現（空孔濃度）

`fig_vacancy_representation.png`: 空孔濃度 $c_{\mathrm{vac}} = 1 - N_{\mathrm{atom}}/N_{\mathrm{site}}$
を横軸にした整理。

- Al過剰側（Ni空孔）: $\bar V$は $c_{\mathrm{vac}}$ にほぼ線形に増加
  （空孔1%あたり +0.10 Å$^3$/atom）し、bcc-Al極限（16.75 Å$^3$/atom）方向へ向かう。
  一方、実効格子定数 $a$ はほぼ一定（2.87→2.82 Å と微減）で、
  「体積はAl的に増えるが骨格の格子定数はB2のまま」という空孔支配の特徴を示す。
- Ni過剰側（Al空孔）: $a$も$\bar V$も減少し、bcc-Alから遠ざかる（Ni富化のため）。

### 配位数8（bcc）極限との関係

MLIPのbcc-Al参照: $a$ = 3.223 Å、$V$ = 16.75 Å$^3$/atom（fcc-Alの16.74とほぼ等体積）。
Al過剰B2のNi空孔枝は$\bar V$がbcc-Al体積へ単調接近するが、$x_{\mathrm{Al}}=0.66$
（$c_{\mathrm{vac}}=24$%）でも14.8 Å$^3$/atomであり、bcc-Al極限への到達には
B2骨格の崩壊（配位数8の純Al bccへの連続変形）が必要。実験的にもB2相はこの前に
不安定化するため、「bcc-Al格子定数への漸近」は外挿傾向としてのみ確認できる。

## 3. fcc Ni(Al)固溶体の全組成掃引とVegard則

`fig_niall_fcc_vegard.png`（`laves_fig6/niall_ext/`）: 32原子fccセル、乱数固溶体×3配置、
$x_{\mathrm{Al}}$ = 0–1.0。

- **Vegard則（fcc-Ni→fcc-Al）に対して大きな負の偏差**: 最大 −1.3 Å$^3$/atom（$x\approx0.5$）。
- $x\lesssim0.25$（実験のNi(Al)固溶体域）では偏差は −0.9 Å$^3$/atom 以下だが既に有意で、
  Fig. 6(a)実験の下向き湾曲と整合。
- $x>0.6$ でVegard線に急速に復帰（Al富化でNi–Al強結合対が希釈されるため）。
  つまりNi(Al)固溶体はVegard則には従わず、Ni–Al間の強い化学結合（B2的短距離秩序）
  による体積収縮が支配的。これは「純元素体積ではなく化合物由来体積を使うべき」
  という論文の主張のMLIP側からの直接的な裏付け。

## 4. B2組成（x=0.5）の秩序度依存

`fig_b2_order_param.png`: 長距離秩序パラメータ $\eta$（完全B2で1、ランダムbccで0）を
Ni↔Al対交換で変化、×3配置。

| $\eta$ | $a$ (Å) | $\bar V$ (Å$^3$/atom) | $\Delta E$ (eV/atom) |
|---|---|---|---|
| 1.00 (完全B2) | 2.8825 | 11.975 | 0 |
| 0.75 | 2.9045 | 12.251 | +0.102 |
| 0.50 | 2.9248 | 12.511 | +0.153 |
| 0.25 | 2.9446 | 12.766 | +0.178 |
| 0.00 (ランダムbcc) | 2.9439 | 12.757 | +0.176 |

- 完全秩序→完全無秩序で $a$ は **+0.06 Å（+2.1%）**、$\bar V$ は **+0.78 Å$^3$（+6.5%）** 増加。
- 変化は $\eta\gtrsim0.5$ でほぼ線形、$\eta<0.25$ で飽和（短距離秩序が残るため）。
- 実験の格子定数ノイズ床（±0.016 Å）に対して秩序度の影響は数倍大きく、
  **不定比組成の$\bar V$解析では秩序度（熱処理条件）の管理が不可欠**。
  逆に、1473 K焼鈍・水焼入れ材ではB2秩序はほぼ完全（$T_c \gg 1473$ K）なので、
  Fig. 6(a)の解析では$\eta\approx1$の仮定は妥当。

## 生成物

- `laves_fig6/c14_prob/`: `run_c14_prob.py`, `make_figures_c14.py`,
  `analysis/c14_prob_volumes.csv`, `analysis/fig6a_digitized_triangles.csv`,
  `analysis/c14_prob_comparison.json`, `figures/fig_c14_prob_vbar.png`
- `laves_fig6/b2_offstoich/`: `run_b2_offstoich_extra_vac.py`, `run_b2_offstoich_wide_vac.py`,
  `analysis/b2_offstoich_volumes_extra_vac.csv`, `analysis/b2_offstoich_volumes_wide_vac.csv`,
  `analysis/b2_offstoich_sizecheck.csv`（4×4×4 vs 5×5×5: $\Delta V$ ≤ 0.06 Å$^3$/atom ≈ 0.5%）
- `laves_fig6/niall_ext/`: `run_niall_ext.py`, `make_figures_ext.py`,
  `analysis/niall_fcc_ext.csv`, `analysis/b2_order_param.csv`,
  `analysis/niall_ext_summary.json`, `figures/fig_niall_fcc_vegard.png`,
  `figures/fig_vacancy_representation.png`, `figures/fig_b2_order_param.png`
