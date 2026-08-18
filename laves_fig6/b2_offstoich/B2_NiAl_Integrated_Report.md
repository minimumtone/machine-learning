# B2-NiAl 不定比相解析と 4SL/8SL 副格子モデル接続：統合レポート

この文書は、B2-NiAl の MLIP 再現検証（Part I）と、点欠陥エネルギーから 4SL/8SL CALPHAD 副格子モデルへの接続検討（Part II）を統合したものです。

---

## Part I: B2-NiAl 不定比・凸包・欠陥化学 検証

### 1. 計算条件

- **MLIP**: MACE-MP-0 medium, float64, CPU
- **緩和**: `LBFGS(FrechetCellFilter)`, $f_{\max}=0.02$ eV/Å, max 500 ステップ
- **超胞**: B2 4$\times$4$\times$4 慣用超胞（128 サイト = 64 B2 単位胞）
- **欠陥様式**:
  - Ni 過剰側 ($x_{\mathrm{Al}}<0.5$): Ni が Al サイトに入る **反サイト**、または Al サイトに **空孔**
  - Al 過剰側 ($x_{\mathrm{Al}}>0.5$): Ni サイトに **空孔**、または Ni サイトに Al が入る **反サイト**
- **サンプリング**: 各組成・各欠陥モデルで 3 シード（$n=3$ は統計的に希薄）
- **元素化学ポテンシャル（MACE fcc 基準）**: $\mu_{\rm Ni}=-5.7319$ eV/atom, $\mu_{\rm Al}=-3.7284$ eV/atom
- **自由エネルギー**: 占有原子あたり Helmholtz 自由エネルギー

$$G_i = \frac{E_i - \mu_{\rm Ni}N_{\rm Ni} - \mu_{\rm Al}N_{\rm Al}}{N_{\rm atom}} - k_{\rm B}T\frac{\ln g_i}{N_{\rm atom}},\qquad g_i=C(64,n_{\rm defect})$$

固定組成での欠陥モデル比較には、占有原子あたり Helmholtz 自由エネルギー $G_i$ を用いる。全系の半巨視正準ポテンシャルは $\Omega_i=N_{\rm atom}G_i$ であるが、$N_{\rm atom}$ が欠陥モデルによって異なる場合、$G_i$ と $\Omega_i$ の大小関係は保存されない。固定組成の準安定 B2 単相では、同じ $x_{\rm Al}$ を維持した欠陥モデル比較の自然な変数は $G_i$ である。盲目的に $\Omega_i$ を使うと $x_{\rm Al}\approx0.60$ で Al-rich 反サイトモデルが空孔モデルより低いと予測し、実験的な Ni 空孔支配方と既存の 1273 K Boltzmann 重み付けに矛盾する。したがって $G_i$ を用いる。

### 2. 点欠陥形成エネルギー

完全 B2 から欠陥 1 個あたりのエネルギー（`analysis/b2_defect_energies.csv`）：

| 欠陥種 | 平均 $\Delta E$ (eV/欠陥) | 標準偏差 | 支配性 |
|---|---|---|---|
| Ni 反サイト on Al サイト（Ni 過剰） | 0.79 | 0.20 | Ni 過剰側で最低 |
| Al 空孔（Ni 過剰） | 1.78 | 0.15 | 常に Ni 反サイトより高い |
| Ni 空孔（Al 過剰） | 1.17 | 0.08 | Al 過剰側で最低 |
| Al 反サイト on Ni サイト（Al 過剰） | 1.59 | 0.10 | 常に Ni 空孔より高い |

B2 単相で $\mu_{\rm Ni}+\mu_{\rm Al}=-10.844$ eV/formula の拘束下、$\Delta\mu=\mu_{\rm Al}-\mu_{\rm Ni}$ には約 **2.768 eV** の自由度がある（$\mu_{\rm Ni}$ 可動域 1.384 eV、$d(\Delta\mu)=-2\,d\mu_{\rm Ni}$）。

### 3. 実験だけによる構造空孔の証明

Taylor & Doyle (1972) は格子定数と密度の両方を測定している（NASA TM-105598 に線形回帰が再録されている）。したがって、**密度由来の平均原子体積** $\bar V$ は $a(x)$ と独立な量になりうる。ただし、Ellner et al. (1991) Table 4 の NiAl 結晶データの出典に Taylor & Doyle が含まれているため、現状では「$a$ と $\bar V$ が独立であることを前提とする」と明記しておく。もし Ellner の $\bar V$ が $a(x)$ に構造空孔モデルを仮定して導出されたものなら、以下の表はその仮定を復元するだけになる。

B2 1 単位胞あたりの体積は $a^3$、それを占める原子数が $N_{\rm atom}$ なら、平均原子体積 $\bar V = a^3/N_{\rm atom}$ より

$$c_{\rm vac} = 1 - \frac{a^3}{2\bar V}$$

が成り立つ（B2 の 1 胞 = 2 サイト）。反サイトだけの場合は $c_{\rm vac}=0$、構造空孔モデルでは $c_{\rm vac}=(2-1/x_{\rm Al})/2$。

Taylor-Doyle の Al-rich 傾きは NASA TM-105598 が示す **45–50 at.% Ni（$x_{\rm Al}=0.55\to0.50$）** の線形回帰を用いる。これまで使われていた 34 at.% Ni（$x_{\rm Al}=0.66$）端点は誤りで、Al-rich 傾きを著しく過小評価していた。

`plot_vacancy_concentration.py` では、Hough 検出で読んだ Ellner Fig. 5 を、Ellner Table 2 の低 Al 領域数表

| $x_{\rm Al}$ | 0.105 | 0.147 | 0.240 |
|---|---|---|---|
| $V^A$ (Å³) | 10.94 | 11.19 | 11.36 |

と突き合わせて絶対校正した。$x=0.105$ の 1 点は 2 相/fcc 領域の外れ値と判断し、$x=0.147$ と $0.240$ の中央値を用いて系統オフセット $+0.0338$ Å³ を補正した。（校正前の raw 値との比較も表に示す。）

`analysis/vacancy_concentration_exp_vs_mace.csv`（図：`figures/fig_b2_vacancy_concentration.png`）：

|| $x_{\rm Al}$ | $a_{\rm T\&D}$ (Å) | $\bar V_{\rm raw}$ (Å³/atom) | $\bar V_{\rm cal}$ (Å³/atom) | $c_{\rm vac}^{\rm exp\,raw}$ | $c_{\rm vac}^{\rm exp\,cal}$ | $c_{\rm vac}^{\rm model}$ | $c_{\rm vac}^{\rm MLIP}$ | $c_{\rm cal}/c_{\rm model}$ | $p_{\rm Al\,antisite}$ |
|---|---|---|---|---|---|---|---|---|---|---|
|| 0.45 | 2.879 | 11.871 | 11.905 | $-0.005$ | 0.000 | 0.000 | 0.000 | — | 0.0000 |
|| 0.48 | 2.884 | 11.961 | 11.995 | $-0.003$ | 0.000 | 0.000 | 0.000 | — | 0.0000 |
|| 0.50 | 2.887 | 12.095 | 12.129 | 0.005 | 0.008 | 0.000 | 0.000 | — | 0.0000 |
|| 0.52 | 2.878 | 12.417 | 12.451 | 0.040 | 0.042 | 0.038 | 0.038 | 1.103 | 0.0000 |
|| 0.55 | 2.865 | 13.044 | 13.078 | 0.098 | 0.101 | 0.091 | 0.091 | 1.108 | 0.0000 |
|| 0.58 | 2.852 | 13.458 | 13.491 | 0.138 | 0.140 | 0.138 | 0.138 | 1.016 | 0.0000 |
|| 0.60 | 2.843 | 13.776 | 13.810 | 0.166 | 0.168 | 0.167 | 0.167 | 1.006 | 0.0000 |

校正後の $c_{\rm vac}^{\rm exp\,cal}$ は構成空孔モデルと一致（$c_{\rm cal}/c_{\rm model}\approx1.0$–1.1）しており、系統的な不足は見られない。4 サブラティス分解 $p_{\rm Al\,antisite}=\max(0, x_{\rm Al}\,a^3/\bar V - 1)$ も 0 に留まる。Al 反サイト 1.59 eV と Ni 空孔 1.17 eV の差（0.42 eV）から、1473 K で $p\sim3$ % が熱的に可能なオーダーではあるが、現在の実験分解能・校正精度では分離できない。したがって §3 の主張は、**「実験2系統の組み合わせだけで B2-NiAl の支配的構成欠陥は構造空孔である」** ことだけで、熱的反サイト量の定量化は MACE の二重検証には至らなかった。

反サイトモデルは $c_{\rm vac}=0$、予測 $\bar V=a^3/2=11.85$ Å³/atom は実験の校正値 13.81 と 16 % 離れて棄却される。これは MLIP を使わない実験2系統の直接証明である。

### 4. Fig. 6(a) B2 欠陥モデルの再現（拘束 B2 単相）

**この節は B2 相を強制した拘束計算（実験も焼入れ・準安定試料）に対応する。§5 では無拘束の相安定性を扱う。**

- 完全 B2: $\bar V=11.975$ Å³/atom, $a=2.882$ Å
- 比較範囲: $0.45\le x_{\rm Al}\le0.60$

| 指標 | 値 |
|---|---|
| 比較点数 | 12 |
| RMSE（全体） | 0.158 Å³/atom |
| MAPE（全体） | 0.99 % |
| RMSE（Ni-rich, $x<0.50$） | 0.086 Å³/atom (3点) |
| MAPE（Ni-rich） | 0.63 % |
| RMSE（Al-rich, $x>0.50$） | 0.175 Å³/atom (9点) |
| MAPE（Al-rich） | 1.11 % |

残留誤差 0.158 Å³/atom は、$\bar V\approx12$ Å³/atom に対して約 1.3 %。§8 の格子定数誤差（体積で 0.5–1.5 %）と同オーダーであり、**既に MACE-MP-0 の体積精度の床に達している**。

### 5. B2 単相域の上限 $x_{\max}$（0 K・1273 K 凸包）

**この節は無拘束の相安定性を扱う。§4 とは問いが異なる。**

`analyze_b2_hull_xmax.py` で作成する凸包：
- **0 K 全相凸包**: 純元素、fcc-SQS、L1$_2$-Ni$_3$Al、Ni$_3$Al$_4$、Ni$_5$Al$_3$、Ni$_2$Al$_3$、NiAl$_3$
- **1273 K 固相凸包**: Ni、Ni$_3$Al、B2、Ni$_2$Al$_3$（Ni$_3$Al$_4$、Ni$_5$Al$_3$、NiAl$_3$ は 1200 $^\circ$C では非安定）
- **1473 K 比較**: この温度では NiAl の Al 側境界は Ni$_2$Al$_3$ ではなく **液相** で決まるため、固相同士の凸包比較としては不成立

| 凸包・欠陥モデル | 3 meV | 5 meV | 10 meV | 20 meV |
|---|---|---|---|---|
| 0 K 全相 / 最安定 B2 欠陥モデル (Ef) | 0.512 | 0.520 | 0.529 | 0.561 |
| 1273 K / 最安定 B2 欠陥モデル (G) | $\ge0.660$ (saturated) | $\ge0.660$ | $\ge0.660$ | $\ge0.660$ |
| 実験（1273 K 固相限界、Okamoto Ni-Al assessment） | ~0.575 | ~0.575 | ~0.575 | ~0.575 |

**解釈**: 0 K では $x_{\max}\approx0.52$ で止まり、実験 ~0.60 を再現しない。1273 K では、B2 空孔モデルに理想配置エントロピーを加えた $G$ が $x\approx0.66$ まで凸包上または下に乗り続け、**実験の ~0.575 を大幅に超過**して下限値 $\ge0.66$ となる。これは

1. MACE が Al-rich 側の格子定数・体積傾きを実験と整合的に再現する（§6）一方、B2 空孔モデルと競合相 Ni$_2$Al$_3$ との相対エネルギー、または理想配置エントロピー近似から生じる超過であり、単純な「空孔周りの局所緩和の過大評価」だけでは説明できない。
2. 競合相 Ni$_2$Al$_3$ の欠陥エントロピー、熱膨張、振動エントロピーを無視しており、1473 K 境界は液相なので固相凸包比較がそもそも不十分。

という近似の帰結である。**実験の 0.575 を「挟む」ことは主張せず、「0 K は 0.055 過小、有限温度補正は 0.085 以上過大」として整理する**。

### 6. 格子定数 $a(x)$ の傾き比較

Taylor & Doyle (1972) 一次データの Al-rich 端点を 34 at.% Ni から正しい 45 at.% Niに修正すると、MACE の Al-rich 傾きは実験と整合する。点ごとの絶対値も $a$ は $\sim0.006$ Å、$\bar V$ は 1–2 % 以内で一致する。ただし Ni-rich 側では MACE がやや急な傾き（+26 %）を示す。

| 領域 | $da/dx$ MACE (Å/$x_{\rm Al}$) | $da/dx$ T&D (Å/$x_{\rm Al}$) | MACE/T&D | $d\bar V/dx$ MACE | $d\bar V/dx$ T&D | MACE/T&D |
|---|---|---|---|---|---|---|
| Ni-rich (0.42–0.50) | +0.199 | +0.158 | 1.27 | +2.47 | +1.96 | 1.26 |
| Al-rich (0.50–0.60) | $-0.410$ | $-0.436$ | 0.94 | +17.91 | +17.62 | 1.02 |
| Al-rich (0.50–0.66) | $-0.391$ | $-0.436$ | 0.90 | +17.69 | +17.04 | 1.04 |

点ごと比較（`analysis/a_comparison_taylor_doyle_mace.csv`）：

| $x_{\rm Al}$ | $a_{\rm T\&D}$ (Å) | $a_{\rm MACE}$ (Å) | $\Delta a$ (Å) |
|---|---|---|---|
| 0.45 | 2.879 | 2.873 | $-0.006$ |
| 0.50 | 2.887 | 2.882 | $-0.005$ |
| 0.55 | 2.865 | 2.861 | $-0.004$ |
| 0.60 | 2.843 | 2.842 | $-0.002$ |
| 0.62 | 2.835 | 2.834 | $-0.001$ |

したがって、Al-rich 側の $x_{\max}$ 超過は「空孔周りの局所緩和を MACE が 3 倍過大評価」では説明できない。1273 K で B2 空孔モデルが $x\ge0.66$ まで凸包に留まる残差は、Ni$_2$Al$_3$ との相対的能量誤差、理想配置エントロピーの近似、振動・熱膨張無視などに帰する。

**セル固定緩和テストの結果**（`fixed_cell_relaxation_test.py` + `analysis/fixed_cell_relaxation_test.csv`）：$x_{\rm Al}=0.58,0.60$ の B2 超胞について、T&D 格子定数にセルを固定して内部座標のみ緩和した場合、空孔モデルのエネルギーは自由緩和とほぼ同じ（差 $\sim10^{-5}$ eV/atom）で、反サイトモデルは固定セルで大きく不安定化する。すなわち、セル形状を実験値に押し込めても空孔モデルが選ばれ、$x_{\max}$ は下がらない。この結果は $x_{\max}$ 超過の主因が空孔周りの局所緩和ではないことを直接的に示す。

### 7. 凸包・中間化合物

0 K 真の凸包には L1$_2$-Ni$_3$Al、Ni$_3$Al$_4$、Ni$_5$Al$_3$、Ni$_2$Al$_3$、NiAl$_3$ を含めた。Ni$_3$Al$_4$ ($x_{\rm Al}=0.571$, $E_f=-0.649$ eV/atom) は 0 K の B2 均一域上限近傍を強く安定化するが、700 $^\circ$C 以下の低温相である。1273 K 以上では Ni$_3$Al$_4$、Ni$_5$Al$_3$、NiAl$_3$ は非安定。

### 8. MLIP誤差評価（MACE vs MP-PBE vs 実験）

`analysis/BENCHMARK_MACE_vs_MP_vs_EXP.md`（セル定数は spglib 規格化後、3 辺を昇順に並べたもの；軸ラベルは結晶学的セッティングに依存しない）：

| 構造 | $x_{\rm Al}$ | MACE $a$ (Å) | MACE $b$ (Å) | MACE $c$ (Å) | MP-PBE $a$ | MP-PBE $b$ | MP-PBE $c$ | 実験 $a$ | 実験 $b$ | 実験 $c$ | MACE 誤差 (%) | $\|err\|$ 平均 (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ni | 0.000 | 3.51 | 3.51 | 3.51 | 3.475 | 3.475 | 3.475 | 3.524 | 3.524 | 3.524 | $-0.40/-0.40/-0.40$ | 0.40 |
| Al | 1.000 | 4.06 | 4.06 | 4.06 | 4.039 | 4.039 | 4.039 | 4.05 | 4.05 | 4.05 | $+0.25/+0.25/+0.25$ | 0.25 |
| B2-NiAl | 0.500 | 2.882 | 2.882 | 2.882 | 2.86 | 2.86 | 2.86 | 2.887 | 2.887 | 2.887 | $-0.18/-0.18/-0.18$ | 0.18 |
| L1$_2$-Ni$_3$Al | 0.250 | 3.554 | 3.555 | 3.555 | 3.523 | 3.523 | 3.523 | 3.572 | 3.572 | 3.572 | $-0.49/-0.49/-0.49$ | 0.49 |
| Ni$_5$Al$_3$ | 0.375 | 3.857 | 6.285 | 7.664 | 3.725 | 6.556 | 7.401 | 3.732 | 6.727 | 7.475 | $+3.35/-6.57/+2.53$ | 4.15 |
| Ni$_2$Al$_3$ | 0.600 | 4.038 | 4.038 | 4.886 | 3.994 | 3.994 | 4.881 | 4.036 | 4.036 | 4.888 | $+0.06/+0.06/-0.05$ | 0.06 |
| NiAl$_3$ | 0.750 | 4.776 | 6.646 | 7.432 | 4.771 | 6.559 | 7.304 | 4.811 | 6.613 | 7.367 | $-0.73/+0.50/+0.88$ | 0.71 |
| Ni$_3$Al$_4$ | 0.571 | 11.40 | 11.40 | 11.40 | 11.312 | 11.312 | 11.312 | 11.408 | 11.408 | 11.408 | $-0.07/-0.07/-0.07$ | 0.07 |

Ni$_5$Al$_3$ の実験値は Khadkikar & Vedula (Pt$_5$Ga$_3$-type orthorhombic) の $a=7.475$ Å, $b=3.732$ Å, $c=6.727$ Åを昇順に並べたもの。MACE は平均で 4 % 程度高い。Ni$_3$Al$_4$ は実験 $a=11.408$ Åとほぼ一致（$-$0.07 %）。Ni$_5$Al$_3$ の誤差は §5 の $x_{\max}$ 上限を押し下げる方向に作用する。

### 9. 4SL/8SL B2 副格子モデルへの接続（概要）

`4SL_B2_MODEL_DESIGN.md` に詳述。要点は：
- 2SL/4SL/8SL のエンドメンバー変換手順
- 副格子置換対称性の必要性
- **秩序化強度は $V=-\Delta E_{\rm order}/4$ として直接決定**（$\Delta E_{\rm order}=0.3509$ eV/formula より $V=-0.088$ eV/bond）
- 定数対近似で孤立点欠陥エネルギーから算定すると $V\approx-0.145$ eV/bond となり、前者に対して 65 % 過大 → 定数対近似の破綻の証拠。これを `icet` クラスター展開で独立に確認した（`run_icet_b2_cluster_expansion.py`）：
  - 第一近接対のみ（cutoff 2.566 Å）：$V_{\rm 1NN}\approx-0.135$ eV/bond、$J_{\rm NiAl}=-1.354$, $J_{\rm NiNi}=-1.500$, $J_{\rm AlAl}=-0.935$ eV
  - 第一＋第二近接対（2.952 Å）：$V\approx-0.102$ eV/bond、RMSE=0.020 eV/atom
  - 第一＋第二近接対＋三点群：$V\approx-0.100$ eV/bond、RMSE=0.009 eV/atom
  - クラスターを充実させると熱力学的 $V=-0.088$ eV/bond に漸近
- **A2-Ni / A2-Al（bcc）を MACE-MP-0 で緩和**: $a_{\rm Ni}=2.790$ Å, $E=-5.662$ eV/atom ($-11.324$ eV/formula); $a_{\rm Al}=3.225$ Å, $E=-3.687$ eV/atom ($-7.374$ eV/formula)

### 10. 限界と次ステップ

1. **$n=3$ 配置**: $x_{\max}$ や $E_f$ の統計誤差を評価するには 10–20 配置が必要
2. **1473 K 境界**: 液相を含む CALPHAD 比較、または MD/phonon による固相エントロピー補正
3. **A2-Ni / A2-Al**: 4SL/8SL 純 bcc 端成分を `run_a2_endmembers.py` で緩和済み
4. **磁性**: MACE-MP-0 に Ni のスピン分極がない
5. **icet クラスター展開**: `run_icet_b2_cluster_expansion.py` で第一近接対相互作用を抽出済み
6. **$E_f$ vs $\Delta\mu$**: Korzhavyi et al. (Phys. Rev. B 61, 6003) との比較

---

## Part II: MLIP 空孔・アンチサイトエネルギーから 4SL/8SL B2 副格子モデルへの接続

### 1. データ源

Part I の `b2_defect_energies.csv` を使用。濃度依存で平均を取った値は §2 と同じ。

### 2. 2 サブラティスモデル（出発点）

```
(Ni, Al, Va)_{0.5} (Al, Ni, Va)_{0.5}
```

エンドメンバーは完全 B2、反 B2、Ni 過剰極限、Al 過剰極限、Ni 欠損、Al 欠損など。A2-Ni($a=2.790$ Å, $E=-5.662$ eV/atom) と A2-Al($a=3.225$ Å, $E=-3.687$ eV/atom) は MACE-MP-0 で緩和済み。$x=0.5$ のランダム A2（Ni$_0.5$Al$_0.5$ on bcc）エネルギー $E_{\rm A2}=-10.493$ eV/formula は `b2_order_param.csv` の$\eta=0$外挿より、B2 との秩序化エネルギー $\Delta E_{\rm order}=+0.3510$ eV/formula（$E_{\rm A2}-E_{\rm B2}$）を与える。

### 3. 4SL/8SL 一般化

```
(Ni,Al,Va)_{1/4}(Ni,Al,Va)_{1/4}(Ni,Al,Va)_{1/4}(Ni,Al,Va)_{1/4}
```

- $\alpha_1, \alpha_2$：元の Ni 副格子を 2 分割
- $\beta_1, \beta_2$：元の Al 副格子を 2 分割

**副格子置換対称性**: 等価サブラティスの置換に対して Gibbs エネルギーが不変でなければならない（Ansara/Dupin/Sundman）。これを課さないとエンドメンバー過剰決定や偽秩序相が生じる。

### 4. 秩序化強度の一本化

B2 構造の最近接対モデルでは、慣用胞（2 原子）あたり 8 本の結合を持つ。完全 B2 は全結合が Ni–Al、完全ランダム A2 は Ni–Ni:Al–Al:Ni–Al = 2:2:4。よって

$$\Delta E_{\rm order} = E_{\rm A2} - E_{\rm B2} = 2(J_{\rm NiNi}+J_{\rm AlAl}) - 4J_{\rm NiAl}$$

定数対モデルでは $\Delta E_{\rm order}=-4V$ となるため、直接測定できる秩序化エネルギーから

$$V = -\frac{\Delta E_{\rm order}}{4} = -\frac{0.3509}{4} = -0.088\ {\rm eV/bond}$$

が得られる。

`icet` クラスター展開（`run_icet_b2_cluster_expansion.py`）で同じことを独立に確認できる。B2 系のみから第一近接対モデルをフィットすると

- $J_{\rm NiAl}=-1.354$ eV、$J_{\rm NiNi}=-1.500$ eV、$J_{\rm AlAl}=-0.935$ eV
- $V_{\rm pair,1NN}=J_{\rm NiAl}-(J_{\rm NiNi}+J_{\rm AlAl})/2=-0.137$ eV/bond

と、孤立点欠陥からの定数対推定 $V\approx-0.145$ eV/bond とほぼ一致する。しかし、第二近接対（同じサブラティス上の Ni–Ni / Al–Al 対）を加えると $V\approx-0.102$ eV/bond、三点群まで含めると $V\approx-0.100$ eV/bond と熱力学的値 $-0.088$ eV/bond に急速に近づく。したがって、**定数対近似の破綻は第一近接対だけでなく、第二近接対・多点項の寄与を無視したことに起因する**。個別の $J$ 値の絶対値には固定体積近似の大きな依存があるため、報告すべき秩序化強度は依然として $V=-0.088$ eV/bond である。

`analysis/b2_pair_interactions.json` と `analysis/icet_b2_cluster_expansion_summary.json`:
- `delta_order_obs_eV`: 0.3509
- `V_from_ordering_eV`: -0.0877
- `V_pair_constant_eV`: -0.1449
- `V_definition`: 上記の整理を含む注記

`analysis/icet_b2_cluster_expansion_summary.json`:
- `V_eff_eV_per_bond`（1NN+2NN+triplets）: -0.1002
- `V_pair_eV_per_bond`（1NN only）: -0.1369
- `J_NiAl_eV`: -1.3544、$J_{\rm NiNi}$: -1.5004、$J_{\rm AlAl}$: -0.9346
- `rmse_eV_per_atom`（1NN+2NN+triplets）: 0.0092

### 5. 未解決点

- **A2-Ni / A2-Al 済み**: 4SL/8SL 用純 bcc 端成分を MACE-MP-0 で計算（`analysis/a2_endmember_energies.csv`）
- **$n=3$**: 配置サンプリングが統計的に希薄
- **振動エントロピー / 液相**: 有限温度凸包の信頼性向上
- **icet クラスター展開**: `run_icet_b2_cluster_expansion.py` により第一近接対相互作用を抽出済み。8SL パラメータへの変換は次の段階

### 6. 成果物

- `b2_offstoich/B2_OFFSTOICH_REPORT.md`
- `b2_offstoich/4SL_B2_MODEL_DESIGN.md`
- `b2_offstoich/B2_NiAl_Integrated_Report.md`
- `b2_offstoich/analysis/b2_pair_interactions.json`
- `b2_offstoich/analysis/b2_branch_finiteT_hull.csv`
- `b2_offstoich/analysis/vacancy_concentration_exp_vs_mace.csv`
- `b2_offstoich/analysis/taylor_doyle_mace_slopes.csv`
- `b2_offstoich/figures/fig_b2_vacancy_concentration.png`
- `b2_offstoich/figures/fig_b2_hull_deviation.png`
- `b2_offstoich/figures/fig_b2_a_taylor_doyle_overlay.png`
- `b2_offstoich/figures/fig_b2_hull_finiteT.png`
- `b2_offstoich/run_icet_b2_cluster_expansion.py`
- `b2_offstoich/analysis/icet_b2_cluster_expansion_summary.json`
- `b2_offstoich/analysis/icet_b2_predictions.csv`
- `b2_offstoich/figures/fig_icet_ce_parity.png`
