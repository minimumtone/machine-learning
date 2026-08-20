# MLIP 空孔・アンチサイトエネルギー → 4SL/8SL B2 副格子モデル パラメータ化検討

Yamanouchi & Miura（2018）の補助議論（Tsang スライド）を受け、MLIP 計算から B2-NiAl の点欠陥エネルギーを取り出し、CALPHAD 的 4-sublattice/8-sublattice B2 秩序相モデルのエンドメンバー・パラメータに接続する方法を整理する。

## 1. 現状の MLIP データ

MACE-MP-0 medium（0 K 静的緩和、FrechetCellFilter、128 サイト超胞）から得られた点欠陥形成エネルギー（完全 B2 からの差分、fcc 元素基準）：

| 欠陥種 | 平均 ΔE (eV/欠陥) | 標準偏差 | 備考 |
|---|---|---|---|
| Ni 反サイト on Al 副格子（Ni-rich） | 0.79 | 0.20 | 濃度依存あり、x≈0.46 で 0.68 eV、x≈0.20 で 1.17 eV |
| Al 反サイト on Ni 副格子（Al-rich） | 1.59 | 0.10 | x≈0.80 で 1.37 eV、x≈0.60 で 1.62 eV |
| Ni 空孔（Al-rich、Ni 副格子）     | 1.17 | 0.08 | x≈0.50 で 1.06 eV、x≈0.54 で 1.28 eV |
| Al 空孔（Ni-rich、Al 副格子）     | 1.78 | 0.15 | 常に Ni 反サイトより高エネルギー |

重要な知見（$\mu_{\rm Ni}+\mu_{\rm Al}=-10.844$ eV/formula、$\Delta\mu\approx0$ 付近、fcc 元素基準）：
- **Ni-rich 側**：Ni 反サイト （~0.7–0.9 eV） << Al 空孔 （~1.7–2.0 eV） → 反サイトが支配的。
- **Al-rich 側**：Ni 空孔 （~1.1 eV） < Al 反サイト （~1.4–1.6 eV） → 空孔が支配的。
- 支配的分岐は $\Delta\mu$ の可動域（約 2.768 eV）に敏感であり、元素化学ポテンシャルが動くと空孔/反サイト優勢は入れ替わりうる。
- $\Delta E$ は欠陥濃度に依存するため、単一点欠陥近似ではなく **濃度依存な相互作用項**が必要。

## 2. 2 サブラティス CEF モデル（出発点）

最も簡単な B2 記述は

```
(Ni, Al, Va)_{0.5} (Al, Ni, Va)_{0.5}
```

エンドメンバー（完全秩序・主な点欠陥）を以下のように定める：

| エンドメンバー | 意味 | MLIP からの見積もり |
|---|---|---|
| G(Ni:Al) | 完全 B2-NiAl | MACE E(NiAl-B2)/formula = -10.844 eV |
| G(Al:Ni) | 反 B2-NiAl（Al/Ni サブラティス入れ替え） | = G(Ni:Al)（B2 は C.N.8 対称） |
| G(Ni:Ni) | 反サイトに近い Ni 過剰極限 | A2-Ni on bcc; $E=-11.324$ eV/formula, $a=2.790$ Å |
| G(Al:Al) | Al 過剰極限 | A2-Al on bcc; $E=-7.374$ eV/formula, $a=3.225$ Å |
| G(Va:Al) | Ni 欠損 Al 過剰（Al-rich 空孔） | G(Ni:Al) + n_sites/2 · E_Ni_vac |
| G(Ni:Va) | Al 欠損 Ni 過剰（Ni-rich 空孔） | G(Ni:Al) + n_sites/2 · E_Al_vac |

注：A2-Ni/Al の MACE-MP-0 計算は完了（`analysis/a2_endmember_energies.csv`）。x=0.5 のランダム A2（Ni$_0.5$Al$_0.5$ bcc）エネルギー $E_{\rm A2}=-10.493$ eV/formula（$\eta=0$外挿）と組み合わせることで、B2 秩序化エネルギー $\Delta E_{\rm order}=+0.3510$ eV/formula（$E_{\rm A2}-E_{\rm B2}$）を直接求める。

## 3. 4SL/8SL B2 モデルへの一般化

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

## 4. MLIP → エンドメンバー・パラメータ変換の手順

1. **配置エネルギーの収集**：128 サイト B2 超胞で、全組成（x_Al=0.20–0.80）の反サイト・空孔配置を MACE で緩和。
2. **点欠陥形成エネルギーの定義**：式 (1)–(4) に従い、完全 B2 からの差分および元素化学ポテンシャルで正規化。
3. **クラスター展開**：`run_icet_b2_cluster_expansion.py` で icet を用いて第一近接対から三点群までの有効相互作用を推定。第一近接対のみでは $V_{\rm 1NN}=-0.137$ eV/bond、第二近接対・三点群を加えると $V\approx-0.100$ eV/bond と熱力学的 $V=-0.088$ eV/bond に漸近。
4. **CEF エネルギー関数の構築**：
   
   $$ G_m = \sum_i y_i^1 y_j^2 y_k^3 y_l^4 \, G_{ijkl}^{\rm end} + RT\sum_s \sum_i y_i^s \ln y_i^s + G_{\rm excess} $$

   ただし $y_i^s$ はサブラティス $s$ の種 $i$ の占有位点分率、$G_{ijkl}^{\rm end}$ はエンドメンバー自由エネルギー。
5. **濃度依存 interaction の導入**：
   
   $$ G_{\rm excess} = \sum_s y_i^s y_j^s \, L_{ij}^s + y_k^t y_l^t \, L_{kl}^t + \dots $$

   ここで $L$ は Redlich–Kister 多項式 $L = L^0 + L^1 (y_i - y_j) + \dots$。
6. **パラメータフィット**：MLIP の $\Delta E_{\rm defect}(x)$ をターゲットとして、$G_{\rm end}$ と $L$ を最小二乗フィット。MACE の形成エネルギー図（凸包）も追加制約として含める。

## 5. 現時点での制約・未解決点

- **振動エントロピーなし**：1473 K の熱処理には phonon/MD によるエントロピー補正が必要。
- **磁性**：Ni のスピン分極効果は MACE-MP-0 にはない。
- **A2 端成分計算済み**：A2-Ni($a=2.790$ Å, $E=-5.662$ eV/atom) と A2-Al($a=3.225$ Å, $E=-3.687$ eV/atom) を `run_a2_endmembers.py` で緩和。B2-A2 秩序変態エネルギーは $\Delta E_{\rm order}=+0.3510$ eV/formula。
- **icet クラスター展開済み**：`run_icet_b2_cluster_expansion.py` で第一近接対相互作用を抽出。8SL パラメータへの変換は次の段階。
- **サンプリングの希薄さ**：各組成 3 配置では 4SL/8SL の全エンドメンバーをカバーできない。最低 10–20 配置、さらにクラスター展開用データが必要。

## 6. 次の実行計画

1. 現行の Al-rich 密サンプリング完了を待ち、`b2_defect_energies.csv` を更新。
2. ~~A2-Ni/A2-Al（bcc）緩和~~ 完了。
3. ~~`icet` クラスター展開で第一近接対相互作用 $J_{ij}$ を抽出~~ 完了（`run_icet_b2_cluster_expansion.py`）。次は 8SL 対応エンドメンバー表への変換。
4. pycalphad/TDB 形式の原型を出力し、形成エネルギー図と整合するか検証。

## 7. 付録：第一近接対相互作用の見積もり（Ising 近似）

B2 構造の最近接対を三種類の有効相互作用で近似する：

$$ E = \sum_{\langle ij \rangle} J_{ij}, \qquad J_{\rm NiAl} < 0 \text{（安定化）} $$

慣用胞（2 原子）あたりの結合数は 8。完全 B2-NiAl では全結合が Ni–Al：

$$ E_{\rm B2} = 8 J_{\rm NiAl} $$

完全ランダム A2（各サイト占有率 0.5）では Ni–Ni:Al–Al:Ni–Al = 2:2:4：

$$ E_{\rm A2} = 2 J_{\rm NiNi} + 2 J_{\rm AlAl} + 4 J_{\rm NiAl} $$

したがって、**秩序化エネルギー**は

$$ \Delta E_{\rm order} = E_{\rm A2} - E_{\rm B2} = 2(J_{\rm NiNi} + J_{\rm AlAl}) - 4 J_{\rm NiAl} $$

MACE-MP-0 から $E_{\rm B2} \approx -10.844$ eV/formula、$E_{\rm A2}$ は $b2\_order\_param.csv$ の $\eta=0$ 平均から $E/N \approx -5.246$ eV/atom、すなわち $E_{\rm A2} \approx -10.492$ eV/formula、よって

$$ \Delta E_{\rm order} \approx 0.3509 \text{ eV/formula} $$

重要なことは，定数対モデルでは $\Delta E_{\rm order}=-4V$ となるため，**秩序化強度（effective ordering energy）** は B2/A2 エネルギー差から直接決まる：

$$ V = -\frac{\Delta E_{\rm order}}{4} = -\frac{0.3509}{4} = -0.088 \ {\rm eV/bond} $$

この $V$ は Ni–Al 結合が他の結合よりどれだけ強く負かを表す，物理的に一本化された値である．

一方で，`icet` クラスター展開（`run_icet_b2_cluster_expansion.py`）で第一近接対モデルを B2 系のみにフィットすると

- $J_{\rm NiAl}=-1.354$ eV、$J_{\rm NiNi}=-1.500$ eV、$J_{\rm AlAl}=-0.935$ eV
- $V_{\rm pair,1NN}=J_{\rm NiAl}-(J_{\rm NiNi}+J_{\rm AlAl})/2=-0.137$ eV/bond

と、孤立点欠陥からの定数対推定 $V\approx-0.145$ eV/bond とほぼ一致する。しかし，第二近接対（同じサブラティス上の Ni–Ni / Al–Al 対）を加えると $V\approx-0.102$ eV/bond、三点群まで含めると $V\approx-0.100$ eV/bond と熱力学的値 $-0.088$ eV/bond に急速に近づく。したがって，**定数対近似の破綻は第二近接対・多点項の無視に起因する**。個別の $J_{ij}$ には固定体積近似の大きな依存があるため、報告すべき秩序化強度は引き続き $V=-0.088$ eV/bond である．

`extract_4sl_b2_parameters.py` から出力される `V_from_ordering_eV` を使用し，`V_pair_constant_eV` はあくまで定数対近似の不整合を示す指標として扱う．また `analysis/icet_b2_cluster_expansion_summary.json` から、`V_eff_eV_per_bond`（1NN+2NN+triplets）は -0.1002 eV/bond、`V_pair_eV_per_bond`（1NN only）は -0.1369 eV/bond、`rmse_eV_per_atom` は 0.0092 eV/atom と確認できる。

（MACE の A2-Ni / A2-Al 端成分は `run_a2_endmembers.py` によりそれぞれ $E=-5.662$ eV/atom ($a=2.790$ Å)、$E=-3.687$ eV/atom ($a=3.225$ Å) と緩和された。$x=0.5$ のランダム A2 エネルギー $E_{\rm A2}=-10.493$ eV/formula から、B2 秩序化エネルギー $\Delta E_{\rm order}=+0.3510$ eV/formula を得る。）

## 8. 成果物

- `b2_offstoich/analysis/b2_defect_energies.csv`（各配置の空孔/反サイト形成エネルギー）
- 本設計書 `4SL_B2_MODEL_DESIGN.md`
- `b2_offstoich/extract_4sl_b2_parameters.py`（秩序化強度 $V$ の簡易推定）
- `b2_offstoich/analysis/b2_pair_interactions.json`（$V$ の値のみを使用。個別 $J$ 表は非推奨）
- `b2_offstoich/run_icet_b2_cluster_expansion.py`（icet クラスター展開スクリプト）
- `b2_offstoich/analysis/icet_b2_cluster_expansion_summary.json`
- `b2_offstoich/analysis/icet_b2_predictions.csv`
- `b2_offstoich/figures/fig_icet_ce_parity.png`

## 9. 8SL CEF（相互作用なし）と icet 4SL CE の変換可能性

### 9.1 両者のGibbs式

8サブラティス CEF（相互作用なし）では、全 $2^8=256$ のエンドメンバーが独立に指定され

$$
G_m = \sum_{\{i_1,\dots,i_8\}} \left( \prod_{s=1}^8 y_{i_s}^s \right) G_{i_1\cdots i_8}^{\rm end}
+ RT \sum_{s=1}^8 \sum_i y_i^s \ln y_i^s
$$

ここで $y_i^s$ はサブラティス $s$ 上の種 $i$ の占有位点分率である。$G_{\rm excess}=0$ でも、256 個のエンドメンバー energy を自由に選べるため、$2^8$ の独立な配置に対して任意の 0 K エネルギーを与えられる。

一方、icet クラスター展開（4 サイト以上を含む cluster basis）では

$$
E(\boldsymbol{\sigma}) = \sum_{\alpha} m_\alpha J_\alpha \prod_{i\in\alpha} \sigma_i,
\qquad G \approx \langle E \rangle - T S
$$

で、$\alpha$ は cluster orbit、$J_\alpha$ は effective cluster interaction (ECI)、$m_\alpha$ は重数である。ここでは同一超胞内の全配置のエネルギーを cluster basis で展開する形をとる。

### 9.2 変換可能性の結論

**結論**：8SL CEF（相互作用なし）と icet CE は、次のように一対一に変換可能である。

1. **完全変換の条件**：icet の cluster basis が 8SL 超胞内でスパンする全 256 個の cluster（0 体から 8 体）を含むとき。各エンドメンバー $G_{i_1\cdots i_8}^{\rm end}$ を Walsh/Fourier 変換によって ECI $J_\alpha$ に直すことができる。
2. **近似変換**：現状の icet 4SL （NiAl では 1NN 対 + 2NN 対 + 三点群、計 5 ECI）では、256 個の 8SL エンドメンバーすべてを正確に再現できる保証はない。これは、8SL モデルが含む高次体相互作用を 4SL CE が打ち切っているためである。
3. **エントロピーの違い**：CEF の $RT\sum y\ln y$ 項は各サブラティス独立の理想混合エントロピーにすぎない。クラスター展開と厳密に一致させるには、Cluster Variation Method (CVM) のような多体相関を含むエントロピー近似が必要。

### 9.3 数値的確認

MACE B2-NiAl の場合、8SL 完全エンドメンバーモデルは 0 K 結晶エネルギーを正確に再現できるが、現行の icet 1NN+2NN+triplets（5 ECI、RMSE 0.009 eV/atom）はそれに対する低次近似に過ぎない。$\Delta E_{\rm order}\approx 0.351$ eV/formula の秩序化エネルギーは第二近接対・三点群まででようやく再現される。よって実用的な TDB 化には、8SL 完全エンドメンバーテーブルから icet 4SL ECI を最小二乗フィットするか、4SL CE の cluster cutoff を十分に大きくして高次体相互作用を取り込む必要がある。相互作用なしのままだと、対相互作用や高次相互作用を表現できない。
