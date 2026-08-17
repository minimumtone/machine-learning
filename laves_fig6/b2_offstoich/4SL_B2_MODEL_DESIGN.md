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

重要な知見：
- **Ni-rich 側**：Ni 反サイト （~0.7–0.9 eV） << Al 空孔 （~1.7–2.0 eV） → 反サイトが支配的。
- **Al-rich 側**：Ni 空孔 （~1.1 eV） < Al 反サイト （~1.4–1.6 eV） → 空孔が支配的。
- ΔE は欠陥濃度に依存するため、単一点欠陥近似ではなく **濃度依存な相互作用項**が必要。

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
| G(Ni:Ni) | 反サイトに近い Ni 過剰极限 | A2-Ni（bcc）への外挿 or x→0.5 Ni-rich 極限 |
| G(Al:Al) | Al 過剰极限 | A2-Al（bcc）への外挿 or x→0.5 Al-rich 極限 |
| G(Va:Al) | Ni 欠損 Al 過剰（Al-rich 空孔） | G(Ni:Al) + n_sites/2 · E_Ni_vac |
| G(Ni:Va) | Al 欠損 Ni 過剰（Ni-rich 空孔） | G(Ni:Al) + n_sites/2 · E_Al_vac |

注：現状では A2-Ni/Al の MLIP 計算がないため、x=0.5 付近の点欠陥データを用いてエンドメンバーを **x 依存として表現**し、相互作用パラメータに変換する方法を採用する。

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

## 4. MLIP → エンドメンバー・パラメータ変換の手順

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

## 5. 現時点での制約・未解決点

- **振動エントロピーなし**：1473 K の熱処理には phonon/MD によるエントロピー補正が必要。
- **磁性**：Ni のスピン分極効果は MACE-MP-0 にはない。
- **A2 端成分未計算**：完全 A2-Ni/A2-Al の MLIP データがないため、B2-A2 秩序変態エネルギーは外挿依存。
- **サンプリングの希薄さ**：各組成 3 配置では 4SL/8SL の全エンドメンバーをカバーできない。最低 10–20 配置、さらにクラスター展開用データが必要。

## 6. 次の実行計画

1. 現行の Al-rich 密サンプリング完了を待ち、`b2_defect_energies.csv` を更新。
2. A2-Ni/A2-Al（bcc）の参考計算を追加し、完全秩序化エネルギーを推定。
3. `icet` クラスター展開で第一近接対相互作用 J_{ij} を抽出し、8SL 対応エンドメンバー表を作成。
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

$$ \Delta E_{\rm order} \approx +0.35\text{–}0.40 \text{ eV/formula} $$

単一点欠陥形成エネルギーからさらに $J_{\rm NiNi}$, $J_{\rm AlAl}$ の制約が得られる。例えば、Ni 反サイト（Ni on Al サイト）を 1 つ作ると周囲 8 本の Ni–Al 結合の一部が Ni–Ni / Al–Al に置き換わる。正確なカウントは 8 サブラティス / クラスター展開が必要だが、**数量級として $J_{\rm NiAl}$ は $J_{\rm NiNi}$, $J_{\rm AlAl}$ より約 0.1 eV 強く負**（Ni–Al 結合が優先的）。

### 現状の推定値（`b2_pair_interactions.json`）

平均点欠陥エネルギーと秩序化エネルギーから定数対近似で推定：

| 相互作用 | 値 (eV/結合) |
|---|---|
| $J_{\rm NiAl}$ | -1.356 |
| $J_{\rm NiNi}$ | -1.257 |
| $J_{\rm AlAl}$ | -1.157 |

定数対モデルから予測される A2 エネルギーは -10.249 eV/formula、観測は -10.493 eV/formula となり、**定数対モデルでは秩序化エネルギーを 0.25 eV/formula 過大評価**する。これは濃度が高い点欠陥での相互作用（反サイト同士・空孔同士）を無視しているためで、4SL/8SL モデルでは組成依存な $L$ パラメータで補正する必要がある。

## 8. 成果物

- `b2_offstoich/analysis/b2_defect_energies.csv`（各配置の空孔/反サイト形成エネルギー）
- 本設計書 `4SL_B2_MODEL_DESIGN.md`
- `b2_offstoich/extract_4sl_b2_parameters.py`（対相互作用の簡易推定）
- `b2_offstoich/analysis/b2_pair_interactions.json`
