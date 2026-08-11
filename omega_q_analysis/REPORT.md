# 検証2・3: King/Alonso構造因子($\Omega_{\mathrm{sf}}$)の有用性と$q$因子の物理的起源

データ: 既存の95 HEA（校正64 + 独立テスト31、BCC 46 / FCC 49）、
DFT由来の構造別$\Omega_{\mathrm{sf}}$（B2 465ペア・L1$_2$ 441ペア、`paper/results_omega_sf.csv`）、
BCC-SQS 16原子DFT $\Omega_{\mathrm{sf}}$（414ペア、`data/sqs_results.csv`）、
King (1966) Table II + Alonso (2022) Table 3の実験$\Omega_{\mathrm{sf}}$（522方向ペア）。

## 1. 検証2: $\Omega_{\mathrm{sf}}$構造因子はHEA格子定数予測にどの程度有用か

独立テスト31 HEAのRMSE (Å)（既存検証 `paper/verify_king_alonso_model.py`・`paper_metrics.json` より）:

| モデル | 全体(31) | BCC(17) | FCC(14) |
|---|---|---|---|
| Vegard則（King体積） | 0.0202 | 0.0226 | 0.0170 |
| **実験$\Omega_{\mathrm{sf}}$（King+Alonso, q=1）** | 0.0564 | 0.0473 | 0.0658 |
| 実験$\Omega_{\mathrm{sf}}$（q校正: $q_{\mathrm{BCC}}$=0.006, $q_{\mathrm{FCC}}$=0.034） | 0.0198 | 0.0226 | 0.0157 |
| DFT-$\Omega_{\mathrm{sf}}$（B2/L1$_2$, $q_{\mathrm{BCC}}$=0.49, $q_{\mathrm{FCC}}$=0.13） | 0.0155 | 0.0157 | 0.0152 |
| **DFT-$\Omega_{\mathrm{sf}}$（BCC-SQS参照, q=1）** | 0.0138 | 0.0125 | (L1$_2$共通) |

**結論**:
1. **実験（King/Alonso）$\Omega_{\mathrm{sf}}$をそのまま（q=1）使うとVegard則より悪化する**
   （希薄極限の実験サイズファクターは濃厚HEAの体積偏差を過大評価）。qを校正すると
   $q \approx 0$に潰れ、実質Vegard則に退化する。つまり**実験$\Omega_{\mathrm{sf}}$はHEA
   格子定数予測にはほぼ寄与しない**。
2. **DFTで構造別（B2/L1$_2$/SQS）に再定義した$\Omega_{\mathrm{sf}}$は有用**。特にBCC系では
   SQS+DFT参照（q=1、校正不要）でVegard比44.5%改善（RMSE 0.0226→0.0125 Å）。
3. FCC系はVegard則RMSE 0.0170 Å（体積偏差が実験ノイズ床レベル）で、$\Omega_{\mathrm{sf}}$
   補正の余地自体が小さい（改善10%程度）。

## 2. 検証3: $q_{\mathrm{BCC}} \to 1$・$q_{\mathrm{FCC}} \to 0$の物理的起源

モデル $V = V_{\mathrm{Vegard}} + q\,C$ の$q$は原点通過回帰の勾配であり、厳密に分解できる:

$$\hat q = \frac{\sum \Delta V_{\mathrm{exp}}\, C}{\sum C^2}
= \frac{r\,\sigma(\Delta V_{\mathrm{exp}})\,\sigma(C) + \overline{\Delta V}\,\bar C}{\sigma(C)^2 + \bar C^2}$$

ここで $\Delta V_{\mathrm{exp}} = V_{\mathrm{exp}} - V_{\mathrm{Vegard}}$（真の非Vegard偏差）、
$C = \sum_i \sum_{j\ne i} c_i c_j V_j \Omega_{\mathrm{sf}}(i,j)$（q=1での$\Omega$補正量）。
$\bar C \ne 0$（補正はほぼ全て負）のため、切片あり回帰の勾配
$b = \mathrm{cov}/\mathrm{var} = r\,\sigma(\Delta V)/\sigma(C)$ とは一致しない点に注意。
$\hat q$は「**(a) 補正と真の偏差の相関 $r$**」と「**(b) 振幅（分散・平均）の整合**」の
両方を反映する（本解析 `analyze_omega_q_origin.py`、95 HEA、
`results_q_decomposition.json`の`q_hat_check_decomposed`で上式の恒等性を数値確認済み）:

| | BCC (B2-$\Omega$) | BCC (SQS-$\Omega$) | FCC (L1$_2$-$\Omega$) |
|---|---|---|---|
| $\sigma(\Delta V_{\mathrm{exp}})$ (Å$^3$/atom) | 0.419 | 0.419 | 0.127 |
| 実験ノイズ床 $\sigma_V^{\mathrm{noise}}$ | 0.250 | 0.250 | 0.159 |
| **信号/ノイズ比** | **1.68** | **1.68** | **0.80** |
| $\sigma(C)$ | 0.194 | 0.116 | 0.226 |
| 平均 $\bar C$ / $\overline{\Delta V}$ | −0.248 / −0.184 | −0.137 / −0.184 | −0.175 / +0.007 |
| 相関 $r$ | 0.14 | 0.41 | 0.60 |
| 切片あり勾配 $b = r\,\sigma(\Delta V)/\sigma(C)$ | 0.29 | 1.49 | 0.34 |
| **$\hat q$（原点通過回帰、主指標）** | **0.57** | **1.41** | **0.19** |

### $q_{\mathrm{FCC}} \to 0$ の物理根源

1. **信号がノイズ以下**: FCC HEAの非Vegard偏差の分散（0.127 Å$^3$/atom）は実験再現性
   由来のノイズ床（0.159）を下回る（信号/ノイズ = 0.8）。**FCC（最密充填, 配位数12,
   充填率0.74）では体積の加法性（Vegard則）がほぼ厳密に成り立ち**、補正すべき真の偏差が
   ほとんど存在しない。
2. **補正の過大振幅**: L1$_2$規則相の$\Omega_{\mathrm{sf}}$補正の振幅（$\sigma(C)$=0.226、
   $|\bar C|$=0.175）は実偏差（$\sigma$=0.127、$|\overline{\Delta V}|$=0.007）を大きく上回る。
   回帰は振幅を合わせるため勾配が縮小し（切片あり$b$=0.34、原点通過$\hat q$=0.19）、
   ノイズによる減衰（attenuation）も加わって$q_{\mathrm{FCC}} \to 0.1$台に落ちる。
3. 物理的解釈: 最密充填構造では各原子の局所体積が12個の最近接原子で幾何的に強く拘束され、
   原子サイズの「環境依存性」（化合物形成による有効半径の変化）が体積に現れにくい。
   さらに典型的FCC HEA（Cantor系3d遷移金属）は$\delta \approx 2.4\%$とサイズミスマッチ
   自体が小さい（BCC系 $\delta \approx 3.9\%$、`fig_delta_mismatch.png`）。

### $q_{\mathrm{BCC}} \to 1$ の物理根源

1. **真の信号が存在**: BCC HEA（主に耐火系）の非Vegard偏差はノイズ床の1.68倍で、
   補正すべき系統的な体積収縮が実在する（$\Omega$の64–86%が負）。
2. **参照構造の秩序度が鍵**: B2（Warren-Cowley $\alpha = -1$、全最近接が異種原子）の
   $\Omega_{\mathrm{sf}}$は異種対効果を過大評価し、$q_{\mathrm{BCC}} = 0.49$–0.57の縮小が必要。
   **SQS（$\alpha \approx 0$、HEAと同じランダム配位）に置き換えると$r$は0.14→0.41へ改善し
   $\hat q$は0.57→1.41と1を跨ぐ**（本解析; 論文の校正では純元素参照体積のキュレーション後
   $q_{\mathrm{opt}} = 0.95$–1.27）。つまり$q = 1$とは「参照二元系がHEAと同じ短距離秩序
   （ランダム）と同じDFT参照系で記述されている」ことの表れであり、$q \ne 1$は
   (i) 秩序度の不一致（B2 vs ランダム）と (ii) 参照体積のDFT–実験不整合の補正係数である。
3. 物理的解釈: BCC（配位数8+6、充填率0.68）は開いた構造で局所緩和の自由度が大きく、
   異種対形成による電荷移動・結合短縮が体積に直接反映される。ゆえに二元系ランダム固溶体
   （SQS）の体積偏差が多元系HEAへそのまま（q=1で）転写される。

### $q=1$の立ち位置（RMSEプラトーの観点）

RMSE$(q)$曲線は最適値付近で平坦であり（`fig_q_profile.png`）、$q$は強くは同定されない:

| | $\hat q$ | 95% CI | RMSE 5%プラトー範囲 | RMSE($q{=}1$)/min |
|---|---|---|---|---|
| BCC (B2-$\Omega$) | 0.57 | [0.18, 0.96] | [0.15, 1.00] | 1.05 |
| BCC (SQS-$\Omega$) | 1.41 | [0.78, 2.03] | [0.73, 2.09] | **1.02** |
| FCC (L1$_2$-$\Omega$) | 0.19 | [0.08, 0.31] | [0.07, 0.32] | 2.25 |

- **BCC（SQS参照）では$q=1$は95%信頼区間のほぼ中央にあり、RMSEの悪化は最小値比で2%のみ**。
  $\hat q=1.41$という点推定値自体に意味は薄く、データは「$q=1$（補正をそのまま使う）を
  棄却できない」というのが正確な言明。つまり$q=1$は「プラトー内の理論的に自然な参照点
  （参照二元系がHEAと同じ短距離秩序・同じDFT参照系なら補正は無調整で転写される）」という
  立ち位置であり、統計的に「1.41が1より良い」とは言えない。
- B2参照でも$q=1$はプラトー端（RMSE+5%）に乗るが、CIの上端。参照の秩序度不一致の分だけ
  系統的に縮小が必要という解釈と整合。
- FCCでは$q=1$はRMSEを最小値の2.25倍に悪化させプラトー外。$q=0$（Vegard）側のみが許容される。

### まとめ（1行）

$q_s$は「参照構造とHEAの短距離秩序・参照体積の整合度」を測る回帰係数であり、
**BCCでは偏差が実在しSQS参照で整合するため$q \to 1$、FCCでは最密充填の幾何拘束により
Vegard則が実験精度内で成立し、回帰の縮小効果で$q \to 0$となる**。

## 図

- `fig_q_scatter_bcc_fcc.png` — $\Delta V_{\mathrm{exp}}$ vs $C$ 散布図と$q$勾配（B2/SQS/L1$_2$）
- `fig_q_decomposition.png` — $q = r \times$振幅比 の分解（ノイズ床付き）
- `fig_delta_mismatch.png` — BCC/FCC HEAのサイズミスマッチ$\delta$分布
- `fig_q_profile.png` — RMSE$(q)$プロファイル（プラトーと$q=1$の位置）
