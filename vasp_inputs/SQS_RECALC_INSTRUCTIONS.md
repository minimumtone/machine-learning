# SQS 2x2x2 再計算指示書

目的: 2x2x2（BCC 16原子 / FCC 32原子）SQSデータセットの完全化。
4x4x4 大型セルは本セット完了後に着手する。

対象の選定根拠は `paper/analyze_sqs_recalc_needs.py` の出力
（`paper/sqs_recalc_needs_report.txt`, `paper/sqs_vs_mp_oqmd_deviation.csv`）。

## 入力生成

```bash
cd vasp_inputs
python generate_sqs_recalc_2x2x2.py   # -> SQS_RECALC_2x2x2/ (93計算)
```

## 対象一覧（優先順）

### P1: BCC純元素 16件 + AF変種2件（BCC_PURE/, 18計算）
King乖離>3%で現在King/MP値に置換されている元素:
Ag, Au, Be, Ca, Cr, Fe, Ir, Mn, Os, Pb, Pd, Pt, Re, Rh, Ru, Sn

- 旧計算との違い: MAGMOM明示（Cr/Mn はAF変種 `_AF` も生成、エネルギーの低い方を採用）、
  EDIFF 1e-6 / NELM 300 / NSW 200 / PREC Accurate / ISMEAR 1。
- 注意: Mn(-10.5%), Pb(+16.2%) は磁性収束・軟らかいポテンシャルが原因の可能性が高い。
- MP-BCCとの比較（同一汎関数チェック）では >3% 乖離は Pb, Cr, Al, Fe, Cu の5件のみ。
  貴金属系(Ag/Au/Ir/Pt/Pd/Rh/Ru/Os)はSQSとMPが一致しており、King乖離は
  「準安定BCC相のDFT-実験差」が本質である可能性が高い。再計算で値が変わらなければ
  その結論が確定する（A1の循環性議論に直結）。

### P2: 欠損ペア 51件（BCC_MISSING/）
B2データには存在するがSQSに無いペア。Co-Mn（テスト合金AlCoMnNiVで必要）を含む。
欠損側の|Ω_B2|平均0.087 vs 収録側0.059 と系統的に偏っているため、
Ω=0フォールバックはバイアス源。Pb系24件・Ca系5件・Sn系10件が主。

### P3: 乖離大ペア 19件（BCC_HIGHDEV/）
SQS体積がMP/OQMDのB2体積から>10%乖離する検証再計算
（Mg-Pt +21%, Ca-Nb -19% 等）。秩序化効果(B2 vs random)による真の差か、
収束不良かを判別する。全ランキングは `paper/sqs_vs_mp_oqmd_deviation.csv`。

### P4: FCC純元素 5件（FCC_PURE/）
Nb, Os, Pb, Pd, Pt（A16A16, 32原子）。

## 実行

```bash
cd vasp_inputs/SQS_RECALC_2x2x2
export VASP_PP_PATH=/path/to/vasp_pp
export VASPBIN=/path/to/vasp_std
bash run_all.sh 8 4        # 8並列 x 4コア（緩和→static を自動実行）
```

- 完了済みディレクトリ（static_OUTCAR あり）は自動スキップ。
- 概算コスト: 16原子 x 93計算。32コアで1〜2日程度。

## 結果の取り込み

```bash
cd vasp_inputs
python extract_vasp_results.py    # sqs_results.csv 形式で抽出
# 抽出結果を data/sqs_results.csv にマージ（同名ディレクトリは新しい値で置換）
cd ../paper
python analyze_sqs_recalc_needs.py   # 置換数・欠損数の減少を確認
python generate_all_figures.py       # 全数値・図・paper_metrics.json 再生成
```

## 判定基準（論文A1への反映）

1. 再計算後もSQS純元素体積がKingから>3%乖離する元素が残る場合:
   King置換は「DFT-実験系統差の補正」であり、循環性の但し書きを維持。
2. 再計算で乖離が解消する（収束不良が原因だった）場合:
   置換を撤廃し raw SQS 一本で q_opt を再評価。q≈1 主張の生存確認。
3. Co-Mn 収録後、AlCoMnNiV 予測とSQSテストRMSEの変化を確認。
