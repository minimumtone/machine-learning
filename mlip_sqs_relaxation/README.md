# MLIPによる二元系SQS安定構造探索

BCC / FCC の 4×4×4 supercell（BCC: 128原子, FCC: 256原子）に対し、
HEAでよく用いられる23元素の二元系SQS構造を生成し、
MLIP（MACE-MP-0）でエネルギー・力を最小化して構造緩和するパイプライン。

## 対象

- 元素（23種）: Ag, Al, Au, Co, Cr, Cu, Fe, Hf, Ir, Mn, Mo, Nb, Ni, Pd, Pt, Re, Rh, Ru, Ta, Ti, V, W, Zr
- 組成: B元素 0 / 25 / 50 / 75 / 100 at.% の5通り
  - 0, 100 at.%（純元素）は理想supercellを1構造
  - 25, 50, 75 at.% は乱数シードを変えたSQSを各3構造
- 格子: BCC, FCC（--lattice で指定）

## セットアップ

```bash
sudo apt-get install python3-dev   # icetのビルドに必要
pip install ase icet mace-torch
```

## 使い方

### 1. SQS構造生成

```bash
# 全23元素・全253ペア（時間がかかるためペア指定・元素サブセットも可能）
python generate_sqs_structures.py --lattice bcc --outdir structures
python generate_sqs_structures.py --lattice fcc --outdir structures

# サブセット例
python generate_sqs_structures.py --lattice bcc --pairs "Fe-Ni,Nb-Ta"
python generate_sqs_structures.py --lattice bcc --elements "Fe,Ni,Cr"
```

出力: `structures/<lattice>/<A>-<B>/A75B25_seed1.extxyz` など、
および構造一覧 `structures/<lattice>/manifest.csv`。

SQSは icet の `generate_sqs_from_supercells`（MCアニーリング）で生成。
初期格子定数は金属半径からVegard則で補間（緩和で最適化されるため初期値の役割のみ）。

### 2. MLIP構造緩和

```bash
python relax_mlip.py --manifest structures/bcc/manifest.csv
python relax_mlip.py --manifest structures/fcc/manifest.csv --device cuda
```

- MACE-MP-0（既定: medium, float64）+ FrechetCellFilter + FIRE で
  原子位置とセルを同時に緩和（既定 fmax = 0.02 eV/Å）。
- 出力: `relaxed/<name>_relaxed.extxyz` と `relaxed/results.csv`
  （全エネルギー, eV/atom, 最大力, 体積, 収束状況など）。
- `--limit N` で先頭N構造のみ（動作確認用）、`--model small` で高速化。

## 計算規模の目安

全253ペア × 3組成 × 3シード × 2格子 ≒ 4,554構造の緩和となるため、
GPU（`--device cuda`）の利用、またはペア分割による並列実行を推奨。
