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

## セットアップ（Linux VM / WSL2 / ネイティブ Linux）

推奨は Linux VM（Ubuntu 22.04/24.04）。Windows ネイティブでも動く可能性がありますが、
icet の C++ ビルド・並列処理の効率から **WSL2 または Linux VM** を推奨します。

```bash
# 1回だけ実行するシステムセットアップ
bash mlip_sqs_relaxation/setup_vm.sh

# 手動で行う場合
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip python3-venv git
pip install ase icet mace-torch
```

`python3-dev` は `icet` の C++ 拡張コンパイルに必須です。

## 使い方

### 1. SQS構造生成（逐次）

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

### 2. MLIP構造緩和（逐次）

```bash
python relax_mlip.py --manifest structures/bcc/manifest.csv
python relax_mlip.py --manifest structures/fcc/manifest.csv --device cuda
```

- MACE-MP-0（既定: medium, float64）+ FrechetCellFilter + FIRE で
  原子位置とセルを同時に緩和（既定 fmax = 0.02 eV/Å）。
- 出力: `relaxed/<name>_relaxed.extxyz` と `relaxed/results.csv`
  （全エネルギー, eV/atom, 最大力, 体積, 収束状況など）。
- `--limit N` で先頭N構造のみ（動作確認用）、`--model small` で高速化。

### 3. 全ケース並列実行（推奨）

24コア VM などで全ケースを一括実行する場合:

```bash
mkdir -p /work/sqs_mlip
python mlip_sqs_relaxation/run_parallel.py \
    --workdir /work/sqs_mlip \
    --lattices bcc fcc \
    --n-steps-sqs 10000 \
    --n-steps-relax 500 \
    --model small \
    --device cpu \
    --workers 24
```

`--skip-generate` ですでに生成済みの構造だけ緩和できます。

## 計算規模と所要時間の目安（CPU実測ベース）

- 構造数（既定 3 シード）: 全 253 ペア × 3 組成 × 3 シード + 純元素 23 = **約 2,300 構造 / 格子**（BCC+FCC 合計 **約 4,600 構造**）
- 構造数（1 シード）: `--n-seeds 1` にすれば **約 780 構造 / 格子**に削減
  （BCC+FCC 合計 **約 1,560 構造**）

実測値（このセッションの CPU、MACE-small、n_steps_sqs=10000）:

| 工程 | 1 構造あたり | 格子あたり（2,300構造） | 備考 |
|---|---|---|---|
| BCC SQS 生成 | 14 s | ~9 h | n_steps=10000 |
| FCC SQS 生成 | ~61 s | ~39 h | n_steps=10000（256原子のため） |
| BCC MLIP 緩和 | ~2 min | ~77 h | max-steps=500、small |
| FCC MLIP 緩和 | ~1 min | ~38 h | max-steps=500、small |

**逐次実行合計: 約 163 h ≒ 6.8 日**

24 コアで効率よく並列化できれば（SQS生成・緩和ともタスク独立):

```
163 h / 24 ≈ 6.8 h
```

現実的にはメモリ・I/O オーバーヘッドを見込んで **約 10–20 時間** と見込んでください。
MACE モデルはワーカーごとにロードされるため、24 ワーカーではメモリに余裕がある前提です
（small モデルで 1 ワーカーあたり 1–2 GB 程度）。

Windows CPU（ネイティブ）でも終了は可能ですが、icet のビルドと Python 並列処理の効率から
WSL2 / Linux VM を強く推奨します。
