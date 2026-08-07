# Be-Co 16原子 MAGMOM / E--V 計算

このディレクトリは、BCC Be8Co8 SQSで観測された
\(\Omega_\mathrm{sf}\) の符号反転を、単なる再緩和ではなく固定体積
\(E(V)\) 曲線で検証するための入力を収録する。主仮説は、Be-Co合金の
E--V曲線が平坦で、平衡体積付近の約2%の違いが小さなエネルギー差にしか
ならないことである。副仮説は、Coの初期磁気モーメントによる高スピン/
低スピンまたは別磁気極小である。

純Be・純Co端点は16原子と128原子で原子体積が0.05%以内に一致している
ため、今回の入力は合金だけを対象とし、端点計算は生成しない。

## ケース

### 全緩和 MAGMOM走査（`MAGMOM/`、5件）

全ケースは既存のBe8Co8再計算設定を基準にし、`ISIF=3`でセル形状・体積・
原子位置を緩和する。

- `NM`: `ISPIN=1`、`MAGMOM`なし
- `FM_low`: Be 8個を0、Co 8個を0.5
- `FM_ref`: Be 8個を0、Co 8個を1.5
- `FM_high`: Be 8個を0、Co 8個を3.0
- `AFM`: Coの符号を `+ - + + - + - -` とする

POSCARはBe 8原子、Co 8原子の順である。AFM配置は、周期境界条件を含む
Coサイト間距離を調べ、最短距離のCo--Coグラフを二部グラフに分けたもの
である。POSCARのCo順では、Co 1, 3, 4, 6を正、Co 2, 5, 7, 8を負に
割り当てると、最短Co--Co結合がすべて反平行になる。

### 固定体積 E--V走査（`EV/`、14件）

再計算16原子の体積
\(V_\mathrm{ref}=150.27357974\ {\rm \AA^3}\)を基準に、
\(V/V_\mathrm{ref}=0.94,0.96,\ldots,1.06\)の7点を作る。

- `EV/FM_ref`: `ISPIN=2`, `MAGMOM = 8*0.0 8*1.5`
- `EV/NM`: `ISPIN=1`, `MAGMOM`なし

`ISIF=4`はVASPでイオン位置とセル形状を緩和し、体積を固定する設定で
あり、今回の固定体積E--V曲線の目的に適合する。`ISIF=2`ではセル形状も
固定されるため、応力異方性を吸収するセル形状緩和を失う。したがって今回
は`ISIF=4`を採用した。出力後はCONTCARの体積が目標値を保っているか確認
する必要がある。

## 実行

```bash
cd vasp_inputs
python generate_beco_magmom_ev.py
cd BECO_MAGMOM_EV
export VASP_PP_PATH=/path/to/vasp
export VASPBIN=/path/to/vasp_std
bash make_potcar.sh
bash run_all.sh 8
```

`make_potcar.sh`がPOTCARを生成する。POTCARはリポジトリにコミットしない。
`run_all.sh`は各ケースを`mpirun -np 8`で順番に実行する。

抽出した固定体積データを`volume_A3,energy_eV`列のCSVにまとめた後、
Birch--Murnaghanフィットを実行する。

```bash
python vasp_inputs/fit_beco_ev.py ev_results.csv
```

フィットの`V0_A3`と、\(B_0\)（GPa）が磁気系列ごとに比較すべき値である。
今回の入力生成時点ではVASP結果は存在しないため、数値結果は記載していない。
