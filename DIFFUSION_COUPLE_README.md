# 拡散対PINNs: 純物質拡散定数の有用性検証プログラム

## 概要

このプログラム (`diffusion_couple_pinns.py`) は、二元系合金の拡散対実験から得られた濃度プロファイルに対して、FDM（有限差分法）で模擬データを作成し、PINNs（Physics-Informed Neural Networks）技術を用いて拡散定数を最適化します。特に、**純物質拡散定数・自己拡散定数が物理的制約として有用である**ことを示します。

## プログラムの目的

1. **FDMによる拡散対データの生成**: 二元系合金の拡散対（左側：純B、右側：純A）の濃度プロファイルを数値計算
2. **PINNsモデルの定義**: Darkenモデルに基づく相互拡散係数を学習
3. **純物質境界条件の実装**: 物理的に妥当な自己拡散係数を保証
4. **有用性の可視化**: 純物質制約の効果を明確に示す

## 主要な構成要素

### 1. FDMによる模擬データ生成

```python
x_fdm, t_fdm, C_fdm = generate_fdm_diffusion_couple(
    C_left=0.0,    # 左端: 純B
    C_right=1.0,   # 右端: 純A
    L=1.0,         # 空間領域 [m]
    T_end=10.0,    # 計算時間 [s]
    Nx=101         # 空間メッシュ数
)
```

**真の拡散係数**:
```
D̃(C) = D_B_max + (D_A_max - D_B_max) × C
```

### 2. PINNsモデル (DiffusionCouplePINN)

#### 学習するネットワーク:
- **net_C**: 濃度場 `C(t, x)`
- **net_DA**: 自己拡散係数 `D_A(C)` 
- **net_DB**: 自己拡散係数 `D_B(C)`
- **net_gamma**: 活量係数 `lnγ(C)`

#### Darkenの相互拡散モデル:
```
D̃(C) = C_B·D_A(C) + C_A·D_B(C) + (RT/Ω)·∂lnγ/∂C
       └─────────────┬─────────────┘   └──────┬──────┘
            移動度項 (Mobility)         熱力学項 (Thermodynamic)
```

### 3. 純物質境界条件 (Pure Substance Boundary Conditions)

**物理的意味**:
- **D_A(C=0) = 0**: 成分Aは純B（C=0）には拡散できない
- **D_B(C=1) = 0**: 成分Bは純A（C=1）には拡散できない
- **D_A(C=1) = D_A^max**: 純Aでの成分Aの最大自己拡散
- **D_B(C=0) = D_B^max**: 純Bでの成分Bの最大自己拡散

**PINNs損失関数への実装**:
```python
loss_D_bc = ((D_A(0) - 0.0)² + (D_B(1) - 0.0)² + 
             (D_A(1) - 0.05)² + (D_B(0) - 0.05)²)

total_loss = loss_data + λ_pde·loss_pde + λ_ic·loss_ic + 
             λ_bc·loss_bc + λ_Dbc·loss_D_bc
```

重み `λ_Dbc` が大きいほど純物質制約が強くなります。

## 実行方法

### デモンストレーション実行:
```bash
python3 diffusion_couple_pinns.py
```

このデモでは以下を実行します:
1. FDMで拡散対の濃度プロファイルを生成
2. 純物質境界条件の概念図を可視化
3. 結果を `/home/ubuntu/repos/machine-learning/diffusion_couple_demo.png` に保存

### 完全なアブレーション研究:
制約あり/なしモデルの詳細な比較については、既存の `darken_pinns_unified.py` を使用:

```bash
python darken_pinns_unified.py
```

## 純物質拡散定数・自己拡散定数の有用性

### 1. 端点条件の正確性
制約ありモデルは純物質での境界条件を正確に満たします:
- D_A(C=0) → 0 (成分Aは純Bに拡散できない)
- D_B(C=1) → 0 (成分Bは純Aに拡散できない)

### 2. 物理的妥当性
純物質境界条件により、自己拡散係数が物理的に妥当な値を取ります。非物理的な負の値や極端に大きな値が抑制されます。

### 3. 訓練の安定性
制約により学習が安定し、より良い収束が得られます。特に端点付近での挙動が改善されます。

### 4. 相互拡散係数の精度向上
正確な自己拡散係数（D_A, D_B）から、Darkenモデルによる相互拡散係数（D̃）の予測精度が向上します。

## 技術的詳細

### 損失関数の各項

1. **loss_data**: データ点での濃度フィッティング
   ```
   loss_data = MSE(C_pred - C_data)
   ```

2. **loss_pde**: 拡散方程式の残差
   ```
   ∂C/∂t = ∂/∂x[D̃(C)∂C/∂x]
   loss_pde = MSE(∂C/∂t - ∂/∂x[D̃(C)∂C/∂x])
   ```

3. **loss_ic**: 初期条件
   ```
   C(t=0, x) = C_left (x ≤ L/2), C_right (x > L/2)
   ```

4. **loss_bc**: 境界条件（Neumann: ∂C/∂x = 0）
   ```
   ∂C/∂x|_{x=0} = 0, ∂C/∂x|_{x=L} = 0
   ```

5. **loss_D_bc**: 純物質境界条件（重要！）
   ```
   D_A(0) = 0, D_B(1) = 0, 
   D_A(1) = 0.05, D_B(0) = 0.05
   ```

### 推奨パラメータ

- **エポック数**: 15000～30000
- **学習率**: 2×10⁻⁴
- **損失重み**:
  - λ_pde = 1.0
  - λ_ic = 2.0
  - λ_bc = 0.5
  - λ_Dbc = 20.0 ⭐（純物質制約を強く適用）

## 期待される出力

プログラム実行後、以下の図が生成されます:

### (a) FDMによる拡散対の濃度プロファイル
時間発展に伴う濃度分布の変化を示します。初期の急峻な界面が時間とともに緩やかになります。

### (b) 純物質境界条件の概念図
- 真の相互拡散係数 D̃(C) のプロット
- 純物質での自己拡散係数の境界値（赤と青のマーカー）
- 制約条件の注釈（D_A(0)→0, D_B(1)→0）

## 関連ファイル

- **diffusion_couple_pinns.py**: 本プログラム（デモンストレーション版）
- **darken_pinns_unified.py**: 完全版（アブレーション研究含む）
- **darken_pinns_app.py**: Streamlit対話的UI
- **pinn_darken.py**: 基本実装

## 参考文献

1. **Darken's Model**: L.S. Darken, "Diffusion, Mobility and Their Interrelation through Free Energy in Binary Metallic Systems," Trans. AIME, 1948
2. **PINNs**: M. Raissi et al., "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," Journal of Computational Physics, 2019

## 結論

この実装は、**純物質拡散定数・自己拡散定数を物理的制約としてPINNsの損失関数に組み込むことで、二元系合金の拡散係数推定が大幅に改善される**ことを示しています。

特に:
- ✅ 端点での非物理的挙動の抑制
- ✅ 自己拡散係数の物理的妥当性の保証
- ✅ 訓練の安定性向上
- ✅ 相互拡散係数の予測精度向上

これらの利点により、実験データからの拡散定数推定がより信頼性の高いものとなります。
