# 非晶質形成モデルの妥当性検証マニュアル

## 概要

このパッケージは、**CALPHAD熱力学**と**Davis-Uhlmannモデル**を組み合わせて、液体の急冷時にガラス（非晶質）が形成されるか、結晶化するかを予測するモデルの検証を行うためのツールです。

## 目次

1. [インストールと起動](#1-インストールと起動)
2. [理論的背景](#2-理論的背景)
3. [モジュール構成](#3-モジュール構成)
4. [検証ステップ](#4-検証ステップ)
5. [使用例](#5-使用例)
6. [材料データベース](#6-材料データベース)
7. [トラブルシューティング](#7-トラブルシューティング)
8. [参考文献](#8-参考文献)

---

## 1. インストールと起動

### 必要なパッケージ

```bash
pip install numpy scipy matplotlib streamlit pandas
```

### Streamlitアプリの起動

```bash
cd /path/to/machine-learning
streamlit run amorphous_formation_app.py
```

### Pythonモジュールとしての使用

```python
from amorphous_formation import (
    MaterialsDatabase,
    CALPHADThermodynamics,
    DoolittleViscosity,
    DavisUhlmannModel,
    SensitivityAnalysis,
    AmorphousVisualization
)
```

---

## 2. 理論的背景

### 2.1 なぜ液体を急冷するとガラスになるのか？

液体を冷却すると、通常は結晶化が起こります。結晶化には以下の2つのプロセスが必要です：

1. **核生成（Nucleation）**: 結晶の「種」となる微小な結晶核の形成
2. **成長（Growth）**: 核から結晶が成長

十分に速く冷却すると、これらのプロセスが完了する前に液体が凍結し、**ガラス（非晶質）**が形成されます。

### 2.2 CALPHAD熱力学

結晶化の**駆動力**は、液体と固体のギブスエネルギー差で表されます：

$$\Delta G_m = G^L - G^S$$

**Thompson-Spaepen近似**を用いると：

$$\Delta G_m = \Delta H_f \cdot \frac{T_m - T}{T_m} \cdot \frac{2T}{T_m + T}$$

ここで：
- $\Delta H_f$: 融解熱 [J/mol]
- $T_m$: 融点 [K]
- $T$: 温度 [K]

**重要な性質：**
- $T = T_m$ で $\Delta G_m = 0$（平衡状態）
- $T < T_m$ で $\Delta G_m > 0$（結晶化の駆動力が存在）

### 2.3 Doolittle粘度モデル

液体の粘度は温度とともに急激に変化します。**Vogel-Fulcher-Tammann (VFT) 方程式**：

$$\eta(T) = \eta_0 \cdot \exp\left(\frac{D^* \cdot T_0}{T - T_0}\right)$$

ここで：
- $\eta_0$: 前指数因子 [Pa·s]
- $D^*$: 脆弱性パラメータ（強度パラメータ）
- $T_0$: VFT温度 [K]

**ガラス転移の定義：**
- $T_g$ で $\eta \approx 10^{12}$ Pa·s

**脆弱性指数 $m$：**

$$m = \frac{d(\log \eta)}{d(T_g/T)}\bigg|_{T=T_g}$$

- 強い液体（SiO₂）: $m \approx 16-20$
- 弱い液体（金属ガラス）: $m \approx 50-100$

### 2.4 Davis-Uhlmannモデル

TTT曲線は**核生成速度 $I$** と**成長速度 $U$** から計算されます。

**核生成速度（古典核生成理論）：**

$$I = I_0 \cdot \exp\left(-\frac{\Delta G^*}{k_B T}\right)$$

**核生成障壁：**

$$\Delta G^* = \frac{16\pi\sigma^3}{3(\Delta G_v)^2}$$

ここで $\sigma$ は固液界面エネルギー [J/m²]。

**成長速度（Wilson-Frenkelモデル）：**

$$U = \frac{D}{a} \cdot \left[1 - \exp\left(-\frac{\Delta G_m}{RT}\right)\right]$$

**JMAK動力学から結晶化時間：**

$$t = \left[\frac{3 \ln(1/(1-X))}{\pi I U^3}\right]^{1/4}$$

### 2.5 臨界冷却速度

TTT曲線の「ノーズ」（最短結晶化時間）から：

$$R_c = \frac{T_m - T_n}{t_n}$$

---

## 3. モジュール構成

### 3.1 materials_database.py

既知のガラス形成材料のパラメータを格納したデータベース。

```python
from amorphous_formation import MaterialsDatabase, Material

db = MaterialsDatabase()

# 利用可能な材料一覧
print(db.list_materials())

# 特定の材料を取得
vitreloy = db.get_material("Zr41Ti14Cu12Ni10Be23")
print(f"T_m = {vitreloy.T_m} K")
print(f"R_c (実験) = {vitreloy.R_c_exp} K/s")
```

**含まれる材料：**
- Pd₈₂Si₁₈
- Pd₄₀Ni₄₀P₂₀
- Zr₄₁.₂Ti₁₃.₈Cu₁₂.₅Ni₁₀Be₂₂.₅ (Vitreloy 1)
- Zr₅₅Cu₃₀Al₁₀Ni₅
- Cu₄₇Ti₃₄Zr₁₁Ni₈
- Fe₈₀B₂₀
- Au₇₇Ge₁₄Si₉
- Mg₆₅Cu₂₅Y₁₀
- SiO₂
- B₂O₃

### 3.2 calphad_thermodynamics.py

ギブスエネルギー差（駆動力）の計算。

```python
from amorphous_formation import CALPHADThermodynamics

thermo = CALPHADThermodynamics(
    T_m=937.0,      # 融点 [K]
    delta_H_f=8200.0,  # 融解熱 [J/mol]
    T_g=625.0       # ガラス転移点 [K]
)

# 特定温度でのΔG計算
T = 750.0
delta_G = thermo.delta_G(T, method="thompson_spaepen")
print(f"ΔG({T} K) = {delta_G:.1f} J/mol")

# 検証レポート
print(thermo.get_validation_report())
```

**利用可能な計算方法：**
- `"turnbull"`: 単純な線形近似
- `"thompson_spaepen"`: 改良近似（推奨）
- `"hoffman"`: Hoffman近似
- `"full"`: ΔCpを含む完全計算

### 3.3 doolittle_viscosity.py

温度依存粘度の計算。

```python
from amorphous_formation import DoolittleViscosity

visc = DoolittleViscosity(
    T_m=937.0,
    T_g=625.0,
    eta_0=1e-5,
    D_star=18.5
)

# 粘度計算
eta = visc.viscosity(700.0)
print(f"η(700 K) = {eta:.2e} Pa·s")

# 脆弱性指数
m = visc.fragility_index_m()
print(f"脆弱性指数 m = {m:.1f}")
print(visc.classify_fragility())

# Angellプロットデータ
Tg_T, log_eta = visc.angell_plot_data()
```

### 3.4 davis_uhlmann_model.py

TTT曲線と臨界冷却速度の計算。

```python
from amorphous_formation import DavisUhlmannModel

model = DavisUhlmannModel(
    T_m=937.0,
    T_g=625.0,
    delta_H_f=8200.0,
    sigma=0.08,      # 界面エネルギー [J/m²]
    V_m=1.1e-5,      # モル体積 [m³/mol]
    eta_0=1e-5,
    D_star=18.5
)

# TTT曲線計算
ttt = model.calculate_ttt_curve()
print(f"ノーズ温度 T_n = {ttt.nose_temperature:.1f} K")
print(f"ノーズ時間 t_n = {ttt.nose_time:.2e} s")
print(f"臨界冷却速度 R_c = {ttt.critical_cooling_rate:.2e} K/s")

# 検証レポート
print(model.get_validation_report())
```

### 3.5 sensitivity_analysis.py

パラメータ感度解析。

```python
from amorphous_formation import SensitivityAnalysis

analysis = SensitivityAnalysis(
    T_m=937.0,
    T_g=625.0,
    delta_H_f=8200.0,
    sigma=0.08,
    V_m=1.1e-5,
    eta_0=1e-5,
    D_star=18.5
)

# σの感度解析（±10%変動）
sigma_result = analysis.analyze_sigma_sensitivity(variation_percent=10.0)

# 感度係数
coeffs = analysis.calculate_sensitivity_coefficients()
print(f"S_σ = {coeffs['σ']:.2f}")

# 感度解析テーブル
print(analysis.get_sensitivity_table())

# 実験値との比較
print(analysis.compare_with_experiment(R_c_exp=1.0, material_name="Vitreloy 1"))
```

### 3.6 visualization.py

すべての必要なプロットを生成。

```python
from amorphous_formation import AmorphousVisualization

viz = AmorphousVisualization()

# 個別プロット
fig1, ax1 = viz.plot_calphad_driving_force(thermo)
fig2, ax2 = viz.plot_viscosity_temperature(visc)
fig3, ax3 = viz.plot_angell(visc)
fig4, ax4 = viz.plot_ttt_curve(model)
fig5, ax5 = viz.plot_sensitivity_analysis(sigma_result)

# 完全な検証図（4パネル）
fig = viz.plot_complete_validation(thermo, visc, model, sigma_result)

# すべての図を保存
saved_files = viz.save_all_figures(
    thermo, visc, model, sigma_result,
    output_dir="./figures",
    prefix="vitreloy1"
)
```

---

## 4. 検証ステップ

### ステップ1: 熱力学データのチェック

**指示：** $T_m$から$T_g$まで、$\Delta G_m$をプロットする。

**合格基準：**
1. $T = T_m$ で $\Delta G_m = 0$
2. 温度低下とともに$\Delta G_m$が増加
3. $\Delta S_f = \Delta H_f / T_m$ が文献値に近い

```python
# 検証コード
result = thermo.calculate_all()
assert abs(thermo.delta_G(thermo.T_m)) < 1e-6, "ΔG(T_m) ≠ 0"
assert thermo.verify_at_melting_point(), "融点での検証失敗"
```

### ステップ2: 粘度モデルのチェック

**指示：** $\eta(T)$をログスケールでプロットし、Angellプロットを作成する。

**合格基準：**
1. $\eta(T_g) \approx 10^{12}$ Pa·s
2. $\eta(T_m) \approx 10^{-3}$ Pa·s
3. Angellプロットで強い/弱い液体の特徴を確認

```python
# 検証コード
assert visc.verify_at_Tg(), "η(T_g)の検証失敗"
assert visc.verify_at_Tm(), "η(T_m)の検証失敗"
```

### ステップ3: TTT曲線のノーズ特定

**指示：** TTT曲線を描画し、ノーズを特定する。

**合格基準：**
1. C字型の曲線
2. $T_n \approx 0.7-0.8 \cdot T_m$

```python
# 検証コード
pass_nose, msg = model.verify_nose_position()
assert pass_nose, msg
```

### ステップ4: パラメータ感度解析

**指示：** $\sigma$を±10%変化させ、$R_c$の変化を記録する。

```python
# 感度解析
result = analysis.analyze_sigma_sensitivity(variation_percent=10.0)
print(f"σを+10%変化: R_cは{10**result.log_R_c_change[-1]:.1f}倍に変化")
```

---

## 5. 使用例

### 5.1 Vitreloy 1の完全な検証

```python
from amorphous_formation import (
    MaterialsDatabase,
    CALPHADThermodynamics,
    DoolittleViscosity,
    DavisUhlmannModel,
    SensitivityAnalysis,
    AmorphousVisualization
)

# 材料データベースから取得
db = MaterialsDatabase()
mat = db.get_material("Zr41Ti14Cu12Ni10Be23")

# 各モデルの初期化
thermo = CALPHADThermodynamics(T_m=mat.T_m, delta_H_f=mat.delta_H_f, T_g=mat.T_g)
visc = DoolittleViscosity(T_m=mat.T_m, T_g=mat.T_g, eta_0=mat.eta_0, D_star=mat.D_star)
model = DavisUhlmannModel(
    T_m=mat.T_m, T_g=mat.T_g, delta_H_f=mat.delta_H_f,
    sigma=mat.sigma, V_m=mat.V_m, eta_0=mat.eta_0, D_star=mat.D_star
)
analysis = SensitivityAnalysis(
    T_m=mat.T_m, T_g=mat.T_g, delta_H_f=mat.delta_H_f,
    sigma=mat.sigma, V_m=mat.V_m, eta_0=mat.eta_0, D_star=mat.D_star
)

# 検証レポート出力
print(thermo.get_validation_report())
print(visc.get_validation_report())
print(model.get_validation_report())

# 実験値との比較
print(analysis.compare_with_experiment(R_c_exp=mat.R_c_exp, material_name=mat.name))

# 可視化
viz = AmorphousVisualization()
fig = viz.plot_complete_validation(
    thermo, visc, model,
    analysis.analyze_sigma_sensitivity(),
    material_name=mat.composition
)
fig.savefig("vitreloy1_validation.png", dpi=150)
```

### 5.2 カスタム材料の解析

```python
# カスタムパラメータで解析
thermo = CALPHADThermodynamics(
    T_m=1000.0,
    delta_H_f=10000.0,
    T_g=600.0
)

visc = DoolittleViscosity(
    T_m=1000.0,
    T_g=600.0,
    eta_0=1e-5,
    D_star=15.0
)

model = DavisUhlmannModel(
    T_m=1000.0,
    T_g=600.0,
    delta_H_f=10000.0,
    sigma=0.1,
    V_m=1e-5
)

R_c = model.critical_cooling_rate()
print(f"臨界冷却速度: {R_c:.2e} K/s")
```

---

## 6. 材料データベース

| 材料 | $T_m$ [K] | $T_g$ [K] | $T_{rg}$ | $R_c$ (実験) [K/s] |
|------|-----------|-----------|----------|-------------------|
| Pd₈₂Si₁₈ | 1071 | 633 | 0.591 | 10³ |
| Pd₄₀Ni₄₀P₂₀ | 884 | 580 | 0.656 | 10⁰ |
| Vitreloy 1 | 937 | 625 | 0.667 | 10⁰ |
| Zr₅₅Cu₃₀Al₁₀Ni₅ | 1100 | 683 | 0.621 | 10¹ |
| Cu₄₇Ti₃₄Zr₁₁Ni₈ | 1150 | 698 | 0.607 | 2.5×10² |
| Fe₈₀B₂₀ | 1448 | 720 | 0.497 | 10⁵ |
| Au₇₇Ge₁₄Si₉ | 629 | 295 | 0.469 | 10⁶ |
| Mg₆₅Cu₂₅Y₁₀ | 730 | 420 | 0.575 | 5×10¹ |
| SiO₂ | 1996 | 1473 | 0.738 | 10⁻⁴ |
| B₂O₃ | 723 | 520 | 0.719 | 10⁻² |

**$T_{rg} = T_g / T_m$** は還元ガラス転移温度で、ガラス形成能の指標です。
$T_{rg} > 0.6$ の材料は一般にガラス形成能が高いとされます。

---

## 7. トラブルシューティング

### 7.1 TTT曲線がC字型にならない

**原因と対策：**

1. **界面エネルギー$\sigma$が不適切**
   - $\sigma$が大きすぎる → ノーズが$T_g$に近すぎる
   - $\sigma$が小さすぎる → ノーズが$T_m$に近すぎる
   - Turnbullの関係式で推定: $\sigma \approx 0.45 \cdot \Delta H_f / (N_A^{1/3} \cdot V_m^{2/3})$

2. **単位のミスマッチ**
   - $R$（ガス定数）vs $k_B$（ボルツマン定数）を確認
   - エネルギー: J/mol vs J/atom
   - モル体積$V_m$の単位を確認

### 7.2 計算値と実験値が大きくずれる

**計算値が過大評価の場合（$R_c^{calc} > R_c^{exp}$）：**
- $\sigma$を増加させる
- $D^*$を減少させる

**計算値が過小評価の場合（$R_c^{calc} < R_c^{exp}$）：**
- $\sigma$を減少させる
- $D^*$を増加させる

### 7.3 粘度が期待値と異なる

1. **$\eta(T_g) \neq 10^{12}$ Pa·s**
   - $D^*$を調整して$\eta(T_g) = 10^{12}$となるようにする
   
2. **$\eta(T_m)$が高すぎる/低すぎる**
   - $\eta_0$を調整

---

## 8. 参考文献

1. Turnbull, D. (1969). "Under what conditions can a glass be formed?" *Contemp. Phys.* 10, 473.

2. Uhlmann, D.R. (1972). "A kinetic treatment of glass formation." *J. Non-Cryst. Solids* 7, 337.

3. Davies, H.A. (1976). "The formation of metallic glasses." *Phys. Chem. Glasses* 17, 159.

4. Angell, C.A. (1995). "Formation of glasses from liquids and biopolymers." *Science* 267, 1924.

5. Inoue, A. (2000). "Stabilization of metallic supercooled liquid and bulk amorphous alloys." *Acta Mater.* 48, 279.

6. Johnson, W.L. (1999). "Bulk glass-forming metallic alloys: Science and technology." *MRS Bull.* 24, 42.

7. Lu, Z.P. & Liu, C.T. (2002). "A new glass-forming ability criterion for bulk metallic glasses." *Acta Mater.* 50, 3501.

8. Thompson, C.V. & Spaepen, F. (1979). "On the approximation of the free energy change on crystallization." *Acta Metall.* 27, 1855.

---

## 付録A: 物理定数

| 定数 | 記号 | 値 | 単位 |
|------|------|-----|------|
| ガス定数 | $R$ | 8.314 | J/(mol·K) |
| ボルツマン定数 | $k_B$ | 1.38×10⁻²³ | J/K |
| アボガドロ数 | $N_A$ | 6.022×10²³ | mol⁻¹ |

## 付録B: 典型的なパラメータ範囲

| パラメータ | 金属ガラス | 酸化物ガラス |
|-----------|-----------|-------------|
| $\sigma$ [J/m²] | 0.05-0.15 | 0.1-0.3 |
| $D^*$ | 10-30 | 5-100 |
| $\eta_0$ [Pa·s] | 10⁻⁵-10⁻⁴ | 10⁻⁷-10⁻⁵ |
| $T_{rg}$ | 0.5-0.7 | 0.6-0.8 |

---

*このマニュアルは非晶質形成モデルの妥当性検証実習用に作成されました。*
