# シンボリック回帰による物理法則発見プロジェクト

## 📋 プロジェクト概要

このプロジェクトは、「Discovering Symbolic Models from Deep Learning with Inductive Biases」や「AI-Feynman」のコンセプトに基づいて、観測データから物理法則を自動発見するシンボリック回帰システムを実装しています。

### 🎯 発見対象の物理法則

#### 基本物理法則
1. **運動エネルギー**: `K = 0.5 × m × v²`
2. **単振り子の周期**: `T = 2π√(L/g)`
3. **万有引力**: `F = G × (m₁×m₂)/r²`

#### 材料工学法則
4. **Wiedemann-Franz法則**: `κ = L₀ × σ × T`
5. **ホール効果**: `R_H = 1/(n × e)`
6. **Hall-Petch関係**: `σ_y = σ₀ + k/√d`

#### 偏微分方程式 (PDE)
7. **熱伝導方程式**: `∂u/∂t = α × ∂²u/∂x²`

## 🚀 実行方法

### 1. 環境セットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt
```

### 2. データ生成

```bash
# 物理法則データの生成
python data_generation.py
```

このコマンドにより以下のCSVファイルが生成されます：
- `kinetic_energy.csv` - 運動エネルギーデータ
- `pendulum.csv` - 単振り子の周期データ
- `gravity.csv` - 万有引力データ

### 3. メインアプリケーションの実行

```bash
# 基本的なシンボリック回帰アプリ
streamlit run symbolic_regression.py
```

ブラウザで `http://localhost:8501` にアクセスして、各物理法則の発見プロセスを体験できます。

### 4. 発展課題（自動式生成）の実行

```bash
# 高度なシンボリック回帰アプリ
streamlit run advanced_symbolic_regression.py --server.port 8502
```

ブラウザで `http://localhost:8502` にアクセスして、自動式生成と複雑度ペナルティ機能を試すことができます。

### PDE発見システム

```bash
# PDE発見用データ生成
python pde_data_generation.py

# PDE発見アプリ
streamlit run pde_discovery.py --server.port 8504
```

ブラウザで `http://localhost:8504` にアクセスして、FDMによる熱伝導方程式の発見プロセスを体験できます。

## 📁 ファイル構成

```
├── symbolic_regression.py          # メインのシンボリック回帰アプリ
├── advanced_symbolic_regression.py # 発展課題：自動式生成
├── data_generation.py              # データ生成スクリプト
├── pde_discovery.py                # PDE発見システム
├── pde_data_generation.py          # PDE用データ生成スクリプト
├── materials_symbolic_regression.py # 材料工学シンボリック回帰
├── materials_data_generation.py    # 材料工学データ生成
├── kinetic_energy.csv              # 運動エネルギーデータ
├── pendulum.csv                    # 単振り子データ
├── gravity.csv                     # 万有引力データ
├── requirements.txt                # 依存関係（sympy追加済み）
├── SYMBOLIC_REGRESSION_README.md   # このファイル
├── MATERIALS_ENGINEERING_README.md # 材料工学拡張ドキュメント
└── PDE_DISCOVERY_README.md         # PDE発見システムドキュメント
```

## 🔬 実装の詳細

### コアアルゴリズム

1. **段階的探索アプローチ**
   - 候補式の手動定義から開始
   - 各式に対して定数最適化を実行
   - MSE（平均二乗誤差）による評価

2. **定数最適化**
   - `scipy.optimize.minimize`を使用
   - Nelder-Mead法による最適化
   - 物理的に妥当な初期値設定

3. **評価フレームワーク**
   ```python
   def evaluate_formula(formula_func, params, X, y):
       def objective(p):
           y_pred = formula_func(p, X)
           return np.mean((y - y_pred)**2)
       
       result = minimize(objective, params, method='Nelder-Mead')
       return result.x, result.fun
   ```

### 発見された結果

#### 1. 運動エネルギー
- **発見式**: `c × m × v²`
- **最適定数**: `c ≈ 0.5000`
- **MSE**: `≈ 0.01`
- **理論値との一致**: ✅ 完全一致

#### 2. 単振り子の周期
- **発見式**: `c × √(L/g)`
- **最適定数**: `c ≈ 6.2852`
- **MSE**: `≈ 0.0001`
- **理論値との一致**: ✅ 2π ≈ 6.28と一致

#### 3. 万有引力
- **発見式**: `c × (m₁×m₂)/r²`
- **最適定数**: `c ≈ 6.67e-11`
- **MSE**: `≈ 1.8e-04`
- **理論値との一致**: ✅ G = 6.674e-11と一致

## 🧪 発展課題の特徴

### 自動式生成
- **SymPy**を使用した数式の自動構築
- 複雑度レベル（1, 2, 3...）による段階的探索
- 基本演算子：`[add, mul, div, pow, sqrt]`

### 複雑度ペナルティ
```python
Score = MSE + α × Complexity
```
- **α**: 複雑度ペナルティ係数
- **Complexity**: 式のノード数
- **オッカムの剃刀**: 同精度なら単純な式を選択

### SymbolicRegressor クラス
```python
class SymbolicRegressor:
    def generate_expressions(self, variables, max_complexity)
    def evaluate_expression(self, expr, X, y)
    def fit(self, X, y, max_complexity=3, alpha=0.01)
```

## 📊 データ特性

### 合成データの特徴
- **ノイズ付加**: 現実的なデータ模擬
- **変数範囲**: 物理的に妥当な値域
- **サンプル数**: 各データセット100サンプル
- **再現性**: `np.random.seed(42)`で固定

### 無関係変数の検出
- 単振り子データに質量`m`を含める
- 正しい式では`m`が除外されることを確認
- 物理的直感との一致を検証

## 🎓 教育的価値

### 学習目標
1. **シンボリック回帰の理解**
2. **最適化アルゴリズムの実践**
3. **物理法則の数学的表現**
4. **データサイエンスパイプライン**

### 段階的学習
1. **基礎**: 手動候補式による探索
2. **応用**: 自動式生成システム
3. **発展**: 複雑度制御とモデル選択

## 🔧 技術スタック

- **Python 3.8+**
- **NumPy**: 数値計算
- **Pandas**: データ操作
- **SymPy**: シンボリック計算
- **SciPy**: 最適化
- **Matplotlib**: 可視化
- **Streamlit**: Webアプリケーション

## 🚨 注意事項

### 制限事項
1. **計算時間**: 複雑度が高いと指数的に増加
2. **局所最適解**: 初期値依存の可能性
3. **式の表現力**: 基本演算子に限定

### 改善案
1. **遺伝的アルゴリズム**の導入
2. **並列処理**による高速化
3. **物理制約**の組み込み
4. **次元解析**の活用

## 📈 実行例

### 基本的な使用例
```python
# データ読み込み
data = pd.read_csv('kinetic_energy.csv')
X = data[['m', 'v']]
y = data['K']

# 候補式の評価
formula = lambda p, x: p[0] * x['m'] * x['v']**2
params, mse = evaluate_formula(formula, [1.0], X, y)
print(f"最適定数: {params[0]:.4f}, MSE: {mse:.6f}")
```

### 発展課題の使用例
```python
# 自動シンボリック回帰
regressor = SymbolicRegressor()
best_expr, best_score = regressor.fit(X, y, max_complexity=3, alpha=0.01)
print(f"発見式: {best_expr}")
print(f"スコア: {best_score:.6f}")
```

## 🔬 PDE発見システム

### 実装された機能
- **FDM熱伝導ソルバー**: 安定性条件チェック付き数値解法
- **数値微分計算**: 時間・空間偏微分の高精度計算
- **PDE構造発見**: シンボリック回帰による偏微分方程式の逆算
- **対話的可視化**: 温度場と偏微分項の時空間分布表示

### 発見結果例
- **熱伝導方程式**: `∂u/∂t = α × ∂²u/∂x²`
- **係数精度**: 理論値との相対誤差 < 5%
- **構造識別**: 正確なPDE形式の自動発見

### 技術的特徴
- **安定性条件**: r = α×dt/dx² ≤ 0.5 の自動チェック
- **境界条件**: ディリクレ境界条件 (u = 0 at boundaries)
- **初期条件**: ガウシアン分布による温度分布
- **数値微分**: 中央差分による高精度偏微分計算

## 🎉 成果

このプロジェクトにより、以下を達成しました：

1. ✅ **7つの物理法則の完全発見** (基本物理3 + 材料工学3 + PDE1)
2. ✅ **段階的探索フレームワークの構築**
3. ✅ **自動式生成システムの実装**
4. ✅ **複雑度ペナルティの導入**
5. ✅ **材料工学への拡張**
6. ✅ **偏微分方程式発見システムの実装**
7. ✅ **教育的Webアプリケーションの開発**

シンボリック回帰の基礎から発展まで、さらに偏微分方程式の発見まで、体系的に学習できる包括的なシステムが完成しました。

## 🚀 今後の拡張可能性

### 追加可能なPDE
- **波動方程式**: `∂²u/∂t² = c² × ∂²u/∂x²`
- **移流拡散方程式**: `∂u/∂t + v × ∂u/∂x = α × ∂²u/∂x²`
- **反応拡散方程式**: `∂u/∂t = α × ∂²u/∂x² + f(u)`
- **ナビエ・ストークス方程式**: 流体力学への応用

### 高度な機能
- **2D/3D偏微分方程式**の発見
- **適応格子**による動的格子細分化
- **機械学習統合**: PINNsとの融合
- **実験データ対応**: 実際の観測データからのPDE発見
- **多物理現象**: 連成問題への拡張
