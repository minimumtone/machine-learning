import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Callable, Any
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="シンボリック回帰による物理法則発見",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 シンボリック回帰による物理法則発見")
st.markdown("""
このアプリは、観測された数値データから背後にある物理法則を記述する数式（シンボリックモデル）を自動的に発見するプログラムです。
AI-Feynmanなどの最先端研究の基礎概念を体験的に学習できます。

**対象となる物理法則:**
1. **運動エネルギー**: K = 0.5 × m × v²
2. **単振り子の周期**: T = 2π√(L/g)  
3. **万有引力**: F = G × (m₁×m₂)/r²
""")

np.random.seed(42)

def generate_kinetic_energy_data(n_samples=100):
    """運動エネルギーのデータを生成"""
    m = np.random.uniform(1, 10, n_samples)
    v = np.random.uniform(1, 20, n_samples)
    K = 0.5 * m * v**2 + np.random.normal(0, 0.1, n_samples)
    return pd.DataFrame({'m': m, 'v': v, 'K': K})

def generate_pendulum_data(n_samples=100):
    """単振り子の周期データを生成"""
    L = np.random.uniform(0.5, 5, n_samples)
    m = np.random.uniform(0.1, 2, n_samples)  # 無関係な変数
    g = np.random.uniform(9.8, 10.2, n_samples)
    T = 2 * np.pi * np.sqrt(L / g) + np.random.normal(0, 0.01, n_samples)
    return pd.DataFrame({'L': L, 'm': m, 'g': g, 'T': T})

def generate_gravity_data(n_samples=100):
    """万有引力のデータを生成"""
    G = 6.674e-11
    m1 = 1e10 * np.random.uniform(1, 10, n_samples)
    m2 = 1e10 * np.random.uniform(1, 10, n_samples)
    r = np.random.uniform(100, 1000, n_samples)
    F = G * (m1 * m2) / r**2 + np.random.normal(0, 1e-5, n_samples)
    return pd.DataFrame({'m1': m1, 'm2': m2, 'r': r, 'F': F})

def evaluate_formula(formula_func: Callable, params: List[float], X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, float]:
    """
    数式を評価し、最適な定数とMSEを返す関数
    
    :param formula_func: 定数と変数を引数にとる関数
    :param params: 定数の初期値
    :param X: 説明変数データ
    :param y: 目的変数データ
    :return: (最適化された定数, 最小MSE)
    """
    def objective(p):
        try:
            y_pred = formula_func(p, X)
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return 1e10
            return np.mean((y - y_pred)**2)
        except:
            return 1e10

    try:
        result = minimize(objective, params, method='Nelder-Mead')
        if result.success:
            return result.x, result.fun
        else:
            return np.array(params), 1e10
    except:
        return np.array(params), 1e10

def run_kinetic_energy_analysis():
    """運動エネルギーの法則発見"""
    st.header("🚀 課題1: 運動エネルギーの法則発見")
    st.markdown("**目標**: K = 0.5 × m × v² の発見")
    
    data = generate_kinetic_energy_data()
    X = data[['m', 'v']]
    y = data['K']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 生成データ")
        st.dataframe(data.head(10))
        
        st.subheader("📈 データ統計")
        st.write(data.describe())
    
    with col2:
        st.subheader("🔍 データ可視化")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].scatter(data['m'], data['K'], alpha=0.6)
        axes[0].set_xlabel('質量 (m)')
        axes[0].set_ylabel('運動エネルギー (K)')
        axes[0].set_title('質量 vs 運動エネルギー')
        
        axes[1].scatter(data['v'], data['K'], alpha=0.6)
        axes[1].set_xlabel('速度 (v)')
        axes[1].set_ylabel('運動エネルギー (K)')
        axes[1].set_title('速度 vs 運動エネルギー')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader("🧮 候補式の評価")
    
    formulas = {
        "c × m × v": (lambda p, x: p[0] * x['m'] * x['v'], [1.0]),
        "c × m × v²": (lambda p, x: p[0] * x['m'] * x['v']**2, [1.0]),
        "c × m² × v": (lambda p, x: p[0] * x['m']**2 * x['v'], [1.0]),
        "c₁×m + c₂×v": (lambda p, x: p[0]*x['m'] + p[1]*x['v'], [1.0, 1.0]),
        "c₁×m + c₂×v²": (lambda p, x: p[0]*x['m'] + p[1]*x['v']**2, [1.0, 1.0]),
        "c × m × v³": (lambda p, x: p[0] * x['m'] * x['v']**3, [1.0])
    }
    
    results = {}
    for name, (func, params) in formulas.items():
        best_params, min_mse = evaluate_formula(func, params, X, y)
        results[name] = {'mse': min_mse, 'params': best_params}
    
    results_df = pd.DataFrame([
        {
            '数式': name,
            'MSE': f"{results[name]['mse']:.6f}",
            '最適化定数': ', '.join([f"{p:.4f}" for p in results[name]['params']]),
            'スコア': results[name]['mse']
        }
        for name in formulas.keys()
    ]).sort_values('スコア')
    
    st.dataframe(results_df)
    
    best_formula = min(results, key=lambda k: results[k]['mse'])
    best_mse = results[best_formula]['mse']
    best_params = results[best_formula]['params']
    
    st.success(f"**最良の数式**: {best_formula}")
    st.info(f"**MSE**: {best_mse:.6f}")
    st.info(f"**最適化定数**: {', '.join([f'{p:.4f}' for p in best_params])}")
    
    if "c × m × v²" in best_formula and abs(best_params[0] - 0.5) < 0.1:
        st.success("✅ 正解！運動エネルギーの法則 K = 0.5 × m × v² を発見しました！")
    
    return results

def run_pendulum_analysis():
    """単振り子の周期の法則発見"""
    st.header("⏰ 課題2: 単振り子の周期の法則発見")
    st.markdown("**目標**: T = 2π√(L/g) の発見")
    
    data = generate_pendulum_data()
    X = data[['L', 'm', 'g']]
    y = data['T']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 生成データ")
        st.dataframe(data.head(10))
        
        st.subheader("📈 データ統計")
        st.write(data.describe())
    
    with col2:
        st.subheader("🔍 データ可視化")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].scatter(data['L'], data['T'], alpha=0.6)
        axes[0,0].set_xlabel('長さ (L)')
        axes[0,0].set_ylabel('周期 (T)')
        axes[0,0].set_title('長さ vs 周期')
        
        axes[0,1].scatter(data['g'], data['T'], alpha=0.6)
        axes[0,1].set_xlabel('重力加速度 (g)')
        axes[0,1].set_ylabel('周期 (T)')
        axes[0,1].set_title('重力加速度 vs 周期')
        
        axes[1,0].scatter(data['m'], data['T'], alpha=0.6)
        axes[1,0].set_xlabel('質量 (m)')
        axes[1,0].set_ylabel('周期 (T)')
        axes[1,0].set_title('質量 vs 周期')
        
        axes[1,1].scatter(data['L']/data['g'], data['T'], alpha=0.6)
        axes[1,1].set_xlabel('L/g')
        axes[1,1].set_ylabel('周期 (T)')
        axes[1,1].set_title('L/g vs 周期')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader("🧮 候補式の評価")
    
    formulas = {
        "c × √L": (lambda p, x: p[0] * np.sqrt(x['L']), [1.0]),
        "c × √g": (lambda p, x: p[0] * np.sqrt(x['g']), [1.0]),
        "c × √(L/g)": (lambda p, x: p[0] * np.sqrt(x['L'] / x['g']), [1.0]),
        "c × √(L×g)": (lambda p, x: p[0] * np.sqrt(x['L'] * x['g']), [1.0]),
        "c × L/m": (lambda p, x: p[0] * x['L'] / x['m'], [1.0]),
        "c × L/g": (lambda p, x: p[0] * x['L'] / x['g'], [1.0]),
        "c₁×L + c₂×g": (lambda p, x: p[0]*x['L'] + p[1]*x['g'], [1.0, 1.0]),
        "c × m × √(L/g)": (lambda p, x: p[0] * x['m'] * np.sqrt(x['L'] / x['g']), [1.0])
    }
    
    results = {}
    for name, (func, params) in formulas.items():
        best_params, min_mse = evaluate_formula(func, params, X, y)
        results[name] = {'mse': min_mse, 'params': best_params}
    
    results_df = pd.DataFrame([
        {
            '数式': name,
            'MSE': f"{results[name]['mse']:.6f}",
            '最適化定数': ', '.join([f"{p:.4f}" for p in results[name]['params']]),
            'スコア': results[name]['mse']
        }
        for name in formulas.keys()
    ]).sort_values('スコア')
    
    st.dataframe(results_df)
    
    best_formula = min(results, key=lambda k: results[k]['mse'])
    best_mse = results[best_formula]['mse']
    best_params = results[best_formula]['params']
    
    st.success(f"**最良の数式**: {best_formula}")
    st.info(f"**MSE**: {best_mse:.6f}")
    st.info(f"**最適化定数**: {', '.join([f'{p:.4f}' for p in best_params])}")
    
    if "√(L/g)" in best_formula and abs(best_params[0] - 2*np.pi) < 0.5:
        st.success("✅ 正解！単振り子の周期の法則 T = 2π√(L/g) を発見しました！")
    
    return results

def run_gravity_analysis():
    """万有引力の法則発見"""
    st.header("🌍 課題3: 万有引力の法則発見")
    st.markdown("**目標**: F = G × (m₁×m₂)/r² の発見")
    
    data = generate_gravity_data()
    X = data[['m1', 'm2', 'r']]
    y = data['F']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 生成データ")
        st.dataframe(data.head(10))
        
        st.subheader("📈 データ統計")
        st.write(data.describe())
    
    with col2:
        st.subheader("🔍 データ可視化")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].scatter(data['m1'], data['F'], alpha=0.6)
        axes[0,0].set_xlabel('質量1 (m1)')
        axes[0,0].set_ylabel('力 (F)')
        axes[0,0].set_title('質量1 vs 力')
        
        axes[0,1].scatter(data['m2'], data['F'], alpha=0.6)
        axes[0,1].set_xlabel('質量2 (m2)')
        axes[0,1].set_ylabel('力 (F)')
        axes[0,1].set_title('質量2 vs 力')
        
        axes[1,0].scatter(data['r'], data['F'], alpha=0.6)
        axes[1,0].set_xlabel('距離 (r)')
        axes[1,0].set_ylabel('力 (F)')
        axes[1,0].set_title('距離 vs 力')
        
        axes[1,1].scatter(1/data['r']**2, data['F'], alpha=0.6)
        axes[1,1].set_xlabel('1/r²')
        axes[1,1].set_ylabel('力 (F)')
        axes[1,1].set_title('1/r² vs 力')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader("🧮 候補式の評価")
    
    formulas = {
        "c × m1 × m2 / r": (lambda p, x: p[0] * x['m1'] * x['m2'] / x['r'], [1e-11]),
        "c × m1 × m2 / r²": (lambda p, x: p[0] * x['m1'] * x['m2'] / x['r']**2, [1e-11]),
        "c × (m1 + m2) / r²": (lambda p, x: p[0] * (x['m1'] + x['m2']) / x['r']**2, [1e-11]),
        "c × m1 × m2 / r³": (lambda p, x: p[0] * x['m1'] * x['m2'] / x['r']**3, [1e-11]),
        "c × m1 / r²": (lambda p, x: p[0] * x['m1'] / x['r']**2, [1e-11]),
        "c × m2 / r²": (lambda p, x: p[0] * x['m2'] / x['r']**2, [1e-11]),
        "c₁×m1×m2 + c₂/r²": (lambda p, x: p[0]*x['m1']*x['m2'] + p[1]/x['r']**2, [1e-21, 1e-5])
    }
    
    results = {}
    for name, (func, params) in formulas.items():
        best_params, min_mse = evaluate_formula(func, params, X, y)
        results[name] = {'mse': min_mse, 'params': best_params}
    
    results_df = pd.DataFrame([
        {
            '数式': name,
            'MSE': f"{results[name]['mse']:.2e}",
            '最適化定数': ', '.join([f"{p:.2e}" for p in results[name]['params']]),
            'スコア': results[name]['mse']
        }
        for name in formulas.keys()
    ]).sort_values('スコア')
    
    st.dataframe(results_df)
    
    best_formula = min(results, key=lambda k: results[k]['mse'])
    best_mse = results[best_formula]['mse']
    best_params = results[best_formula]['params']
    
    st.success(f"**最良の数式**: {best_formula}")
    st.info(f"**MSE**: {best_mse:.2e}")
    st.info(f"**最適化定数**: {', '.join([f'{p:.2e}' for p in best_params])}")
    
    G_theoretical = 6.674e-11
    if "m1 × m2 / r²" in best_formula and abs(best_params[0] - G_theoretical) / G_theoretical < 0.1:
        st.success("✅ 正解！万有引力の法則 F = G × (m₁×m₂)/r² を発見しました！")
        st.info(f"理論値 G = {G_theoretical:.3e}, 発見値 = {best_params[0]:.3e}")
    
    return results

def main():
    st.sidebar.header("🔬 解析選択")
    analysis_type = st.sidebar.selectbox(
        "実行する解析を選択:",
        ["概要", "運動エネルギー", "単振り子の周期", "万有引力", "全解析実行"]
    )
    
    if analysis_type == "概要":
        st.header("📚 シンボリック回帰とは")
        st.markdown("""
        **シンボリック回帰**は、数値データから数式を自動的に発見する機械学習手法です。
        従来の回帰分析とは異なり、予め決められた関数形ではなく、データに最も適合する数式の構造自体を探索します。
        
        
        1. **段階的探索アプローチ**: シンプルな式から複雑な式へと段階的に探索
        2. **定数最適化**: 各候補式に対して最適な定数を自動計算
        3. **物理法則の発見**: 実際の物理現象から法則を導出
        4. **教育的価値**: AI-Feynmanの基本概念を体験学習
        
        
        1. **候補式生成**: 基本演算子と変数の組み合わせで式を生成
        2. **定数最適化**: SciPyの最適化アルゴリズムで定数を調整
        3. **評価**: MSE（平均二乗誤差）で式の適合度を評価
        4. **選択**: 最も低いMSEを持つ式を最良として選択
        
        
        - **運動エネルギー**: K = 0.5 × m × v²
        - **単振り子の周期**: T = 2π√(L/g)
        - **万有引力**: F = G × (m₁×m₂)/r²
        """)
        
        st.header("🚀 使用方法")
        st.markdown("""
        1. 左サイドバーから解析したい物理法則を選択
        2. 自動生成されたデータを確認
        3. 候補式の評価結果を確認
        4. 最良の数式が正解と一致するかを確認
        """)
    
    elif analysis_type == "運動エネルギー":
        run_kinetic_energy_analysis()
    
    elif analysis_type == "単振り子の周期":
        run_pendulum_analysis()
    
    elif analysis_type == "万有引力":
        run_gravity_analysis()
    
    elif analysis_type == "全解析実行":
        st.header("🔬 全解析実行")
        st.markdown("3つの物理法則すべてについて解析を実行します。")
        
        with st.expander("🚀 運動エネルギー解析", expanded=True):
            kinetic_results = run_kinetic_energy_analysis()
        
        with st.expander("⏰ 単振り子の周期解析", expanded=True):
            pendulum_results = run_pendulum_analysis()
        
        with st.expander("🌍 万有引力解析", expanded=True):
            gravity_results = run_gravity_analysis()
        
        st.header("📊 総合結果")
        st.success("✅ 全ての解析が完了しました！")
        st.markdown("""
        各解析で最も適合度の高い数式が発見されました。
        これらの結果から、データに隠された物理法則を自動的に発見できることが確認できます。
        """)

if __name__ == "__main__":
    main()
