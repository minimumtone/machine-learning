"""
材料工学シンボリック回帰アプリケーション
材料科学の物理法則をデータから発見するためのStreamlitアプリ
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import sympy as sp

plt.rcParams['font.family'] = 'DejaVu Sans'

def generate_thermal_conductivity_data(n_samples=100):
    """
    熱伝導率データを生成
    Wiedemann-Franz法則: κ = L₀ × σ × T (L₀: ローレンツ数)
    """
    np.random.seed(42)
    sigma = np.random.uniform(1e6, 1e8, n_samples)  # 電気伝導率 [S/m]
    T = np.random.uniform(200, 400, n_samples)      # 温度 [K]
    L0 = 2.44e-8  # ローレンツ数 [W·Ω/K²]
    
    kappa = L0 * sigma * T + np.random.normal(0, 0.1, n_samples)
    
    return pd.DataFrame({
        'sigma': sigma,
        'T': T,
        'kappa': kappa
    })

def generate_hall_effect_data(n_samples=100):
    """
    ホール効果データを生成
    ホール係数: R_H = 1/(n × e) (n: キャリア密度, e: 電子電荷)
    """
    np.random.seed(43)
    n = np.random.uniform(1e20, 1e24, n_samples)  # キャリア密度 [m⁻³]
    e = 1.602e-19  # 電子電荷 [C]
    
    R_H = 1 / (n * e) + np.random.normal(0, 1e-8, n_samples)
    
    return pd.DataFrame({
        'n': n,
        'R_H': R_H
    })

def generate_youngs_modulus_data(n_samples=100):
    """
    ヤング率データを生成
    Hall-Petch関係: σ_y = σ₀ + k/√d (d: 結晶粒径)
    """
    np.random.seed(44)
    d = np.random.uniform(1e-6, 1e-4, n_samples)  # 結晶粒径 [m]
    sigma_0 = 50e6  # 基準応力 [Pa]
    k = 0.5e-3      # Hall-Petch定数
    
    sigma_y = sigma_0 + k / np.sqrt(d) + np.random.normal(0, 1e6, n_samples)
    
    return pd.DataFrame({
        'd': d,
        'sigma_y': sigma_y
    })

def evaluate_formula(formula_func, params, X, y):
    """
    数式を評価し、最適な定数とMSEを返す
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
            return None, np.inf
    except:
        return None, np.inf

def analyze_thermal_conductivity(data):
    """熱伝導率データの解析"""
    X = data[['sigma', 'T']]
    y = data['kappa']
    
    formulas = {
        "c * sigma": (lambda p, x: p[0] * x['sigma'], [1e-8]),
        "c * T": (lambda p, x: p[0] * x['T'], [1e-3]),
        "c * sigma * T": (lambda p, x: p[0] * x['sigma'] * x['T'], [1e-8]),  # 正解
        "c1 * sigma + c2 * T": (lambda p, x: p[0] * x['sigma'] + p[1] * x['T'], [1e-8, 1e-3]),
        "c * sigma / T": (lambda p, x: p[0] * x['sigma'] / x['T'], [1e-5]),
    }
    
    results = {}
    for name, (func, params) in formulas.items():
        best_params, min_mse = evaluate_formula(func, params, X, y)
        results[name] = {'mse': min_mse, 'params': best_params}
    
    return results

def analyze_hall_effect(data):
    """ホール効果データの解析"""
    X = data[['n']]
    y = data['R_H']
    
    formulas = {
        "c / n": (lambda p, x: p[0] / x['n'], [1e-1]),  # 正解
        "c * n": (lambda p, x: p[0] * x['n'], [1e-28]),
        "c / sqrt(n)": (lambda p, x: p[0] / np.sqrt(x['n']), [1e-11]),
        "c / n**2": (lambda p, x: p[0] / x['n']**2, [1e15]),
    }
    
    results = {}
    for name, (func, params) in formulas.items():
        best_params, min_mse = evaluate_formula(func, params, X, y)
        results[name] = {'mse': min_mse, 'params': best_params}
    
    return results

def analyze_youngs_modulus(data):
    """ヤング率データの解析"""
    X = data[['d']]
    y = data['sigma_y']
    
    formulas = {
        "c1 + c2 / sqrt(d)": (lambda p, x: p[0] + p[1] / np.sqrt(x['d']), [50e6, 0.5e-3]),  # 正解
        "c * d": (lambda p, x: p[0] * x['d'], [1e12]),
        "c / d": (lambda p, x: p[0] / x['d'], [1e-3]),
        "c1 + c2 * d": (lambda p, x: p[0] + p[1] * x['d'], [50e6, 1e12]),
        "c * sqrt(d)": (lambda p, x: p[0] * np.sqrt(x['d']), [1e9]),
    }
    
    results = {}
    for name, (func, params) in formulas.items():
        best_params, min_mse = evaluate_formula(func, params, X, y)
        results[name] = {'mse': min_mse, 'params': best_params}
    
    return results

def plot_data_and_fit(data, x_col, y_col, best_formula, best_params, title):
    """データと最適フィットの可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(data[x_col], data[y_col], alpha=0.6, color='blue')
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.set_title(f'{title} - データ分布')
    ax1.grid(True, alpha=0.3)
    
    if best_params is not None:
        x_range = np.linspace(data[x_col].min(), data[x_col].max(), 100)
        if 'sigma * T' in best_formula:
            T_mean = data['T'].mean()
            y_fit = best_params[0] * x_range * T_mean
        elif '/ n' in best_formula:
            y_fit = best_params[0] / x_range
        elif 'sqrt(d)' in best_formula:
            y_fit = best_params[0] + best_params[1] / np.sqrt(x_range)
        else:
            y_fit = None
            
        if y_fit is not None:
            ax2.scatter(data[x_col], data[y_col], alpha=0.6, color='blue', label='データ')
            ax2.plot(x_range, y_fit, 'r-', linewidth=2, label='フィット')
            ax2.set_xlabel(x_col)
            ax2.set_ylabel(y_col)
            ax2.set_title(f'{title} - フィット結果')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    st.title("🔬 材料工学シンボリック回帰")
    st.markdown("材料科学の物理法則をデータから発見")
    
    st.sidebar.header("📊 解析設定")
    analysis_type = st.sidebar.selectbox(
        "解析する物理法則を選択",
        ["熱伝導率 (Wiedemann-Franz法則)", "ホール効果", "機械的強度 (Hall-Petch関係)"]
    )
    
    if analysis_type == "熱伝導率 (Wiedemann-Franz法則)":
        st.header("🌡️ 熱伝導率解析")
        st.markdown("""
        **目標法則**: κ = L₀ × σ × T
        - κ: 熱伝導率 [W/(m·K)]
        - σ: 電気伝導率 [S/m]
        - T: 温度 [K]
        - L₀: ローレンツ数 ≈ 2.44×10⁻⁸ [W·Ω/K²]
        """)
        
        data = generate_thermal_conductivity_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 データ概要")
            st.dataframe(data.describe())
        
        with col2:
            st.subheader("🔍 相関行列")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
        
        if st.button("🚀 シンボリック回帰実行", key="thermal"):
            with st.spinner("解析中..."):
                results = analyze_thermal_conductivity(data)
                
                st.subheader("📊 候補式の評価結果")
                results_df = pd.DataFrame([
                    {
                        'Formula': name,
                        'MSE': f"{result['mse']:.2e}",
                        'Parameters': str(result['params']) if result['params'] is not None else "Failed"
                    }
                    for name, result in results.items()
                ])
                st.dataframe(results_df)
                
                best_formula = min(results, key=lambda k: results[k]['mse'])
                best_result = results[best_formula]
                
                st.success(f"🎯 **最適式**: {best_formula}")
                st.info(f"📈 **MSE**: {best_result['mse']:.2e}")
                st.info(f"🔢 **定数**: {best_result['params']}")
                
                if best_result['params'] is not None and 'sigma * T' in best_formula:
                    L0_found = best_result['params'][0]
                    L0_theory = 2.44e-8
                    st.success(f"✅ **発見されたローレンツ数**: {L0_found:.2e}")
                    st.success(f"📚 **理論値**: {L0_theory:.2e}")
                    st.success(f"🎯 **誤差**: {abs(L0_found - L0_theory)/L0_theory*100:.1f}%")
    
    elif analysis_type == "ホール効果":
        st.header("⚡ ホール効果解析")
        st.markdown("""
        **目標法則**: R_H = 1/(n × e)
        - R_H: ホール係数 [m³/C]
        - n: キャリア密度 [m⁻³]
        - e: 電子電荷 ≈ 1.602×10⁻¹⁹ [C]
        """)
        
        data = generate_hall_effect_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 データ概要")
            st.dataframe(data.describe())
        
        with col2:
            st.subheader("📊 データ分布")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.loglog(data['n'], data['R_H'], 'o', alpha=0.6)
            ax.set_xlabel('キャリア密度 n [m⁻³]')
            ax.set_ylabel('ホール係数 R_H [m³/C]')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        if st.button("🚀 シンボリック回帰実行", key="hall"):
            with st.spinner("解析中..."):
                results = analyze_hall_effect(data)
                
                st.subheader("📊 候補式の評価結果")
                results_df = pd.DataFrame([
                    {
                        'Formula': name,
                        'MSE': f"{result['mse']:.2e}",
                        'Parameters': str(result['params']) if result['params'] is not None else "Failed"
                    }
                    for name, result in results.items()
                ])
                st.dataframe(results_df)
                
                best_formula = min(results, key=lambda k: results[k]['mse'])
                best_result = results[best_formula]
                
                st.success(f"🎯 **最適式**: {best_formula}")
                st.info(f"📈 **MSE**: {best_result['mse']:.2e}")
                st.info(f"🔢 **定数**: {best_result['params']}")
                
                if best_result['params'] is not None and '/ n' in best_formula:
                    const_found = best_result['params'][0]
                    e_theory = 1.602e-19
                    e_found = 1 / const_found
                    st.success(f"✅ **発見された電子電荷**: {e_found:.2e}")
                    st.success(f"📚 **理論値**: {e_theory:.2e}")
                    st.success(f"🎯 **誤差**: {abs(e_found - e_theory)/e_theory*100:.1f}%")
    
    else:  # Hall-Petch関係
        st.header("🔧 機械的強度解析")
        st.markdown("""
        **目標法則**: σ_y = σ₀ + k/√d
        - σ_y: 降伏応力 [Pa]
        - σ₀: 基準応力 [Pa]
        - k: Hall-Petch定数
        - d: 結晶粒径 [m]
        """)
        
        data = generate_youngs_modulus_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 データ概要")
            st.dataframe(data.describe())
        
        with col2:
            st.subheader("📊 データ分布")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(1/np.sqrt(data['d']), data['sigma_y'], alpha=0.6)
            ax.set_xlabel('1/√d [m⁻¹/²]')
            ax.set_ylabel('降伏応力 σ_y [Pa]')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        if st.button("🚀 シンボリック回帰実行", key="hall_petch"):
            with st.spinner("解析中..."):
                results = analyze_youngs_modulus(data)
                
                st.subheader("📊 候補式の評価結果")
                results_df = pd.DataFrame([
                    {
                        'Formula': name,
                        'MSE': f"{result['mse']:.2e}",
                        'Parameters': str(result['params']) if result['params'] is not None else "Failed"
                    }
                    for name, result in results.items()
                ])
                st.dataframe(results_df)
                
                best_formula = min(results, key=lambda k: results[k]['mse'])
                best_result = results[best_formula]
                
                st.success(f"🎯 **最適式**: {best_formula}")
                st.info(f"📈 **MSE**: {best_result['mse']:.2e}")
                st.info(f"🔢 **定数**: {best_result['params']}")
                
                if best_result['params'] is not None and 'sqrt(d)' in best_formula:
                    sigma_0_found = best_result['params'][0]
                    k_found = best_result['params'][1]
                    st.success(f"✅ **発見された基準応力 σ₀**: {sigma_0_found:.2e} Pa")
                    st.success(f"✅ **発見されたHall-Petch定数 k**: {k_found:.2e}")
                    st.success(f"📚 **理論値 σ₀**: 5.0×10⁷ Pa")
                    st.success(f"📚 **理論値 k**: 5.0×10⁻⁴")

if __name__ == "__main__":
    main()
