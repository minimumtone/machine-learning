"""
PINNsとBICによるモデル選択 - 拡散方程式発見システム

このプログラムは、拡散方程式を記号回帰で発見するプロセスを、
「全状態探索」と「BIC基準によるスコアリング」という2つの核となる概念を用いて実現します。

対象方程式: ∂u/∂t = D × ∂²u/∂x²
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Callable
from itertools import combinations, product
import time

matplotlib.rcParams['font.family'] = ['IPAGothic', 'IPAPGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    import torch
    from pinns_discovery import PINNsDiffusionSolver
    PINNS_AVAILABLE = True
except ImportError:
    PINNS_AVAILABLE = False

class DiffusionFDM:
    """有限差分法による拡散方程式の数値解法"""
    
    def __init__(self, L: float = 0.02, T_final: float = 1000, 
                 nx: int = 30, nt: int = 50, D: float = 1e-11):
        """
        Parameters:
        L: 空間領域の長さ (m)
        T_final: 最終時刻 (s)
        nx: 空間格子点数
        nt: 時間格子点数
        D: 拡散係数 (m²/s)
        """
        self.L = L
        self.T_final = T_final
        self.nx = nx
        self.nt = nt
        self.D = D
        
        self.dx = L / (nx - 1)
        self.dt = T_final / (nt - 1)
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T_final, nt)
        
        self.r = D * self.dt / (self.dx**2)
        if self.r > 0.5:
            st.warning(f"安定性条件違反: r = {self.r:.3f} > 0.5")
    
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """初期条件: ガウシアン分布"""
        return np.exp(-50 * (x - self.L/2)**2 / self.L**2)
    
    def boundary_conditions(self, u: np.ndarray, n: int) -> np.ndarray:
        """境界条件: 両端で0"""
        u[0] = 0.0
        u[-1] = 0.0
        return u
    
    def solve(self) -> np.ndarray:
        """FDMによる拡散方程式の数値解"""
        u = np.zeros((self.nt, self.nx))
        
        u[0, :] = self.initial_condition(self.x)
        u[0, :] = self.boundary_conditions(u[0, :], 0)
        
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                u[n+1, i] = u[n, i] + self.r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
            u[n+1, :] = self.boundary_conditions(u[n+1, :], n+1)
        
        return u

class NumericalDerivatives:
    """数値微分計算クラス"""
    
    @staticmethod
    def compute_dt(u: np.ndarray, dt: float) -> np.ndarray:
        """時間微分 ∂u/∂t の計算"""
        dudt = np.zeros_like(u)
        dudt[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dt)
        dudt[0, :] = (u[1, :] - u[0, :]) / dt
        dudt[-1, :] = (u[-1, :] - u[-2, :]) / dt
        return dudt
    
    @staticmethod
    def compute_dx(u: np.ndarray, dx: float) -> np.ndarray:
        """空間微分 ∂u/∂x の計算"""
        dudx = np.zeros_like(u)
        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        dudx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
        return dudx
    
    @staticmethod
    def compute_d2x(u: np.ndarray, dx: float) -> np.ndarray:
        """2次空間微分 ∂²u/∂x² の計算"""
        d2udx2 = np.zeros_like(u)
        d2udx2[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / (dx**2)
        return d2udx2

class ComplexityCalculator:
    """モデル複雑度計算クラス"""
    
    def __init__(self):
        self.operator_weights = {
            'constant': 1,
            'variable': 1,
            'add': 1,
            'multiply': 2,
            'power': 3,
            'derivative': 2,
            'second_derivative': 3
        }
    
    def calculate_complexity(self, expression_str: str, n_params: int) -> float:
        """式の複雑度を計算"""
        complexity = n_params  # パラメータ数
        
        complexity += expression_str.count('+') * self.operator_weights['add']
        complexity += expression_str.count('×') * self.operator_weights['multiply']
        complexity += expression_str.count('∂²') * self.operator_weights['second_derivative']
        complexity += expression_str.count('∂') * self.operator_weights['derivative']
        
        return complexity

class FullStateSearchBIC:
    """全状態探索とBIC基準によるモデル選択"""
    
    def __init__(self, u_data: np.ndarray, x: np.ndarray, t: np.ndarray):
        self.u = u_data
        self.x = x
        self.t = t
        self.dx = x[1] - x[0]
        self.dt = t[1] - t[0]
        
        self.dudt = NumericalDerivatives.compute_dt(u_data, self.dt)
        self.dudx = NumericalDerivatives.compute_dx(u_data, self.dx)
        self.d2udx2 = NumericalDerivatives.compute_d2x(u_data, self.dx)
        
        self.complexity_calc = ComplexityCalculator()
    
    def generate_candidate_expressions(self, max_complexity: int = 4) -> List[Dict]:
        """全状態探索による候補式生成"""
        
        base_terms = {
            "c": {
                "func": lambda p, u, dudx, d2udx2: np.full_like(u, p[0]),
                "description": "定数項",
                "complexity": 1
            },
            "c × u": {
                "func": lambda p, u, dudx, d2udx2: p[0] * u,
                "description": "濃度項",
                "complexity": 2
            },
            "c × ∂u/∂x": {
                "func": lambda p, u, dudx, d2udx2: p[0] * dudx,
                "description": "1次微分項",
                "complexity": 2
            },
            "c × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2,
                "description": "2次微分項（拡散項）",
                "complexity": 3
            },
            "c × u²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * u**2,
                "description": "非線形濃度項",
                "complexity": 3
            },
            "c × u × ∂u/∂x": {
                "func": lambda p, u, dudx, d2udx2: p[0] * u * dudx,
                "description": "非線形対流項",
                "complexity": 4
            },
            "c × u × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * u * d2udx2,
                "description": "濃度依存拡散項",
                "complexity": 4
            }
        }
        
        candidates = []
        
        for complexity in range(1, max_complexity + 1):
            st.write(f"**複雑度 {complexity}** の候補式を生成中...")
            
            if complexity == 1:
                for term_name, term_info in base_terms.items():
                    if term_info["complexity"] <= complexity:
                        candidates.append({
                            'name': f"∂u/∂t = {term_name}",
                            'terms': [term_name],
                            'func': term_info['func'],
                            'n_params': 1,
                            'complexity': complexity
                        })
            
            else:
                for n_terms in range(2, min(complexity + 1, 4)):  # 最大3項まで
                    for term_combo in combinations(base_terms.keys(), n_terms):
                        total_complexity = sum(base_terms[term]["complexity"] for term in term_combo)
                        if total_complexity <= complexity:
                            
                            def make_combined_func(terms):
                                def combined_func(params, u, dudx, d2udx2):
                                    result = np.zeros_like(u)
                                    for i, term in enumerate(terms):
                                        result += base_terms[term]['func']([params[i]], u, dudx, d2udx2)
                                    return result
                                return combined_func
                            
                            formula_name = " + ".join([f"c{i+1} × {term.replace('c × ', '')}" for i, term in enumerate(term_combo)])
                            
                            candidates.append({
                                'name': f"∂u/∂t = {formula_name}",
                                'terms': list(term_combo),
                                'func': make_combined_func(term_combo),
                                'n_params': len(term_combo),
                                'complexity': total_complexity
                            })
        
        return candidates
    
    def evaluate_candidate(self, candidate: Dict) -> Dict:
        """候補式の評価（パラメータ最適化とBIC計算）"""
        
        def objective(params):
            try:
                predicted = candidate['func'](params, self.u[1:-1, 1:-1], 
                                             self.dudx[1:-1, 1:-1], 
                                             self.d2udx2[1:-1, 1:-1])
                target = self.dudt[1:-1, 1:-1]
                
                mse = np.mean((predicted - target)**2)
                return mse
            except:
                return 1e10
        
        initial_params = [1e-11] * candidate['n_params']
        
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B')
            optimized_params = result.x
            mse = result.fun
        except:
            optimized_params = initial_params
            mse = objective(initial_params)
        
        n_data = (self.u.shape[0] - 2) * (self.u.shape[1] - 2)
        
        likelihood = np.exp(-n_data * mse / 2)
        
        bic = -2 * np.log(likelihood + 1e-10) + candidate['n_params'] * np.log(n_data)
        
        aic = -2 * np.log(likelihood + 1e-10) + 2 * candidate['n_params']
        
        return {
            'name': candidate['name'],
            'mse': mse,
            'likelihood': likelihood,
            'bic': bic,
            'aic': aic,
            'n_params': candidate['n_params'],
            'complexity': candidate['complexity'],
            'optimized_params': optimized_params
        }
    
    def calculate_model_weights(self, bic_scores: np.ndarray) -> np.ndarray:
        """BICスコアからモデル重みを計算"""
        delta_bic = bic_scores - np.min(bic_scores)
        weights = np.exp(-0.5 * delta_bic)
        return weights / np.sum(weights)
    
    def run_full_search(self, max_complexity: int = 4) -> Dict:
        """全状態探索の実行"""
        
        st.write("### 🔍 全状態探索による候補式生成")
        
        candidates = self.generate_candidate_expressions(max_complexity)
        st.write(f"生成された候補式数: **{len(candidates)}**")
        
        st.write("### 📊 BIC基準による候補式評価")
        
        results = []
        progress_bar = st.progress(0)
        
        for i, candidate in enumerate(candidates):
            result = self.evaluate_candidate(candidate)
            results.append(result)
            progress_bar.progress((i + 1) / len(candidates))
        
        results.sort(key=lambda x: x['bic'])
        
        bic_scores = np.array([r['bic'] for r in results])
        model_weights = self.calculate_model_weights(bic_scores)
        
        for i, result in enumerate(results):
            result['model_weight'] = model_weights[i]
            result['posterior_prob'] = model_weights[i]
        
        return {
            'results': results,
            'best_model': results[0] if results else None,
            'model_weights': model_weights
        }

def create_results_table(results: List[Dict]) -> pd.DataFrame:
    """結果をテーブル形式で表示"""
    
    df_data = []
    for i, result in enumerate(results[:10]):  # 上位10個まで表示
        df_data.append({
            '順位': i + 1,
            '候補式': result['name'],
            'データ適合度 (MSE)': f"{result['mse']:.2e}",
            '複雑さ (k)': result['n_params'],
            'BICスコア': f"{result['bic']:.1f}",
            '事後確率': f"{result['posterior_prob']:.3f}",
            '最適化パラメータ': ', '.join([f"{p:.2e}" for p in result['optimized_params']])
        })
    
    return pd.DataFrame(df_data)

def visualize_results(search_results: Dict, true_D: float):
    """結果の可視化"""
    
    results = search_results['results']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    bic_scores = [r['bic'] for r in results[:10]]
    model_names = [r['name'].replace('∂u/∂t = ', '') for r in results[:10]]
    
    colors = ['red' if i == 0 else 'blue' for i in range(len(bic_scores))]
    bars = ax1.bar(range(len(bic_scores)), bic_scores, color=colors, alpha=0.7)
    ax1.set_xlabel('候補式')
    ax1.set_ylabel('BICスコア')
    ax1.set_title('BICスコア比較（低いほど良い）')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels([f"式{i+1}" for i in range(len(model_names))], rotation=45)
    
    bars[0].set_color('red')
    bars[0].set_alpha(1.0)
    
    posterior_probs = [r['posterior_prob'] for r in results[:10]]
    ax2.bar(range(len(posterior_probs)), posterior_probs, color='green', alpha=0.7)
    ax2.set_xlabel('候補式')
    ax2.set_ylabel('事後確率')
    ax2.set_title('モデルの事後確率')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels([f"式{i+1}" for i in range(len(model_names))], rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    best_model = search_results['best_model']
    if best_model:
        st.write("### 🏆 最適モデル発見結果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**発見された式**: {best_model['name']}")
            st.write(f"**BICスコア**: {best_model['bic']:.1f}")
            st.write(f"**事後確率**: {best_model['posterior_prob']:.3f}")
            st.write(f"**MSE**: {best_model['mse']:.2e}")
        
        with col2:
            if len(best_model['optimized_params']) > 0:
                discovered_D = best_model['optimized_params'][0]
                st.write(f"**発見された拡散係数**: {discovered_D:.2e}")
                st.write(f"**真の拡散係数**: {true_D:.2e}")
                error = abs(discovered_D - true_D) / true_D * 100
                st.write(f"**相対誤差**: {error:.1f}%")
                
                if error < 10:
                    st.success("✅ 正解！拡散方程式を正確に発見しました！")
                elif error < 50:
                    st.warning("⚠️ 近似的に発見できました")
                else:
                    st.error("❌ 発見に失敗しました")

def main():
    """メインアプリケーション"""
    
    st.set_page_config(
        page_title="PINNsとBICによるモデル選択",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 PINNsとBICによるモデル選択")
    st.markdown("### 拡散方程式の記号回帰による発見")
    
    st.markdown("""
    このアプリケーションは、**全状態探索**と**BIC基準によるスコアリング**を用いて、
    拡散方程式 ∂u/∂t = D × ∂²u/∂x² を数値データから自動発見するプロセスを実演します。
    
    **プロセス概要:**
    1. **データ準備**: FDMによる拡散方程式の数値解生成
    2. **全状態探索**: 単純な式から複雑な式へと段階的に候補式を生成
    3. **BIC評価**: 各候補式を「データ適合度」と「式の簡潔さ」で評価
    4. **モデル選択**: BICスコアが最小の式を最適解として選択
    """)
    
    st.sidebar.header("⚙️ パラメータ設定")
    
    st.sidebar.subheader("📊 データ生成パラメータ")
    L = st.sidebar.number_input("空間領域長さ (m)", min_value=0.01, max_value=0.1, value=0.02, format="%.3f")
    T_final = st.sidebar.number_input("最終時刻 (s)", min_value=100, max_value=5000, value=1000)
    nx = st.sidebar.number_input("空間格子点数", min_value=20, max_value=50, value=30)
    nt = st.sidebar.number_input("時間格子点数", min_value=30, max_value=100, value=50)
    D_true = st.sidebar.number_input("真の拡散係数 (m²/s)", min_value=1e-12, max_value=1e-10, value=1e-11, format="%.2e")
    
    st.sidebar.subheader("🔍 探索パラメータ")
    max_complexity = st.sidebar.slider("最大複雑度", 2, 5, 4)
    
    if st.sidebar.button("🚀 拡散方程式発見を開始", type="primary"):
        
        st.write("## Step 1: 問題の定式化とデータ準備")
        
        with st.spinner("FDMによる拡散方程式の数値解を生成中..."):
            fdm_solver = DiffusionFDM(L=L, T_final=T_final, nx=nx, nt=nt, D=D_true)
            u_numerical = fdm_solver.solve()
        
        st.success("✅ 数値解生成完了")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        X, T = np.meshgrid(fdm_solver.x, fdm_solver.t)
        im1 = ax1.contourf(X, T, u_numerical, levels=20, cmap='viridis')
        ax1.set_xlabel('位置 x (m)')
        ax1.set_ylabel('時間 t (s)')
        ax1.set_title('拡散方程式の数値解')
        plt.colorbar(im1, ax=ax1, label='濃度 u')
        
        time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
        for i in time_indices:
            ax2.plot(fdm_solver.x, u_numerical[i, :], label=f't = {fdm_solver.t[i]:.0f}s')
        ax2.set_xlabel('位置 x (m)')
        ax2.set_ylabel('濃度 u')
        ax2.set_title('時間発展による濃度分布の変化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("## Step 2: 全状態探索による候補式生成")
        
        with st.spinner("全状態探索を実行中..."):
            search_engine = FullStateSearchBIC(u_numerical, fdm_solver.x, fdm_solver.t)
            search_results = search_engine.run_full_search(max_complexity)
        
        st.write("## Step 3: BIC基準による最適モデル選択")
        
        if search_results['results']:
            results_df = create_results_table(search_results['results'])
            st.write("### 📋 候補式評価結果")
            st.dataframe(results_df, use_container_width=True)
            
            visualize_results(search_results, D_true)
            
            st.write("### 📖 結果の解釈")
            st.markdown("""
            **BICスコアの解釈:**
            - BIC = -2ln(L) + k×ln(n)
            - 第1項: データ適合度（小さいほど良い）
            - 第2項: 複雑度ペナルティ（パラメータ数kに比例）
            - **総合スコアが最小の式が最適解**
            
            **期待される結果:**
            - 真の拡散方程式 ∂u/∂t = D × ∂²u/∂x² が最良のBICスコアを獲得
            - より複雑な式は不要なパラメータによりペナルティを受ける
            - 単純すぎる式はデータ適合度が悪くスコアが悪化
            """)
            
        else:
            st.error("❌ 候補式の評価に失敗しました")
    
    if PINNS_AVAILABLE:
        st.write("## オプション: PINNsによる検証")
        
        if st.button("🧠 PINNsで検証"):
            st.info("PINNsによる検証機能は実装中です")
    else:
        st.info("💡 PINNsライブラリが利用できません。FDMベースの解析のみ実行されます。")

if __name__ == "__main__":
    main()
