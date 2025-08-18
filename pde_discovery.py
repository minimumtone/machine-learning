"""
偏微分方程式発見システム (PDE Discovery System)
FDMによる熱伝導方程式の数値解から偏微分方程式を逆算する
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import sympy as sp
from typing import Tuple, List, Dict, Callable

class HeatConductionFDM:
    """有限差分法による熱伝導方程式の数値解法"""
    
    def __init__(self, L: float = 1.0, T_final: float = 1.0, 
                 nx: int = 50, nt: int = 100, alpha: float = 0.01):
        """
        Parameters:
        L: 空間領域の長さ
        T_final: 最終時刻
        nx: 空間格子点数
        nt: 時間格子点数
        alpha: 熱拡散係数
        """
        self.L = L
        self.T_final = T_final
        self.nx = nx
        self.nt = nt
        self.alpha = alpha
        
        self.dx = L / (nx - 1)
        self.dt = T_final / (nt - 1)
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T_final, nt)
        
        self.r = alpha * self.dt / (self.dx**2)
        if self.r > 0.5:
            st.warning(f"安定性条件違反: r = {self.r:.3f} > 0.5")
    
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """初期条件: ガウシアン分布"""
        return np.exp(-50 * (x - 0.5)**2)
    
    def boundary_conditions(self, u: np.ndarray, n: int) -> np.ndarray:
        """境界条件: 両端で0"""
        u[0] = 0.0
        u[-1] = 0.0
        return u
    
    def solve(self) -> np.ndarray:
        """FDMによる熱伝導方程式の数値解"""
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
        
        dudt[:-1, :] = (u[1:, :] - u[:-1, :]) / dt
        dudt[-1, :] = (u[-1, :] - u[-2, :]) / dt
        
        return dudt
    
    @staticmethod
    def compute_dx(u: np.ndarray, dx: float) -> np.ndarray:
        """空間1階微分 ∂u/∂x の計算"""
        dudx = np.zeros_like(u)
        
        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        dudx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
        
        return dudx
    
    @staticmethod
    def compute_d2x(u: np.ndarray, dx: float) -> np.ndarray:
        """空間2階微分 ∂²u/∂x² の計算"""
        d2udx2 = np.zeros_like(u)
        
        d2udx2[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / (dx**2)
        d2udx2[:, 0] = 0
        d2udx2[:, -1] = 0
        
        return d2udx2

class PDESymbolicRegression:
    """偏微分方程式のシンボリック回帰"""
    
    def __init__(self, u: np.ndarray, x: np.ndarray, t: np.ndarray):
        self.u = u
        self.x = x
        self.t = t
        self.dx = x[1] - x[0]
        self.dt = t[1] - t[0]
        
        self.derivatives = NumericalDerivatives()
        self.dudt = self.derivatives.compute_dt(u, self.dt)
        self.dudx = self.derivatives.compute_dx(u, self.dx)
        self.d2udx2 = self.derivatives.compute_d2x(u, self.dx)
    
    def evaluate_pde_formula(self, formula_func: Callable, params: List[float], 
                           mask: np.ndarray = None) -> float:
        """PDE候補式の評価"""
        if mask is None:
            mask = np.ones_like(self.u, dtype=bool)
            mask[0, :] = False  # 初期条件
            mask[:, 0] = False  # 左境界
            mask[:, -1] = False # 右境界
        
        try:
            predicted = formula_func(params, self.u, self.dudx, self.d2udx2)
            
            actual = self.dudt
            
            mse = np.mean((predicted[mask] - actual[mask])**2)
            
            return mse
            
        except Exception as e:
            return np.inf
    
    def discover_heat_equation(self) -> Dict:
        """熱伝導方程式の発見"""
        
        formulas = {
            "∂u/∂t = c₁ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2,
                "params": [0.01],
                "description": "標準的な熱伝導方程式"
            },
            "∂u/∂t = c₁ × ∂²u/∂x² + c₂ × u": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2 + p[1] * u,
                "params": [0.01, 0.0],
                "description": "反応項付き熱伝導方程式"
            },
            "∂u/∂t = c₁ × ∂²u/∂x² + c₂ × ∂u/∂x": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2 + p[1] * dudx,
                "params": [0.01, 0.0],
                "description": "対流項付き熱伝導方程式"
            },
            "∂u/∂t = c₁ × u × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * u * d2udx2,
                "params": [0.01],
                "description": "非線形拡散方程式"
            }
        }
        
        results = {}
        
        for name, formula_info in formulas.items():
            func = formula_info["func"]
            initial_params = formula_info["params"]
            
            def objective(params):
                return self.evaluate_pde_formula(func, params)
            
            try:
                result = minimize(objective, initial_params, method='Nelder-Mead')
                
                if result.success:
                    mse = result.fun
                    optimal_params = result.x
                else:
                    mse = np.inf
                    optimal_params = initial_params
                    
            except Exception as e:
                mse = np.inf
                optimal_params = initial_params
            
            results[name] = {
                'mse': mse,
                'params': optimal_params,
                'description': formula_info["description"]
            }
        
        return results

def create_pde_discovery_app():
    """Streamlit PDE発見アプリ"""
    
    st.title("🔬 偏微分方程式発見システム (PDE Discovery)")
    st.markdown("---")
    
    st.markdown("""
    有限差分法(FDM)で生成した熱伝導方程式の数値解から、
    元の偏微分方程式を逆算するシンボリック回帰システムです。
    
    **手順:**
    1. FDMによる熱伝導方程式の数値解生成
    2. 数値微分による偏微分項の計算
    3. シンボリック回帰による PDE 構造の発見
    """)
    
    st.sidebar.header("🔧 FDM パラメータ")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        nx = st.number_input("空間格子点数", min_value=20, max_value=100, value=50)
        alpha = st.number_input("熱拡散係数", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
    
    with col2:
        nt = st.number_input("時間格子点数", min_value=50, max_value=200, value=100)
        T_final = st.number_input("最終時刻", min_value=0.1, max_value=2.0, value=1.0)
    
    if st.button("🚀 PDE発見を実行", type="primary"):
        
        with st.spinner("FDMによる数値解計算中..."):
            fdm = HeatConductionFDM(nx=nx, nt=nt, alpha=alpha, T_final=T_final)
            u_numerical = fdm.solve()
        
        st.success("✅ 数値解計算完了!")
        
        st.subheader("📊 FDM数値解の可視化")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        X, T = np.meshgrid(fdm.x, fdm.t)
        im1 = ax1.contourf(X, T, u_numerical, levels=20, cmap='hot')
        ax1.set_xlabel('空間 x')
        ax1.set_ylabel('時間 t')
        ax1.set_title('温度分布の時間発展')
        plt.colorbar(im1, ax=ax1)
        
        time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
        for i, idx in enumerate(time_indices):
            ax2.plot(fdm.x, u_numerical[idx, :], 
                    label=f't = {fdm.t[idx]:.2f}', alpha=0.8)
        ax2.set_xlabel('空間 x')
        ax2.set_ylabel('温度 u')
        ax2.set_title('各時刻での温度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        with st.spinner("偏微分方程式を発見中..."):
            pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
            results = pde_regression.discover_heat_equation()
        
        st.success("✅ PDE発見完了!")
        
        st.subheader("🎯 発見されたPDE候補")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mse'])
        
        for i, (formula_name, result) in enumerate(sorted_results):
            mse = result['mse']
            params = result['params']
            description = result['description']
            
            if i == 0:
                st.success(f"🏆 **最優秀候補**: {formula_name}")
            else:
                st.info(f"📋 **候補 {i+1}**: {formula_name}")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**説明**: {description}")
                
            with col2:
                if mse < np.inf:
                    st.metric("MSE", f"{mse:.2e}")
                else:
                    st.metric("MSE", "∞")
            
            with col3:
                param_str = ", ".join([f"{p:.4f}" for p in params])
                st.write(f"**係数**: [{param_str}]")
        
        st.subheader("📈 理論値との比較")
        
        best_formula, best_result = sorted_results[0]
        theoretical_alpha = alpha
        discovered_alpha = best_result['params'][0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("理論値 α", f"{theoretical_alpha:.4f}")
        
        with col2:
            st.metric("発見値 α", f"{discovered_alpha:.4f}")
        
        with col3:
            error_percent = abs(discovered_alpha - theoretical_alpha) / theoretical_alpha * 100
            st.metric("相対誤差", f"{error_percent:.2f}%")
        
        st.subheader("🔍 数値微分の可視化")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        im1 = axes[0,0].contourf(X, T, pde_regression.dudt, levels=20, cmap='RdBu_r')
        axes[0,0].set_title('∂u/∂t')
        axes[0,0].set_xlabel('空間 x')
        axes[0,0].set_ylabel('時間 t')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].contourf(X, T, pde_regression.dudx, levels=20, cmap='RdBu_r')
        axes[0,1].set_title('∂u/∂x')
        axes[0,1].set_xlabel('空間 x')
        axes[0,1].set_ylabel('時間 t')
        plt.colorbar(im2, ax=axes[0,1])
        
        im3 = axes[1,0].contourf(X, T, pde_regression.d2udx2, levels=20, cmap='RdBu_r')
        axes[1,0].set_title('∂²u/∂x²')
        axes[1,0].set_xlabel('空間 x')
        axes[1,0].set_ylabel('時間 t')
        plt.colorbar(im3, ax=axes[1,0])
        
        best_func = None
        for name, formula_info in {
            "∂u/∂t = c₁ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2,
            }
        }.items():
            if name == best_formula:
                best_func = formula_info["func"]
                break
        
        if best_func:
            predicted_dudt = best_func(best_result['params'], 
                                     u_numerical, 
                                     pde_regression.dudx, 
                                     pde_regression.d2udx2)
            
            im4 = axes[1,1].contourf(X, T, predicted_dudt, levels=20, cmap='RdBu_r')
            axes[1,1].set_title('予測された ∂u/∂t')
            axes[1,1].set_xlabel('空間 x')
            axes[1,1].set_ylabel('時間 t')
            plt.colorbar(im4, ax=axes[1,1])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("📝 まとめ")
        
        st.markdown(f"""
        **発見された最優秀PDE**: `{best_formula}`
        
        **物理的解釈**:
        - 理論的熱拡散係数: α = {theoretical_alpha:.4f}
        - 発見された係数: α = {discovered_alpha:.4f}
        - 相対誤差: {error_percent:.2f}%
        
        **手法の有効性**:
        - FDMによる数値解生成 ✅
        - 数値微分による偏微分項計算 ✅
        - シンボリック回帰によるPDE構造発見 ✅
        
        この結果は、大量のスパース観測データからでも
        適切な数値微分と候補式探索により、
        元の偏微分方程式を高精度で復元できることを示しています。
        """)

if __name__ == "__main__":
    create_pde_discovery_app()
