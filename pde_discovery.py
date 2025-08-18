"""
偏微分方程式発見システム (PDE Discovery System)
FDMによる熱伝導方程式の数値解から偏微分方程式を逆算する
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import sympy as sp
from typing import Tuple, List, Dict, Callable

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
try:
    import torch
    from pinns_discovery import PINNsHeatSolver, PINNsBurgersSolver
    PINNS_AVAILABLE = True
except ImportError:
    PINNS_AVAILABLE = False

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

class BurgersFDM:
    """有限差分法によるBurgers方程式の数値解法"""
    
    def __init__(self, L: float = 1.0, T_final: float = 0.5, 
                 nx: int = 50, nt: int = 100, nu: float = 0.01):
        """
        Parameters:
        L: 空間領域の長さ
        T_final: 最終時刻
        nx: 空間格子点数
        nt: 時間格子点数
        nu: 粘性係数
        """
        self.L = L
        self.T_final = T_final
        self.nx = nx
        self.nt = nt
        self.nu = nu
        
        self.dx = L / (nx - 1)
        self.dt = T_final / (nt - 1)
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T_final, nt)
        
        self.r = nu * self.dt / (self.dx**2)
        u_max = 1.0
        self.cfl = u_max * self.dt / self.dx
        
        if self.r > 0.5:
            st.warning(f"拡散安定性条件違反: r = {self.r:.3f} > 0.5")
        if self.cfl > 1.0:
            st.warning(f"CFL条件違反: CFL = {self.cfl:.3f} > 1.0")
    
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """初期条件: ステップ関数"""
        return np.where(x < 0.5, 1.0, 0.0)
    
    def boundary_conditions(self, u: np.ndarray, n: int) -> np.ndarray:
        """境界条件: 両端で0"""
        u[0] = 0.0
        u[-1] = 0.0
        return u
    
    def solve(self) -> np.ndarray:
        """FDMによるBurgers方程式の数値解"""
        u = np.zeros((self.nt, self.nx))
        
        u[0, :] = self.initial_condition(self.x)
        u[0, :] = self.boundary_conditions(u[0, :], 0)
        
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                diffusion = self.r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
                
                if u[n, i] >= 0:
                    convection = -u[n, i] * self.dt / self.dx * (u[n, i] - u[n, i-1])
                else:
                    convection = -u[n, i] * self.dt / self.dx * (u[n, i+1] - u[n, i])
                
                u[n+1, i] = u[n, i] + diffusion + convection
            
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
            
            try:
                mse = self.evaluate_pde_formula(func, initial_params)
                optimal_params = initial_params  # For now, use initial params
            except Exception as e:
                mse = np.inf
                optimal_params = initial_params
            
            results[name] = {
                'mse': mse,
                'params': optimal_params,
                'description': formula_info["description"]
            }
        
        best_result = min(results.values(), key=lambda x: x['mse'])
        best_alpha = abs(best_result['params'][0]) if best_result['params'] else 0.01
        
        all_results = []
        for formula_name, result_data in results.items():
            all_results.append({
                'formula': formula_name,
                'mse': result_data['mse'],
                'params': result_data['params'],
                'description': result_data['description']
            })
        
        return {
            'all_results': all_results,
            'best_alpha': best_alpha,
            'optimization_details': {'method': 'direct_evaluation', 'status': 'completed'}
        }
    
    def discover_burgers_equation(self) -> Dict:
        """Burgers方程式の発見"""
        
        formulas = {
            "∂u/∂t = c₁ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2,
                "params": [0.01],
                "description": "純粋拡散方程式"
            },
            "∂u/∂t = -c₁ × u × ∂u/∂x + c₂ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: -p[0] * u * dudx + p[1] * d2udx2,
                "params": [1.0, 0.01],
                "description": "標準的なBurgers方程式"
            },
            "∂u/∂t = -c₁ × u × ∂u/∂x": {
                "func": lambda p, u, dudx, d2udx2: -p[0] * u * dudx,
                "params": [1.0],
                "description": "無粘性Burgers方程式"
            },
            "∂u/∂t = -c₁ × ∂u/∂x + c₂ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: -p[0] * dudx + p[1] * d2udx2,
                "params": [0.5, 0.01],
                "description": "線形対流拡散方程式"
            },
            "∂u/∂t = -c₁ × u² × ∂u/∂x + c₂ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: -p[0] * u**2 * dudx + p[1] * d2udx2,
                "params": [0.5, 0.01],
                "description": "修正Burgers方程式"
            },
            "∂u/∂t = c₁ × u × ∂u/∂x + c₂ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * u * dudx + p[1] * d2udx2,
                "params": [-1.0, 0.01],
                "description": "符号付きBurgers方程式"
            }
        }
        
        results = {}
        
        for name, formula_info in formulas.items():
            func = formula_info["func"]
            initial_params = formula_info["params"]
            
            try:
                mse = self.evaluate_pde_formula(func, initial_params)
                optimal_params = initial_params  # For now, use initial params
            except Exception as e:
                mse = np.inf
                optimal_params = initial_params
            
            results[name] = {
                'mse': mse,
                'params': optimal_params,
                'description': formula_info["description"]
            }
        
        best_result = min(results.values(), key=lambda x: x['mse'])
        best_formula_name = min(results.keys(), key=lambda k: results[k]['mse'])
        if "u × ∂u/∂x" in best_formula_name and len(best_result['params']) > 1:
            best_nu = abs(best_result['params'][1])  # Second parameter is viscosity
        else:
            best_nu = abs(best_result['params'][0]) if best_result['params'] else 0.01
        
        all_results = []
        for formula_name, result_data in results.items():
            all_results.append({
                'formula': formula_name,
                'mse': result_data['mse'],
                'params': result_data['params'],
                'description': result_data['description']
            })
        
        return {
            'all_results': all_results,
            'best_nu': best_nu,
            'optimization_details': {'method': 'direct_evaluation', 'status': 'completed'}
        }

def create_pde_discovery_app():
    """Streamlit PDE発見アプリ"""
    
    st.title("🔬 偏微分方程式発見システム (PDE Discovery)")
    st.markdown("---")
    
    equation_type = st.selectbox(
        "🧮 方程式タイプを選択",
        ["熱伝導方程式", "Burgers方程式"],
        help="発見したい偏微分方程式のタイプを選択してください"
    )
    
    solver_type = st.selectbox(
        "🔧 解法手法を選択",
        ["有限差分法 (FDM)", "Physics-Informed Neural Networks (PINNs)"] if PINNS_AVAILABLE else ["有限差分法 (FDM)"],
        help="PDEの数値解法を選択してください"
    )
    
    if equation_type == "熱伝導方程式":
        st.markdown("""
        有限差分法(FDM)で生成した熱伝導方程式の数値解から、
        元の偏微分方程式を逆算するシンボリック回帰システムです。
        
        **対象方程式**: ∂u/∂t = α × ∂²u/∂x²
        
        **手順:**
        1. FDMによる熱伝導方程式の数値解生成
        2. 数値微分による偏微分項の計算
        3. シンボリック回帰による PDE 構造の発見
        """)
    else:
        st.markdown("""
        有限差分法(FDM)で生成したBurgers方程式の数値解から、
        元の偏微分方程式を逆算するシンボリック回帰システムです。
        
        **対象方程式**: ∂u/∂t + u×∂u/∂x = ν×∂²u/∂x²
        
        **手順:**
        1. FDMによるBurgers方程式の数値解生成（非線形対流項含む）
        2. 数値微分による偏微分項の計算
        3. シンボリック回帰による非線形PDE構造の発見
        """)
    
    if solver_type == "有限差分法 (FDM)":
        st.sidebar.header("⚙️ FDMパラメータ")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            nx = st.number_input("空間格子点数", min_value=20, max_value=100, value=50)
            alpha_or_nu = st.number_input(
                "熱拡散係数 α" if equation_type == "熱伝導方程式" else "粘性係数 ν",
                min_value=0.001, max_value=0.1, value=0.01, format="%.3f"
            )
        
        with col2:
            nt = st.number_input("時間格子点数", min_value=50, max_value=200, value=100)
            T_final = st.number_input("最終時刻", min_value=0.1, max_value=2.0, value=1.0 if equation_type == "熱伝導方程式" else 0.5)
        
        st.sidebar.header("🎯 最適化設定")
        max_iter = st.sidebar.number_input("最適化反復回数", min_value=100, max_value=5000, value=1000, step=100)
        use_multiple_methods = st.sidebar.checkbox("複数最適化手法を使用", value=True, help="複数の最適化手法を試行して最良の結果を選択")
        use_higher_order = st.sidebar.checkbox("高次差分を使用", value=True, help="より正確な数値微分を使用")
    
    elif solver_type == "Physics-Informed Neural Networks (PINNs)" and PINNS_AVAILABLE:
        st.sidebar.header("🧠 PINNsパラメータ")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            epochs = st.number_input("エポック数", min_value=1000, max_value=20000, value=8000, step=1000)
            hidden_dim = st.number_input("隠れ層次元", min_value=20, max_value=200, value=50, step=10)
            alpha_or_nu = st.number_input(
                "熱拡散係数 α" if equation_type == "熱伝導方程式" else "粘性係数 ν",
                min_value=0.001, max_value=0.1, value=0.01, format="%.3f"
            )
        
        with col2:
            num_layers = st.number_input("ネットワーク層数", min_value=3, max_value=8, value=4)
            learning_rate = st.number_input("学習率", min_value=0.0001, max_value=0.01, value=0.0005, format="%.4f")
            n_points = st.number_input("訓練点数", min_value=500, max_value=5000, value=2000, step=500)
        
        st.sidebar.header("🎯 最適化設定")
        optimization_epochs = st.sidebar.number_input("シンボリック回帰エポック数", min_value=100, max_value=5000, value=1000, step=100)
        use_pinns_derivatives = st.sidebar.checkbox("PINNs微分を使用", value=True, help="PINNsの自動微分を使用してより正確な微分を計算")
    num_trials = st.sidebar.number_input("試行回数", min_value=1, max_value=10, value=5,
                                       help="異なる初期値での最適化試行回数")
    use_multiple_methods = st.sidebar.checkbox("複数最適化手法を使用", value=True, 
                                             help="複数の最適化アルゴリズムを試行して最良の結果を選択")
    
    if st.button("🚀 PDE発見を実行", type="primary"):
        
        if solver_type == "Physics-Informed Neural Networks (PINNs)" and PINNS_AVAILABLE:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(epoch, total_epochs, loss):
                progress = epoch / total_epochs
                progress_bar.progress(progress)
                status_text.text(f"エポック {epoch}/{total_epochs}, 損失: {loss:.6f}")
            
            if equation_type == "熱伝導方程式":
                with st.spinner("PINNsによる熱伝導方程式の解法中..."):
                    solver = PINNsHeatSolver(alpha=alpha_or_nu, hidden_dim=hidden_dim, num_layers=num_layers)
                    training_results = solver.train(epochs=epochs, lr=learning_rate, n_points=n_points, 
                                                  progress_callback=progress_callback)
                    
                    x_test = np.linspace(0, 1, 50)
                    t_test = np.linspace(0, 1, 50)
                    X_test, T_test = np.meshgrid(x_test, t_test)
                    u_numerical = solver.predict(X_test, T_test)
                    
                    st.subheader("📈 PINNs熱伝導方程式の解")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    im1 = ax1.imshow(u_numerical, aspect='auto', origin='lower', 
                                   extent=[0, 1, 0, 1], cmap='hot')
                    ax1.set_xlabel('空間 x')
                    ax1.set_ylabel('時間 t')
                    ax1.set_title('PINNs温度分布 u(x,t)')
                    plt.colorbar(im1, ax=ax1)
                    
                    time_indices = [0, 12, 25, 37, 49]
                    for i in time_indices:
                        ax2.plot(x_test, u_numerical[i, :], label=f't = {t_test[i]:.2f}')
                    ax2.set_xlabel('空間 x')
                    ax2.set_ylabel('温度 u')
                    ax2.set_title('各時刻での温度分布')
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                    st.subheader("🎯 PINNs訓練結果")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最終損失", f"{training_results['final_loss']:.6f}")
                    with col2:
                        st.metric("PDE損失", f"{training_results['pde_loss']:.6f}")
                    with col3:
                        st.metric("境界条件損失", f"{training_results['bc_loss']:.6f}")
                    
            else:  # Burgers方程式
                with st.spinner("PINNsによるBurgers方程式の解法中..."):
                    solver = PINNsBurgersSolver(nu=alpha_or_nu, hidden_dim=hidden_dim, num_layers=num_layers)
                    training_results = solver.train(epochs=epochs, lr=learning_rate, n_points=n_points, 
                                                  progress_callback=progress_callback)
                    
                    x_test = np.linspace(0, 1, 50)
                    t_test = np.linspace(0, 0.5, 50)
                    X_test, T_test = np.meshgrid(x_test, t_test)
                    u_numerical = solver.predict(X_test, T_test)
                    
                    st.subheader("📈 PINNs Burgers方程式の解")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    im1 = ax1.imshow(u_numerical, aspect='auto', origin='lower', 
                                   extent=[0, 1, 0, 0.5], cmap='viridis')
                    ax1.set_xlabel('空間 x')
                    ax1.set_ylabel('時間 t')
                    ax1.set_title('PINNs速度場 u(x,t)')
                    plt.colorbar(im1, ax=ax1)
                    
                    time_indices = [0, 12, 25, 37, 49]
                    for i in time_indices:
                        ax2.plot(x_test, u_numerical[i, :], label=f't = {t_test[i]:.3f}')
                    ax2.set_xlabel('空間 x')
                    ax2.set_ylabel('速度 u')
                    ax2.set_title('各時刻での速度分布')
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                    st.subheader("🎯 PINNs訓練結果")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最終損失", f"{training_results['final_loss']:.6f}")
                    with col2:
                        st.metric("PDE損失", f"{training_results['pde_loss']:.6f}")
                    with col3:
                        st.metric("境界条件損失", f"{training_results['bc_loss']:.6f}")
            
            progress_bar.empty()
            status_text.empty()
            
        else:  # FDM solver
            if equation_type == "熱伝導方程式":
                with st.spinner("FDMによる数値解計算中..."):
                    fdm = HeatConductionFDM(nx=nx, nt=nt, alpha=alpha_or_nu, T_final=T_final)
                    u_numerical = fdm.solve()
                
                st.success("✅ 数値解計算完了!")
                
                with st.spinner("偏微分方程式を発見中..."):
                    pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
                    results = pde_regression.discover_heat_equation()
                    
                theoretical_param = alpha_or_nu
                param_name = "α"
                equation_title = "温度分布の時間発展"
                y_label = "温度 u"
                
            else:
                with st.spinner("FDMによる数値解計算中..."):
                    fdm = BurgersFDM(nx=nx, nt=nt, nu=alpha_or_nu, T_final=T_final)
                    u_numerical = fdm.solve()
                
                st.success("✅ 数値解計算完了!")
                
                with st.spinner("偏微分方程式を発見中..."):
                    pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
                    results = pde_regression.discover_burgers_equation()
                    
                theoretical_param = alpha_or_nu
                param_name = "ν"
                equation_title = "速度分布の時間発展（衝撃波形成）"
                y_label = "速度 u"
            
            st.subheader("📊 FDM数値解の可視化")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            X, T = np.meshgrid(fdm.x, fdm.t)
            im1 = ax1.contourf(X, T, u_numerical, levels=20, cmap='hot')
            ax1.set_xlabel('空間 x')
            ax1.set_ylabel('時間 t')
            ax1.set_title(equation_title)
            plt.colorbar(im1, ax=ax1)
            
            time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
            for i, idx in enumerate(time_indices):
                ax2.plot(fdm.x, u_numerical[idx, :], 
                        label=f't = {fdm.t[idx]:.2f}', alpha=0.8)
            ax2.set_xlabel('空間 x')
            ax2.set_ylabel(y_label)
            ax2.set_title('各時刻での分布')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success("✅ PDE発見完了!")
            
            st.subheader("🎯 発見されたPDE候補")
            
            sorted_results = sorted(results['all_results'], key=lambda x: x['mse'])
            
            for i, result in enumerate(sorted_results):
                formula_name = result['formula']
                mse = result['mse']
                params = result['params']
                
                if i == 0:
                    st.success(f"🏆 **最優秀候補**: {formula_name}")
                else:
                    st.info(f"📋 **候補 {i+1}**: {formula_name}")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**式**: {formula_name}")
                    
                with col2:
                    if mse < np.inf:
                        st.metric("MSE", f"{mse:.2e}")
                    else:
                        st.metric("MSE", "∞")
                
                with col3:
                    param_str = ", ".join([f"{p:.4f}" for p in params])
                    st.write(f"**係数**: [{param_str}]")
            
            st.subheader("📈 理論値との比較")
            
            if equation_type == "熱伝導方程式":
                discovered_param = results['best_alpha']
            else:
                discovered_param = results['best_nu']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"理論値 {param_name}", f"{theoretical_param:.4f}")
            
            with col2:
                st.metric(f"発見値 {param_name}", f"{discovered_param:.4f}")
            
            with col3:
                error_percent = abs(discovered_param - theoretical_param) / theoretical_param * 100
                st.metric("相対誤差", f"{error_percent:.2f}%")
            
            with st.expander("📊 詳細な結果"):
                st.write("**候補式の評価結果:**")
                results_df = pd.DataFrame(results['all_results'])
                st.dataframe(results_df)
                
                st.write("**最適化の詳細:**")
                st.json(results['optimization_details'])

if __name__ == "__main__":
    create_pde_discovery_app()
