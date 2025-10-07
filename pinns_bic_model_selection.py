"""
拡散方程式発見システム - PINNsとBICモデル選択
Physics-Informed Neural Networks (PINNs) と Bayesian Information Criterion (BIC) を用いた
拡散方程式のパラメータ発見システム

完全な自己完結型ファイル - 外部インポートなし
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Callable, Optional
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from itertools import product
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PINNS_AVAILABLE = True
except ImportError:
    PINNS_AVAILABLE = False


if PINNS_AVAILABLE:
    class DiffusionPINN(nn.Module):
        """拡散方程式を解くためのPhysics-Informed Neural Network"""
        
        def __init__(self, hidden_dim: int = 50, num_layers: int = 4):
            super(DiffusionPINN, self).__init__()
            
            layers = []
            layers.append(nn.Linear(2, hidden_dim))
            layers.append(nn.Tanh())
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Tanh())
            
            layers.append(nn.Linear(hidden_dim, 1))
            
            self.network = nn.Sequential(*layers)
            
            for m in self.network.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """ニューラルネットワークの順伝播
            
            Args:
                x: 空間座標 (batch_size, 1)
                t: 時間座標 (batch_size, 1)
            
            Returns:
                濃度 u(x,t) (batch_size, 1)
            """
            inputs = torch.cat([x, t], dim=1)
            return self.network(inputs)


    class PINNsDiffusionSolver:
        """PINNsを用いた拡散方程式ソルバー
        
        ∂u/∂t = D × ∂²u/∂x²
        """
        
        def __init__(self, D: float = 1e-9, L: float = 0.02, T_final: float = 3600.0,
                     hidden_dim: int = 50, num_layers: int = 4):
            """
            Args:
                D: 拡散係数 [m²/s]
                L: 空間領域の長さ [m]
                T_final: 最終時間 [s]
                hidden_dim: 隠れ層の次元
                num_layers: 隠れ層の数
            """
            self.D = D
            self.L = L
            self.T_final = T_final
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = DiffusionPINN(hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
            
        def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
            """初期条件: u(x, 0) = sin(πx/L)"""
            return torch.sin(np.pi * x / self.L)
        
        def boundary_condition_left(self, t: torch.Tensor) -> torch.Tensor:
            """左境界条件: u(0, t) = 0"""
            return torch.zeros_like(t)
        
        def boundary_condition_right(self, t: torch.Tensor) -> torch.Tensor:
            """右境界条件: u(L, t) = 0"""
            return torch.zeros_like(t)
        
        def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """PDE残差の計算: ∂u/∂t - D × ∂²u/∂x²"""
            x.requires_grad_(True)
            t.requires_grad_(True)
            
            u = self.model(x, t)
            
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                     create_graph=True)[0]
            
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                     create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                      create_graph=True)[0]
            
            residual = u_t - self.D * u_xx
            return residual
        
        def train(self, epochs: int = 5000, lr: float = 0.001, n_points: int = 2000,
                 progress_callback: Optional[Callable] = None) -> Dict[str, float]:
            """PINNsの訓練
            
            Args:
                epochs: エポック数
                lr: 学習率
                n_points: サンプリング点数
                progress_callback: 進捗コールバック関数
            
            Returns:
                訓練結果の辞書
            """
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                x_pde = torch.rand(n_points, 1, device=self.device) * self.L
                t_pde = torch.rand(n_points, 1, device=self.device) * self.T_final
                
                x_ic = torch.rand(n_points // 4, 1, device=self.device) * self.L
                t_ic = torch.zeros(n_points // 4, 1, device=self.device)
                
                t_bc = torch.rand(n_points // 4, 1, device=self.device) * self.T_final
                x_bc_left = torch.zeros(n_points // 4, 1, device=self.device)
                x_bc_right = torch.ones(n_points // 4, 1, device=self.device) * self.L
                
                pde_loss = torch.mean(self.pde_residual(x_pde, t_pde) ** 2)
                
                u_ic_pred = self.model(x_ic, t_ic)
                u_ic_true = self.initial_condition(x_ic)
                ic_loss = torch.mean((u_ic_pred - u_ic_true) ** 2)
                
                u_bc_left_pred = self.model(x_bc_left, t_bc)
                u_bc_left_true = self.boundary_condition_left(t_bc)
                bc_left_loss = torch.mean((u_bc_left_pred - u_bc_left_true) ** 2)
                
                u_bc_right_pred = self.model(x_bc_right, t_bc)
                u_bc_right_true = self.boundary_condition_right(t_bc)
                bc_right_loss = torch.mean((u_bc_right_pred - u_bc_right_true) ** 2)
                
                bc_loss = bc_left_loss + bc_right_loss
                
                loss = pde_loss + 10.0 * ic_loss + 10.0 * bc_loss
                
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                
                if progress_callback is not None and epoch % 100 == 0:
                    progress_callback(epoch, epochs, loss.item())
            
            return {
                'final_loss': loss.item(),
                'pde_loss': pde_loss.item(),
                'bc_loss': bc_loss.item()
            }
        
        def predict(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
            """濃度場の予測
            
            Args:
                X: 空間メッシュグリッド
                T: 時間メッシュグリッド
            
            Returns:
                濃度場 u(x,t)
            """
            self.model.eval()
            
            x_flat = torch.FloatTensor(X.flatten().reshape(-1, 1)).to(self.device)
            t_flat = torch.FloatTensor(T.flatten().reshape(-1, 1)).to(self.device)
            
            with torch.no_grad():
                u_flat = self.model(x_flat, t_flat)
            
            u = u_flat.cpu().numpy().reshape(X.shape)
            return u


class DiffusionFDM:
    """有限差分法による拡散方程式の数値解法"""
    
    def __init__(self, L: float = 0.02, T_final: float = 3600.0, 
                 nx: int = 50, nt: int = 100, D: float = 1e-9):
        """
        Args:
            L: 空間領域の長さ [m]
            T_final: 最終時間 [s]
            nx: 空間グリッド数
            nt: 時間ステップ数
            D: 拡散係数 [m²/s]
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
        
        self.r = D * self.dt / (self.dx ** 2)
        if self.r > 0.5:
            st.warning(f"⚠️ 安定性条件違反: r = {self.r:.3f} > 0.5")
        
        self.u = np.zeros((nt, nx))
    
    def initial_condition(self):
        """初期条件の設定"""
        self.u[0, :] = np.sin(np.pi * self.x / self.L)
    
    def boundary_conditions(self):
        """境界条件の設定"""
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
    
    def solve(self):
        """拡散方程式を解く"""
        self.initial_condition()
        self.boundary_conditions()
        
        for n in range(0, self.nt - 1):
            for i in range(1, self.nx - 1):
                self.u[n + 1, i] = self.u[n, i] + self.r * (
                    self.u[n, i + 1] - 2 * self.u[n, i] + self.u[n, i - 1]
                )
            
            self.u[n + 1, 0] = 0.0
            self.u[n + 1, -1] = 0.0
        
        return self.u


class NumericalDerivatives:
    """数値微分計算クラス"""
    
    @staticmethod
    def compute_dt(u: np.ndarray, dt: float) -> np.ndarray:
        """時間微分の計算"""
        dudt = np.zeros_like(u)
        dudt[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dt)
        dudt[0, :] = (u[1, :] - u[0, :]) / dt
        dudt[-1, :] = (u[-1, :] - u[-2, :]) / dt
        return dudt
    
    @staticmethod
    def compute_dx(u: np.ndarray, dx: float) -> np.ndarray:
        """空間微分の計算"""
        dudx = np.zeros_like(u)
        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        dudx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
        return dudx
    
    @staticmethod
    def compute_d2x(u: np.ndarray, dx: float) -> np.ndarray:
        """空間2階微分の計算"""
        d2udx2 = np.zeros_like(u)
        d2udx2[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dx ** 2)
        d2udx2[:, 0] = (u[:, 2] - 2 * u[:, 1] + u[:, 0]) / (dx ** 2)
        d2udx2[:, -1] = (u[:, -1] - 2 * u[:, -2] + u[:, -3]) / (dx ** 2)
        return d2udx2


class ComplexityCalculator:
    """モデル複雑度計算クラス"""
    
    def __init__(self):
        self.term_complexity = {
            'constant': 1,
            'linear': 1,
            'quadratic': 2,
            'interaction': 2,
            'derivative_1': 1,
            'derivative_2': 2
        }
    
    def calculate_pde_complexity(self, formula_name: str) -> int:
        """PDE式の複雑度を計算"""
        complexity = 0
        
        if '∂u/∂t' in formula_name or 'dudt' in formula_name:
            complexity += self.term_complexity['derivative_1']
        
        if '∂²u/∂x²' in formula_name or 'd2udx2' in formula_name:
            complexity += self.term_complexity['derivative_2']
        
        if '∂u/∂x' in formula_name or 'dudx' in formula_name:
            complexity += self.term_complexity['derivative_1']
        
        if 'u²' in formula_name or 'u**2' in formula_name:
            complexity += self.term_complexity['quadratic']
        elif 'u' in formula_name and '∂u' not in formula_name:
            complexity += self.term_complexity['linear']
        
        return max(complexity, 1)


class FullStateSearch:
    """完全状態探索によるPDE発見"""
    
    def __init__(self, u: np.ndarray, x: np.ndarray, t: np.ndarray):
        self.u = u
        self.x = x
        self.t = t
        self.dx = x[1] - x[0]
        self.dt = t[1] - t[0]
        
        self.derivatives = NumericalDerivatives()
        self.complexity_calc = ComplexityCalculator()
        
        self.dudt = self.derivatives.compute_dt(u, self.dt)
        self.dudx = self.derivatives.compute_dx(u, self.dx)
        self.d2udx2 = self.derivatives.compute_d2x(u, self.dx)
    
    def evaluate_pde_formula(self, formula_func: Callable, params: List[float], 
                           mask: Optional[np.ndarray] = None) -> float:
        """PDE候補式の評価"""
        if mask is None:
            mask = np.ones_like(self.u, dtype=bool)
        
        try:
            residual = formula_func(params)
            mse = np.mean(residual[mask] ** 2)
            return mse
        except:
            return np.inf
    
    def calculate_bic(self, mse: float, n_params: int, n_samples: int) -> float:
        """BIC (Bayesian Information Criterion) の計算"""
        return n_samples * np.log(mse + 1e-10) + n_params * np.log(n_samples)
    
    def calculate_aic(self, mse: float, n_params: int, n_samples: int) -> float:
        """AIC (Akaike Information Criterion) の計算"""
        return n_samples * np.log(mse + 1e-10) + 2 * n_params
    
    def search_diffusion_equation(self, param_range: Tuple[float, float] = (1e-11, 1e-7)) -> Dict[str, Any]:
        """拡散方程式の探索: ∂u/∂t = c × ∂²u/∂x²"""
        
        candidates = []
        n_samples = np.prod(self.u.shape)
        
        mask = (self.u > 0.01) & (np.abs(self.d2udx2) > 1e-6)
        
        def formula_simple(params):
            c = params[0]
            return self.dudt - c * self.d2udx2
        
        try:
            bounds = [param_range]
            result = differential_evolution(
                lambda p: self.evaluate_pde_formula(formula_simple, p, mask),
                bounds, maxiter=1000, seed=42, tol=1e-10
            )
            params = result.x.tolist()
            mse = result.fun
        except:
            params = [(param_range[0] + param_range[1]) / 2]
            mse = self.evaluate_pde_formula(formula_simple, params, mask)
        
        n_params = 1
        bic = self.calculate_bic(mse, n_params, n_samples)
        aic = self.calculate_aic(mse, n_params, n_samples)
        complexity = self.complexity_calc.calculate_pde_complexity('c × ∂²u/∂x²')
        
        candidates.append({
            'name': '∂u/∂t = c × ∂²u/∂x²',
            'formula': 'c × ∂²u/∂x²',
            'optimized_params': params if isinstance(params, list) else params.tolist(),
            'mse': mse,
            'bic': bic,
            'aic': aic,
            'complexity': complexity,
            'n_params': n_params
        })
        
        def formula_with_drift(params):
            c1, c2 = params
            return self.dudt - c1 * self.d2udx2 - c2 * self.dudx
        
        try:
            bounds = [param_range, (-1e-5, 1e-5)]
            result = differential_evolution(
                lambda p: self.evaluate_pde_formula(formula_with_drift, p, mask),
                bounds, maxiter=1000, seed=42
            )
            params = result.x.tolist()
            mse = result.fun
        except:
            params = [(param_range[0] + param_range[1]) / 2, 0.0]
            mse = self.evaluate_pde_formula(formula_with_drift, params, mask)
        
        n_params = 2
        bic = self.calculate_bic(mse, n_params, n_samples)
        aic = self.calculate_aic(mse, n_params, n_samples)
        complexity = self.complexity_calc.calculate_pde_complexity('c1 × ∂²u/∂x² + c2 × ∂u/∂x')
        
        candidates.append({
            'name': '∂u/∂t = c1 × ∂²u/∂x² + c2 × ∂u/∂x',
            'formula': 'c1 × ∂²u/∂x² + c2 × ∂u/∂x',
            'optimized_params': params if isinstance(params, list) else params.tolist(),
            'mse': mse,
            'bic': bic,
            'aic': aic,
            'complexity': complexity,
            'n_params': n_params
        })
        
        def formula_nonlinear(params):
            c1, c2 = params
            return self.dudt - c1 * self.d2udx2 - c2 * self.u * self.dudx
        
        try:
            bounds = [param_range, (-1e-5, 1e-5)]
            result = differential_evolution(
                lambda p: self.evaluate_pde_formula(formula_nonlinear, p, mask),
                bounds, maxiter=1000, seed=42
            )
            params = result.x.tolist()
            mse = result.fun
        except:
            params = [(param_range[0] + param_range[1]) / 2, 0.0]
            mse = self.evaluate_pde_formula(formula_nonlinear, params, mask)
        
        n_params = 2
        bic = self.calculate_bic(mse, n_params, n_samples)
        aic = self.calculate_aic(mse, n_params, n_samples)
        complexity = self.complexity_calc.calculate_pde_complexity('c1 × ∂²u/∂x² + c2 × u × ∂u/∂x')
        
        candidates.append({
            'name': '∂u/∂t = c1 × ∂²u/∂x² + c2 × u × ∂u/∂x',
            'formula': 'c1 × ∂²u/∂x² + c2 × u × ∂u/∂x',
            'optimized_params': params if isinstance(params, list) else params.tolist(),
            'mse': mse,
            'bic': bic,
            'aic': aic,
            'complexity': complexity,
            'n_params': n_params
        })
        
        candidates_sorted = sorted(candidates, key=lambda x: x['bic'])
        
        bic_values = np.array([c['bic'] for c in candidates_sorted])
        bic_min = np.min(bic_values)
        delta_bic = bic_values - bic_min
        weights = np.exp(-0.5 * delta_bic)
        posterior_probs = weights / np.sum(weights)
        
        for i, candidate in enumerate(candidates_sorted):
            candidate['posterior_prob'] = posterior_probs[i]
        
        best_model = candidates_sorted[0]
        
        return {
            'best_model': best_model,
            'all_candidates': candidates_sorted,
            'posterior_probabilities': posterior_probs.tolist()
        }


def create_app():
    """Streamlitアプリケーションの作成"""
    
    st.set_page_config(page_title="拡散方程式発見システム", layout="wide")
    
    st.title("🔬 拡散方程式発見システム")
    
    if not PINNS_AVAILABLE:
        st.warning("⚠️ PyTorchがインストールされていません。PINNs機能を使用するにはPyTorchをインストールしてください。")
    st.markdown("### Physics-Informed Neural Networks (PINNs) と Bayesian Information Criterion (BIC)")
    
    st.markdown("""
    このアプリケーションは、以下の手順で拡散方程式のパラメータを発見します：
    
    1. **📊 数値データ生成**: 有限差分法(FDM)で拡散方程式を解く
    2. **🔍 完全状態探索**: 候補式を全探索しBICで評価
    3. **🧠 PINNs検証** (オプション): 発見した式をPINNsで検証
    
    **対象方程式**: ∂u/∂t = D × ∂²u/∂x² (拡散方程式)
    """)
    
    st.sidebar.header("⚙️ パラメータ設定")
    
    st.sidebar.subheader("物理パラメータ")
    D_true = st.sidebar.number_input(
        "拡散係数 D [m²/s]",
        min_value=1e-11,
        max_value=1e-7,
        value=1e-9,
        format="%.2e"
    )
    
    L = st.sidebar.number_input(
        "空間領域の長さ L [m]",
        min_value=0.001,
        max_value=1.0,
        value=0.02,
        format="%.3f"
    )
    
    T_final = st.sidebar.number_input(
        "計算時間 T [s]",
        min_value=100.0,
        max_value=10000.0,
        value=3600.0,
        step=100.0
    )
    
    st.sidebar.subheader("数値計算パラメータ")
    nx = st.sidebar.slider("空間グリッド数", 20, 100, 50)
    nt = st.sidebar.slider("時間ステップ数", 50, 200, 100)
    
    noise_level = st.sidebar.slider(
        "ノイズレベル",
        0.0,
        0.1,
        0.01,
        0.005,
        help="データに加えるガウスノイズの標準偏差"
    )
    
    if st.button("🚀 拡散方程式発見を開始", type="primary"):
        
        st.header("Step 1: 数値データ生成 (FDM)")
        
        with st.spinner("有限差分法で拡散方程式を解いています..."):
            fdm = DiffusionFDM(L=L, T_final=T_final, nx=nx, nt=nt, D=D_true)
            u_numerical = fdm.solve()
            
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, u_numerical.shape)
                u_numerical = u_numerical + noise
        
        st.success("✅ 数値解の計算完了")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        im = ax1.imshow(u_numerical, aspect='auto', origin='lower',
                       extent=[0, L, 0, T_final], cmap='viridis')
        ax1.set_xlabel('Position x (m)')
        ax1.set_ylabel('Time t (s)')
        ax1.set_title('FDM: Concentration Distribution c(x,t)')
        plt.colorbar(im, ax=ax1, label='Concentration c')
        
        time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
        for i in time_indices:
            ax2.plot(fdm.x, u_numerical[i, :], label=f't = {fdm.t[i]:.0f}s', linewidth=2)
        ax2.set_xlabel('Position x (m)')
        ax2.set_ylabel('Concentration c')
        ax2.set_title('Concentration Distribution at Different Times')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.header("Step 2: 完全状態探索とBICモデル選択")
        
        with st.spinner("候補式を探索中..."):
            searcher = FullStateSearch(u_numerical, fdm.x, fdm.t)
            search_results = searcher.search_diffusion_equation(param_range=(1e-11, 1e-7))
        
        st.session_state['search_results'] = search_results
        st.session_state['L'] = L
        st.session_state['T_final'] = T_final
        st.session_state['D_true'] = D_true
        
        st.success("✅ 探索完了")
        
        st.subheader("🏆 最優秀モデル (BIC基準)")
        best_model = search_results['best_model']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("モデル", best_model['name'])
        with col2:
            st.metric("MSE", f"{best_model['mse']:.2e}")
        with col3:
            st.metric("BIC", f"{best_model['bic']:.1f}")
        with col4:
            st.metric("事後確率", f"{best_model['posterior_prob']:.3f}")
        
        st.info(f"**最適パラメータ**: {', '.join([f'{p:.2e}' for p in best_model['optimized_params']])}")
        
        st.subheader("📊 全候補モデルの比較")
        
        results_df = pd.DataFrame([
            {
                'モデル': c['name'],
                'MSE': f"{c['mse']:.2e}",
                'BIC': f"{c['bic']:.1f}",
                'AIC': f"{c['aic']:.1f}",
                '事後確率': f"{c['posterior_prob']:.3f}",
                '複雑度': c['complexity'],
                'パラメータ数': c['n_params']
            }
            for c in search_results['all_candidates']
        ])
        
        st.dataframe(results_df, use_container_width=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = [c['name'][:30] + '...' if len(c['name']) > 30 else c['name'] 
                 for c in search_results['all_candidates']]
        bic_scores = [c['bic'] for c in search_results['all_candidates']]
        
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(models))]
        ax1.barh(models, bic_scores, color=colors)
        ax1.set_xlabel('BIC Score (lower is better)')
        ax1.set_title('BIC Comparison')
        ax1.invert_yaxis()
        
        posterior_probs = [c['posterior_prob'] for c in search_results['all_candidates']]
        colors = ['green' if i == 0 else 'coral' for i in range(len(models))]
        ax2.barh(models, posterior_probs, color=colors)
        ax2.set_xlabel('Posterior Probability')
        ax2.set_title('Model Posterior Probabilities')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("📈 理論値との比較")
        
        discovered_D = best_model['optimized_params'][0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("真の拡散係数 D", f"{D_true:.2e} m²/s")
        with col2:
            st.metric("発見された拡散係数", f"{discovered_D:.2e} m²/s")
        with col3:
            error = abs(discovered_D - D_true) / D_true * 100
            st.metric("相対誤差", f"{error:.1f}%")
    
    if PINNS_AVAILABLE and 'search_results' in st.session_state:
        search_results = st.session_state['search_results']
        best_model = search_results['best_model']
        L = st.session_state['L']
        T_final = st.session_state['T_final']
        D_true = st.session_state['D_true']
        
        if True:
            st.write("---")
            st.write("## 🧠 オプション: PINNsによる検証")
            st.markdown("""
            シンボリック回帰で発見された拡散係数を使用して、PINNsでも同じ方程式を解き、
            結果を比較します。これにより、発見された方程式の妥当性を検証できます。
            """)
            
            if st.button("🧠 PINNsで検証"):
                if best_model and len(best_model['optimized_params']) > 0:
                    discovered_D = best_model['optimized_params'][0]
                    
                    st.subheader("🧠 PINNsによる拡散方程式の解法")
                    st.info(f"発見された拡散係数 D = {discovered_D:.2e} を使用してPINNsを訓練します")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(epoch, total_epochs, loss):
                        progress = epoch / total_epochs
                        progress_bar.progress(progress)
                        status_text.text(f"エポック {epoch}/{total_epochs}, 損失: {loss:.6f}")
                    
                    epochs = 5000
                    hidden_dim = 50
                    num_layers = 4
                    learning_rate = 0.0005
                    n_points = 2000
                    
                    with st.spinner("PINNsによる拡散方程式の解法中..."):
                        solver = PINNsDiffusionSolver(
                            D=discovered_D, 
                            L=L,
                            T_final=T_final,
                            hidden_dim=hidden_dim, 
                            num_layers=num_layers
                        )
                        training_results = solver.train(
                            epochs=epochs, 
                            lr=learning_rate, 
                            n_points=n_points, 
                            progress_callback=progress_callback
                        )
                        
                        x_test = np.linspace(0, L, 50)
                        t_test = np.linspace(0, T_final, 50)
                        X_test, T_test = np.meshgrid(x_test, t_test)
                        u_pinns = solver.predict(X_test, T_test)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("✅ PINNs訓練完了!")
                    
                    st.subheader("📈 PINNs拡散方程式の解")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    im1 = ax1.imshow(u_pinns, aspect='auto', origin='lower', 
                                   extent=[0, L, 0, T_final], cmap='plasma')
                    ax1.set_xlabel('Position x (m)')
                    ax1.set_ylabel('Time t (s)')
                    ax1.set_title('PINNs: Concentration Distribution c(x,t)')
                    plt.colorbar(im1, ax=ax1, label='Concentration c')
                    
                    time_indices = [0, 12, 25, 37, 49]
                    for i in time_indices:
                        ax2.plot(x_test, u_pinns[i, :], label=f't = {t_test[i]:.0f}s', linewidth=2)
                    ax2.set_xlabel('Position x (m)')
                    ax2.set_ylabel('Concentration c')
                    ax2.set_title('Concentration Distribution at Different Times')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.subheader("🎯 PINNs訓練結果")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最終損失", f"{training_results['final_loss']:.6f}")
                    with col2:
                        st.metric("PDE損失", f"{training_results['pde_loss']:.6f}")
                    with col3:
                        st.metric("境界条件損失", f"{training_results['bc_loss']:.6f}")
                    
                    st.subheader("📊 シンボリック回帰 vs PINNs 比較")
                    st.markdown("""
                    **シンボリック回帰**: データから数式を直接発見（BIC基準で最適モデル選択）  
                    **PINNs**: 物理法則を制約として組み込んだニューラルネットワーク
                    """)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**シンボリック回帰で発見された式**")
                        st.write(best_model['name'])
                        st.metric("発見された拡散係数", f"{discovered_D:.2e}")
                        st.metric("真の拡散係数", f"{D_true:.2e}")
                        error = abs(discovered_D - D_true) / D_true * 100
                        st.metric("相対誤差", f"{error:.1f}%")
                    
                    with col2:
                        st.write("**PINNsによる検証**")
                        st.write("∂u/∂t = D × ∂²u/∂x² を解いた結果")
                        st.metric("使用した拡散係数", f"{discovered_D:.2e}")
                        st.metric("PDE損失（フィッティング精度）", f"{training_results['pde_loss']:.6f}")
                        
                        if training_results['pde_loss'] < 0.01:
                            st.success("✅ 高精度でPDEを満たしています")
                        elif training_results['pde_loss'] < 0.1:
                            st.info("ℹ️ 妥当な精度でPDEを満たしています")
                        else:
                            st.warning("⚠️ PDE損失が高いです。訓練パラメータの調整が必要かもしれません")
                    
                else:
                    st.error("❌ シンボリック回帰による最適モデルが見つかりません。先に拡散方程式発見を実行してください。")


if __name__ == "__main__":
    create_app()
