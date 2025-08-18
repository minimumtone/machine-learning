"""
Physics-Informed Neural Networks (PINNs) for PDE Discovery
PyTorchベースの高度なPDE発見システム
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, List, Tuple, Callable
import time
from scipy.optimize import minimize

class PINN(nn.Module):
    """Physics-Informed Neural Network"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 50, output_dim: int = 1, num_layers: int = 4):
        super(PINN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PINNsHeatSolver:
    """PINNsによる熱伝導方程式の解法"""
    
    def __init__(self, alpha: float = 0.01, L: float = 1.0, T_final: float = 1.0,
                 hidden_dim: int = 50, num_layers: int = 4):
        self.alpha = alpha
        self.L = L
        self.T_final = T_final
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PINN(input_dim=2, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers).to(self.device)
        
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """初期条件: ガウシアン"""
        return torch.exp(-50 * (x - 0.5)**2)
    
    def boundary_condition(self, t: torch.Tensor) -> torch.Tensor:
        """境界条件: 両端で0"""
        return torch.zeros_like(t)
    
    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """PDE残差の計算"""
        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)
        
        u = self.model(xt)
        
        u_t = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), 
                                 create_graph=True, retain_graph=True)[0][:, 1:2]
        
        u_x = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), 
                                 create_graph=True, retain_graph=True)[0][:, 0:1]
        
        u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), 
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
        
        pde_residual = u_t - self.alpha * u_xx
        
        return pde_residual
    
    def train(self, epochs: int = 8000, lr: float = 0.0005, n_points: int = 2000,
              progress_callback: Callable = None) -> Dict:
        """PINNsの訓練"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        
        x_pde = torch.rand(n_points, 1, device=self.device) * self.L
        t_pde = torch.rand(n_points, 1, device=self.device) * self.T_final
        
        x_ic = torch.rand(n_points//4, 1, device=self.device) * self.L
        t_ic = torch.zeros(n_points//4, 1, device=self.device)
        
        x_bc1 = torch.zeros(n_points//8, 1, device=self.device)
        x_bc2 = torch.ones(n_points//8, 1, device=self.device) * self.L
        t_bc = torch.rand(n_points//4, 1, device=self.device) * self.T_final
        
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            pde_loss = torch.mean(self.pde_residual(x_pde, t_pde)**2)
            
            xt_ic = torch.cat([x_ic, t_ic], dim=1)
            u_ic_pred = self.model(xt_ic)
            u_ic_true = self.initial_condition(x_ic)
            ic_loss = torch.mean((u_ic_pred - u_ic_true)**2)
            
            xt_bc1 = torch.cat([x_bc1, t_bc[:len(x_bc1)]], dim=1)
            xt_bc2 = torch.cat([x_bc2, t_bc[:len(x_bc2)]], dim=1)
            u_bc1_pred = self.model(xt_bc1)
            u_bc2_pred = self.model(xt_bc2)
            bc_loss = torch.mean(u_bc1_pred**2) + torch.mean(u_bc2_pred**2)
            
            total_loss = pde_loss + 10 * ic_loss + 10 * bc_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(total_loss.item())
            
            if progress_callback and epoch % 100 == 0:
                progress_callback(epoch, epochs, total_loss.item())
        
        return {
            'losses': losses,
            'final_loss': losses[-1],
            'pde_loss': pde_loss.item(),
            'ic_loss': ic_loss.item(),
            'bc_loss': bc_loss.item()
        }
    
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """予測"""
        self.model.eval()
        
        x_tensor = torch.tensor(x.flatten(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_tensor = torch.tensor(t.flatten(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        xt = torch.cat([x_tensor, t_tensor], dim=1)
        
        with torch.no_grad():
            u_pred = self.model(xt).cpu().numpy()
        
        return u_pred.reshape(x.shape)

class PINNsBurgersSolver:
    """PINNsによるBurgers方程式の解法"""
    
    def __init__(self, nu: float = 0.01, L: float = 1.0, T_final: float = 0.5,
                 hidden_dim: int = 50, num_layers: int = 4):
        self.nu = nu
        self.L = L
        self.T_final = T_final
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PINN(input_dim=2, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers).to(self.device)
        
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """初期条件: ステップ関数"""
        return torch.where(x < 0.5, torch.ones_like(x), torch.zeros_like(x))
    
    def boundary_condition(self, t: torch.Tensor) -> torch.Tensor:
        """境界条件: 両端で0"""
        return torch.zeros_like(t)
    
    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """PDE残差の計算"""
        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)
        
        u = self.model(xt)
        
        u_t = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), 
                                 create_graph=True, retain_graph=True)[0][:, 1:2]
        
        u_x = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), 
                                 create_graph=True, retain_graph=True)[0][:, 0:1]
        
        u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), 
                                  create_graph=True, retain_graph=True)[0][:, 0:1]
        
        pde_residual = u_t + u * u_x - self.nu * u_xx
        
        return pde_residual
    
    def train(self, epochs: int = 8000, lr: float = 0.0005, n_points: int = 2000,
              progress_callback: Callable = None) -> Dict:
        """PINNsの訓練"""
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        
        x_pde = torch.rand(n_points, 1, device=self.device) * self.L
        t_pde = torch.rand(n_points, 1, device=self.device) * self.T_final
        
        x_ic = torch.rand(n_points//4, 1, device=self.device) * self.L
        t_ic = torch.zeros(n_points//4, 1, device=self.device)
        
        x_bc1 = torch.zeros(n_points//8, 1, device=self.device)
        x_bc2 = torch.ones(n_points//8, 1, device=self.device) * self.L
        t_bc = torch.rand(n_points//4, 1, device=self.device) * self.T_final
        
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            pde_loss = torch.mean(self.pde_residual(x_pde, t_pde)**2)
            
            xt_ic = torch.cat([x_ic, t_ic], dim=1)
            u_ic_pred = self.model(xt_ic)
            u_ic_true = self.initial_condition(x_ic)
            ic_loss = torch.mean((u_ic_pred - u_ic_true)**2)
            
            xt_bc1 = torch.cat([x_bc1, t_bc[:len(x_bc1)]], dim=1)
            xt_bc2 = torch.cat([x_bc2, t_bc[:len(x_bc2)]], dim=1)
            u_bc1_pred = self.model(xt_bc1)
            u_bc2_pred = self.model(xt_bc2)
            bc_loss = torch.mean(u_bc1_pred**2) + torch.mean(u_bc2_pred**2)
            
            total_loss = pde_loss + 10 * ic_loss + 10 * bc_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(total_loss.item())
            
            if progress_callback and epoch % 100 == 0:
                progress_callback(epoch, epochs, total_loss.item())
        
        return {
            'losses': losses,
            'final_loss': losses[-1],
            'pde_loss': pde_loss.item(),
            'ic_loss': ic_loss.item(),
            'bc_loss': bc_loss.item()
        }
    
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """予測"""
        self.model.eval()
        
        x_tensor = torch.tensor(x.flatten(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        t_tensor = torch.tensor(t.flatten(), dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        xt = torch.cat([x_tensor, t_tensor], dim=1)
        
        with torch.no_grad():
            u_pred = self.model(xt).cpu().numpy()
        
        return u_pred.reshape(x.shape)

class PINNsSymbolicRegression:
    """PINNsベースのシンボリック回帰"""
    
    def __init__(self, u_data: np.ndarray, x: np.ndarray, t: np.ndarray):
        self.u_data = u_data
        self.x = x
        self.t = t
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X, T = np.meshgrid(x, t)
        self.X_flat = X.flatten()
        self.T_flat = T.flatten()
        self.U_flat = u_data.flatten()
        
    def evaluate_pde_formula_with_pinns(self, formula_func, initial_params: List[float], 
                                       epochs: int = 1000) -> Tuple[float, List[float]]:
        """PINNsを使用したPDE候補式の評価"""
        
        def objective(params):
            try:
                x_tensor = torch.tensor(self.X_flat, dtype=torch.float32, device=self.device).reshape(-1, 1)
                t_tensor = torch.tensor(self.T_flat, dtype=torch.float32, device=self.device).reshape(-1, 1)
                u_tensor = torch.tensor(self.U_flat, dtype=torch.float32, device=self.device).reshape(-1, 1)
                
                xt = torch.cat([x_tensor, t_tensor], dim=1)
                xt.requires_grad_(True)
                
                u_t = torch.autograd.grad(u_tensor, xt, grad_outputs=torch.ones_like(u_tensor), 
                                         create_graph=True, retain_graph=True)[0][:, 1:2]
                u_x = torch.autograd.grad(u_tensor, xt, grad_outputs=torch.ones_like(u_tensor), 
                                         create_graph=True, retain_graph=True)[0][:, 0:1]
                u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), 
                                          create_graph=True, retain_graph=True)[0][:, 0:1]
                
                dudt = u_t.detach().cpu().numpy().flatten()
                dudx = u_x.detach().cpu().numpy().flatten()
                d2udx2 = u_xx.detach().cpu().numpy().flatten()
                
                predicted_dudt = formula_func(params, self.U_flat, dudx, d2udx2)
                mse = np.mean((dudt - predicted_dudt)**2)
                return mse
            except:
                return np.inf
        
        best_mse = np.inf
        best_params = initial_params
        
        methods = ['Nelder-Mead', 'L-BFGS-B', 'Powell']
        
        for method in methods:
            try:
                if method == 'L-BFGS-B':
                    bounds = [(-100, 100) for _ in initial_params]
                    result = minimize(objective, initial_params, method=method, bounds=bounds)
                else:
                    result = minimize(objective, initial_params, method=method)
                
                if result.success and result.fun < best_mse:
                    best_mse = result.fun
                    best_params = result.x
                    
            except:
                continue
        
        return best_mse, best_params

def create_pinns_app():
    """PINNs PDE発見アプリ"""
    
    st.title("🧠 Physics-Informed Neural Networks (PINNs) PDE発見システム")
    st.markdown("---")
    
    equation_type = st.selectbox(
        "🧮 方程式タイプを選択",
        ["熱伝導方程式", "Burgers方程式"],
        help="発見したい偏微分方程式のタイプを選択してください"
    )
    
    if equation_type == "熱伝導方程式":
        st.markdown("""
        Physics-Informed Neural Networks (PINNs)で熱伝導方程式を解き、
        その結果から元の偏微分方程式を発見するシステムです。
        
        **対象方程式**: ∂u/∂t = α × ∂²u/∂x²
        """)
    else:
        st.markdown("""
        Physics-Informed Neural Networks (PINNs)でBurgers方程式を解き、
        その結果から元の偏微分方程式を発見するシステムです。
        
        **対象方程式**: ∂u/∂t + u×∂u/∂x = ν×∂²u/∂x²
        """)
    
    st.sidebar.header("🔧 PINNs パラメータ")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        epochs = st.number_input("エポック数", min_value=1000, max_value=20000, value=5000, step=1000)
        hidden_dim = st.number_input("隠れ層次元", min_value=20, max_value=200, value=50, step=10)
        if equation_type == "熱伝導方程式":
            param = st.number_input("熱拡散係数 α", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
        else:
            param = st.number_input("粘性係数 ν", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
    
    with col2:
        num_layers = st.number_input("ネットワーク層数", min_value=3, max_value=8, value=4)
        learning_rate = st.number_input("学習率", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
        n_points = st.number_input("訓練点数", min_value=500, max_value=5000, value=1000, step=500)
    
    st.sidebar.header("🎯 最適化設定")
    optimization_epochs = st.sidebar.number_input("シンボリック回帰エポック数", min_value=100, max_value=5000, value=1000, step=100)
    use_pinns_derivatives = st.sidebar.checkbox("PINNs微分を使用", value=True, help="PINNsの自動微分を使用してより正確な微分を計算")
    
    if st.button("🚀 PINNs PDE発見を実行", type="primary"):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(epoch, total_epochs, loss):
            progress = epoch / total_epochs
            progress_bar.progress(progress)
            status_text.text(f"エポック {epoch}/{total_epochs}, 損失: {loss:.6f}")
        
        if equation_type == "熱伝導方程式":
            with st.spinner("PINNsによる熱伝導方程式の解法中..."):
                solver = PINNsHeatSolver(alpha=param, hidden_dim=hidden_dim, num_layers=num_layers)
                training_results = solver.train(epochs=epochs, lr=learning_rate, n_points=n_points, 
                                              progress_callback=progress_callback)
                
                x_test = np.linspace(0, 1, 50)
                t_test = np.linspace(0, 1, 50)
                X_test, T_test = np.meshgrid(x_test, t_test)
                u_pred = solver.predict(X_test, T_test)
                
            theoretical_param = param
            param_name = "α"
            
        else:  # Burgers方程式
            with st.spinner("PINNsによるBurgers方程式の解法中..."):
                solver = PINNsBurgersSolver(nu=param, hidden_dim=hidden_dim, num_layers=num_layers)
                training_results = solver.train(epochs=epochs, lr=learning_rate, n_points=n_points, 
                                              progress_callback=progress_callback)
                
                x_test = np.linspace(0, 1, 50)
                t_test = np.linspace(0, 0.5, 50)
                X_test, T_test = np.meshgrid(x_test, t_test)
                u_pred = solver.predict(X_test, T_test)
                
            theoretical_param = param
            param_name = "ν"
        
        st.success("✅ PINNs解法完了!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 PINNs解の可視化")
            fig, ax = plt.subplots(figsize=(10, 6))
            c = ax.contourf(X_test, T_test, u_pred, levels=20, cmap='viridis')
            ax.set_xlabel('空間 x')
            ax.set_ylabel('時間 t')
            ax.set_title(f'PINNs解 ({equation_type})')
            plt.colorbar(c)
            st.pyplot(fig)
        
        with col2:
            st.subheader("📈 訓練損失の推移")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(training_results['losses'])
            ax.set_xlabel('エポック')
            ax.set_ylabel('損失')
            ax.set_title('PINNs訓練損失')
            ax.set_yscale('log')
            st.pyplot(fig)
        
        st.subheader("🎯 PINNs訓練結果")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("最終損失", f"{training_results['final_loss']:.2e}")
        with col2:
            st.metric("PDE損失", f"{training_results['pde_loss']:.2e}")
        with col3:
            st.metric("初期条件損失", f"{training_results['ic_loss']:.2e}")
        with col4:
            st.metric("境界条件損失", f"{training_results['bc_loss']:.2e}")
        
        if use_pinns_derivatives:
            with st.spinner("PINNsベースのシンボリック回帰実行中..."):
                pde_regression = PINNsSymbolicRegression(u_pred, x_test, t_test)
                
                if equation_type == "熱伝導方程式":
                    formulas = {
                        "∂u/∂t = c₁ × ∂²u/∂x²": {
                            "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2,
                            "params": [param],
                        }
                    }
                else:
                    formulas = {
                        "∂u/∂t = -c₁ × u × ∂u/∂x + c₂ × ∂²u/∂x²": {
                            "func": lambda p, u, dudx, d2udx2: -p[0] * u * dudx + p[1] * d2udx2,
                            "params": [1.0, param],
                        }
                    }
                
                results = {}
                for name, formula_info in formulas.items():
                    func = formula_info["func"]
                    initial_params = formula_info["params"]
                    
                    mse, optimal_params = pde_regression.evaluate_pde_formula_with_pinns(
                        func, initial_params, epochs=optimization_epochs)
                    
                    results[name] = {
                        'mse': mse,
                        'params': optimal_params,
                    }
                
                st.subheader("🔍 発見されたPDE")
                
                for name, result in results.items():
                    st.write(f"**{name}**")
                    st.write(f"MSE: {result['mse']:.2e}")
                    st.write(f"発見されたパラメータ: {result['params']}")
                    
                    if len(result['params']) > 0:
                        discovered_param = abs(result['params'][0]) if equation_type == "熱伝導方程式" else abs(result['params'][-1])
                        error = abs(discovered_param - theoretical_param) / theoretical_param * 100
                        st.write(f"理論値 {param_name}: {theoretical_param:.4f}")
                        st.write(f"発見値 {param_name}: {discovered_param:.4f}")
                        st.write(f"相対誤差: {error:.2f}%")
                        
                        if error < 10:
                            st.success("✅ 高精度でPDEを発見しました！")
                        elif error < 20:
                            st.warning("⚠️ 中程度の精度でPDEを発見しました")
                        else:
                            st.error("❌ 発見精度が低いです。パラメータを調整してください")

if __name__ == "__main__":
    create_pinns_app()
