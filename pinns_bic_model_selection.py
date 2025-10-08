"""
拡散方程式の拡散係数発見システム - PINNsによる逆問題解法
Physics-Informed Neural Networks (PINNs) を用いた拡散係数の推定

FDMで生成した疑似実験データから、PINNsが拡散係数Dを逆問題として求めます。
完全な自己完結型ファイル
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Callable, Optional
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
        """PINNsを用いた拡散方程式の逆問題ソルバー
        
        ∂u/∂t = D × ∂²u/∂x²
        
        データから拡散係数Dを推定します。
        """
        
        def __init__(self, L: float = 0.02, T_final: float = 3600.0,
                     hidden_dim: int = 50, num_layers: int = 4, D_init: float = 1e-9):
            """
            Args:
                L: 空間領域の長さ [m]
                T_final: 最終時間 [s]
                hidden_dim: 隠れ層の次元
                num_layers: 隠れ層の数
                D_init: 拡散係数の初期値 [m²/s]
            """
            self.L = L
            self.T_final = T_final
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = DiffusionPINN(hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
            
            log_D_init = np.log(D_init)
            self.log_D = nn.Parameter(torch.tensor([log_D_init], dtype=torch.float32, device=self.device))
            
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
            
            D = torch.exp(self.log_D)
            residual = u_t - D * u_xx
            return residual
        
        def train(self, epochs: int = 5000, lr: float = 0.001, n_points: int = 2000,
                 u_data: np.ndarray = None, x_data: np.ndarray = None, t_data: np.ndarray = None,
                 progress_callback: Optional[Callable] = None) -> Dict[str, float]:
            """PINNsの訓練（逆問題：Dを発見）
            
            Args:
                epochs: エポック数
                lr: 学習率
                n_points: PDE残差計算用サンプリング点数
                u_data: FDMから得た濃度データ
                x_data: 空間座標データ
                t_data: 時間座標データ
                progress_callback: 進捗コールバック関数
            
            Returns:
                訓練結果の辞書（discovered_D含む）
            """
            optimizer = optim.Adam(list(self.model.parameters()) + [self.log_D], lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)
            
            if u_data is not None:
                x_data_tensor = torch.FloatTensor(x_data).to(self.device)
                t_data_tensor = torch.FloatTensor(t_data).to(self.device)
                u_data_tensor = torch.FloatTensor(u_data).to(self.device)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                if u_data is not None:
                    u_pred = self.model(x_data_tensor, t_data_tensor)
                    data_loss = torch.mean((u_pred - u_data_tensor) ** 2)
                else:
                    data_loss = torch.tensor(0.0, device=self.device)
                
                x_pde = torch.rand(n_points, 1, device=self.device) * self.L
                t_pde = torch.rand(n_points, 1, device=self.device) * self.T_final
                pde_loss = torch.mean(self.pde_residual(x_pde, t_pde) ** 2)
                
                x_ic = torch.rand(n_points // 4, 1, device=self.device) * self.L
                t_ic = torch.zeros(n_points // 4, 1, device=self.device)
                u_ic_pred = self.model(x_ic, t_ic)
                u_ic_true = self.initial_condition(x_ic)
                ic_loss = torch.mean((u_ic_pred - u_ic_true) ** 2)
                
                t_bc = torch.rand(n_points // 4, 1, device=self.device) * self.T_final
                x_bc_left = torch.zeros(n_points // 4, 1, device=self.device)
                x_bc_right = torch.ones(n_points // 4, 1, device=self.device) * self.L
                
                u_bc_left_pred = self.model(x_bc_left, t_bc)
                u_bc_left_true = self.boundary_condition_left(t_bc)
                bc_left_loss = torch.mean((u_bc_left_pred - u_bc_left_true) ** 2)
                
                u_bc_right_pred = self.model(x_bc_right, t_bc)
                u_bc_right_true = self.boundary_condition_right(t_bc)
                bc_right_loss = torch.mean((u_bc_right_pred - u_bc_right_true) ** 2)
                
                bc_loss = bc_left_loss + bc_right_loss
                
                loss = data_loss + 0.1 * pde_loss + 10.0 * ic_loss + 10.0 * bc_loss
                
                loss.backward()
                optimizer.step()
                scheduler.step(loss)
                
                if progress_callback is not None and epoch % 100 == 0:
                    progress_callback(epoch, epochs, loss.item())
            
            discovered_D = torch.exp(self.log_D).item()
            
            return {
                'final_loss': loss.item(),
                'data_loss': data_loss.item() if u_data is not None else 0.0,
                'pde_loss': pde_loss.item(),
                'ic_loss': ic_loss.item(),
                'bc_loss': bc_loss.item(),
                'discovered_D': discovered_D
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


def create_app():
    """Streamlitアプリケーションの作成"""
    
    st.set_page_config(page_title="拡散方程式発見システム", layout="wide")
    
    st.title("🔬 拡散係数発見システム")
    
    if not PINNS_AVAILABLE:
        st.warning("⚠️ PyTorchがインストールされていません。PINNs機能を使用するにはPyTorchをインストールしてください。")
    st.markdown("### Physics-Informed Neural Networks (PINNs) による逆問題解法")
    
    st.markdown("""
    このアプリケーションは、以下の手順で拡散係数を発見します：
    
    1. **📊 疑似実験データ生成**: 有限差分法(FDM)で拡散方程式を解く
    2. **🧠 PINNsによる発見**: データから拡散係数Dを逆問題として求める
    
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
    
    if st.button("🚀 拡散係数発見を開始", type="primary"):
        
        st.header("Step 1: 疑似実験データ生成 (FDM)")
        
        with st.spinner("有限差分法で拡散方程式を解いています..."):
            fdm = DiffusionFDM(L=L, T_final=T_final, nx=nx, nt=nt, D=D_true)
            u_numerical = fdm.solve()
            
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, u_numerical.shape)
                u_numerical = u_numerical + noise
        
        st.success("✅ 疑似実験データの生成完了")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        im = ax1.imshow(u_numerical, aspect='auto', origin='lower',
                       extent=[0, L, 0, T_final], cmap='viridis')
        ax1.set_xlabel('Position x (m)')
        ax1.set_ylabel('Time t (s)')
        ax1.set_title('FDM: Pseudo-Experimental Data c(x,t)')
        plt.colorbar(im, ax=ax1, label='Concentration c')
        
        time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
        for i in time_indices:
            ax2.plot(fdm.x, u_numerical[i, :], label=f't = {fdm.t[i]:.0f}s', linewidth=2)
        ax2.set_xlabel('Position x (m)')
        ax2.set_ylabel('Concentration c')
        ax2.set_title('Concentration at Different Times')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.session_state['u_numerical'] = u_numerical
        st.session_state['fdm'] = fdm
        st.session_state['L'] = L
        st.session_state['T_final'] = T_final
        st.session_state['D_true'] = D_true
    
    if PINNS_AVAILABLE and 'u_numerical' in st.session_state:
        st.write("---")
        st.header("Step 2: PINNsによる拡散係数の発見")
        
        st.markdown("""
        疑似実験データ（FDMで生成）を訓練データとして、PINNsが拡散係数Dを逆問題として推定します。
        ニューラルネットワークは、データへのフィッティングと物理法則（PDE、初期条件、境界条件）を
        同時に満たすように訓練されます。
        """)
        
        st.subheader("🎛️ PINNs訓練パラメータ")
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input("訓練エポック数", 1000, 20000, 5000, 1000)
        with col2:
            hidden_dim = st.number_input("隠れ層の次元", 20, 100, 50, 10)
        with col3:
            num_layers = st.number_input("隠れ層の数", 2, 8, 4, 1)
        
        if st.button("🧠 PINNsで拡散係数を発見"):
            u_numerical = st.session_state['u_numerical']
            fdm = st.session_state['fdm']
            L = st.session_state['L']
            T_final = st.session_state['T_final']
            D_true = st.session_state['D_true']
            
            st.subheader("🧠 PINNsによる訓練")
            st.info("データから拡散係数Dを推定しています...")
            
            X_grid, T_grid = np.meshgrid(fdm.x, fdm.t)
            x_data = X_grid.flatten().reshape(-1, 1)
            t_data = T_grid.flatten().reshape(-1, 1)
            u_data = u_numerical.flatten().reshape(-1, 1)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(epoch, total_epochs, loss):
                progress = epoch / total_epochs
                progress_bar.progress(progress)
                status_text.text(f"エポック {epoch}/{total_epochs}, 損失: {loss:.6f}")
            
            D_init = 1e-9
            learning_rate = 0.001
            n_points = 2000
            
            with st.spinner("PINNsによる拡散係数の推定中..."):
                solver = PINNsDiffusionSolver(
                    L=L,
                    T_final=T_final,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    D_init=D_init
                )
                training_results = solver.train(
                    epochs=epochs,
                    lr=learning_rate,
                    n_points=n_points,
                    u_data=u_data,
                    x_data=x_data,
                    t_data=t_data,
                    progress_callback=progress_callback
                )
                
                x_test = np.linspace(0, L, 50)
                t_test = np.linspace(0, T_final, 50)
                X_test, T_test = np.meshgrid(x_test, t_test)
                u_pinns = solver.predict(X_test, T_test)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ PINNs訓練完了！拡散係数を発見しました")
            
            discovered_D = training_results['discovered_D']
            
            st.subheader("🎯 発見された拡散係数")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("真の拡散係数 D", f"{D_true:.2e} m²/s")
            with col2:
                st.metric("発見された拡散係数", f"{discovered_D:.2e} m²/s", 
                         delta=f"{discovered_D - D_true:.2e}")
            with col3:
                error = abs(discovered_D - D_true) / D_true * 100
                st.metric("相対誤差", f"{error:.1f}%")
            
            if error < 10:
                st.success("✅ 高精度で拡散係数を推定できました！")
            elif error < 30:
                st.info("ℹ️ 妥当な精度で拡散係数を推定できました")
            else:
                st.warning("⚠️ 推定精度が低いです。訓練パラメータの調整が必要かもしれません")
            
            st.subheader("📊 訓練結果")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("総損失", f"{training_results['final_loss']:.6f}")
            with col2:
                st.metric("データ損失", f"{training_results['data_loss']:.6f}")
            with col3:
                st.metric("PDE損失", f"{training_results['pde_loss']:.6f}")
            with col4:
                st.metric("初期条件損失", f"{training_results['ic_loss']:.6f}")
            with col5:
                st.metric("境界条件損失", f"{training_results['bc_loss']:.6f}")
            
            st.subheader("📈 FDMデータ vs PINN予測の比較")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            im1 = axes[0, 0].imshow(u_numerical, aspect='auto', origin='lower',
                                   extent=[0, L, 0, T_final], cmap='viridis')
            axes[0, 0].set_xlabel('Position x (m)')
            axes[0, 0].set_ylabel('Time t (s)')
            axes[0, 0].set_title('FDM: Pseudo-Experimental Data')
            plt.colorbar(im1, ax=axes[0, 0], label='Concentration c')
            
            im2 = axes[0, 1].imshow(u_pinns, aspect='auto', origin='lower',
                                   extent=[0, L, 0, T_final], cmap='viridis')
            axes[0, 1].set_xlabel('Position x (m)')
            axes[0, 1].set_ylabel('Time t (s)')
            axes[0, 1].set_title(f'PINNs Prediction (D = {discovered_D:.2e})')
            plt.colorbar(im2, ax=axes[0, 1], label='Concentration c')
            
            time_indices = [0, 12, 25, 37, 49]
            for i in time_indices:
                axes[1, 0].plot(fdm.x, u_numerical[int(i * fdm.nt / 50), :],
                              label=f't = {fdm.t[int(i * fdm.nt / 50)]:.0f}s', linewidth=2, linestyle='--')
                axes[1, 0].plot(x_test, u_pinns[i, :], linewidth=2)
            axes[1, 0].set_xlabel('Position x (m)')
            axes[1, 0].set_ylabel('Concentration c')
            axes[1, 0].set_title('FDM (dashed) vs PINNs (solid)')
            axes[1, 0].grid(True, alpha=0.3)
            
            X_grid_test, T_grid_test = np.meshgrid(fdm.x, fdm.t)
            u_pinns_full = solver.predict(X_grid_test, T_grid_test)
            error_map = np.abs(u_numerical - u_pinns_full)
            
            im3 = axes[1, 1].imshow(error_map, aspect='auto', origin='lower',
                                   extent=[0, L, 0, T_final], cmap='hot')
            axes[1, 1].set_xlabel('Position x (m)')
            axes[1, 1].set_ylabel('Time t (s)')
            axes[1, 1].set_title('Absolute Error |FDM - PINNs|')
            plt.colorbar(im3, ax=axes[1, 1], label='Error')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("📝 まとめ")
            st.markdown(f"""
            **手法**: Physics-Informed Neural Networks (PINNs) による逆問題解法
            
            **入力**: FDMで生成した疑似実験データ（濃度分布 c(x,t)）
            
            **出力**: 拡散係数 D = {discovered_D:.2e} m²/s
            
            **精度**: 相対誤差 {error:.1f}%
            
            **訓練戦略**: 
            - データフィッティング損失（FDMとの一致度）
            - PDE損失（物理法則の満足度）
            - 初期条件・境界条件損失
            
            PINNsは、データと物理法則を同時に学習することで、
            ノイズを含むデータからでも頑健に拡散係数を推定できます。
            """)

    elif not PINNS_AVAILABLE:
        st.info("💡 PINNsライブラリが利用できません。PyTorchをインストールしてPINNs機能を有効にしてください。")


if __name__ == "__main__":
    create_app()
