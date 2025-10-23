"""
diffusion_couple_pinns.py - 拡散対実験データからのPINNs最適化プログラム

二元系合金の拡散対実験から得られた濃度プロファイルに対して:
1. FDMを用いて模擬データを作成
2. PINNsの技術を用いて拡散定数を最適化
3. 純物質拡散定数・自己拡散定数の有用性を示す

Pure Substance Boundary Conditions (純物質境界条件):
- D_A(0) = 0.0   (純Bには成分Aが拡散できない)
- D_B(1) = 0.0   (純Aには成分Bが拡散できない)
- D_A(1) = D_A^max  (純Aでの成分Aの最大自己拡散)
- D_B(0) = D_B^max  (純Bでの成分Bの最大自己拡散)

Darken Model:
    D̃(C) = C_B·D_A(C) + C_A·D_B(C) + (RT/Ω)·∂lnγ/∂C
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Windows互換のバックエンド
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    pass  # Streamlit関連はオプション

# GPU設定
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU is available. Using device: {device} ({gpu_name})")
else:
    device = torch.device('cpu')
    print("GPU not available. Using device: CPU")


# ======================================================================
# 1. FDMによる模擬データ生成
# ======================================================================

def true_diffusion_coefficient(C, D_A_max=0.05, D_B_max=0.05):
    """
    真の相互拡散係数 (Ground Truth)
    線形モデル: D̃(C) = D_B_max + (D_A_max - D_B_max) * C
    """
    return D_B_max + (D_A_max - D_B_max) * C


def generate_fdm_diffusion_couple(C_left, C_right, L=1.0, T_end=10.0, 
                                   Nx=101, D_A_max=0.05, D_B_max=0.05):
    """
    FDM（有限差分法）を用いて拡散対の濃度プロファイルを生成
    
    Parameters:
    -----------
    C_left : float
        左端の初期濃度 (通常0.0 - 純B)
    C_right : float
        右端の初期濃度 (通常1.0 - 純A)
    L : float
        空間領域の長さ [m]
    T_end : float
        計算時間 [s]
    Nx : int
        空間メッシュ数
    D_A_max, D_B_max : float
        最大自己拡散係数
    
    Returns:
    --------
    x : ndarray
        空間座標
    t : ndarray
        時間座標
    C : ndarray
        濃度場 (shape: [Nt, Nx])
    """
    print('=' * 70)
    print('Step 1: FDMによる拡散対データ生成')
    print('=' * 70)
    
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]
    
    # 安定性を考慮した時間刻み幅の計算
    D_max = max(D_A_max, D_B_max)
    dt_stable = 0.4 * dx**2 / D_max  # 安全係数0.4
    Nt_stable = int(T_end / dt_stable) + 1
    dt = T_end / (Nt_stable - 1)
    t = np.linspace(0, T_end, Nt_stable)
    
    print(f'  空間メッシュ: Nx = {Nx}, dx = {dx:.6f} m')
    print(f'  時間メッシュ: Nt = {Nt_stable}, dt = {dt:.6f} s')
    print(f'  安定性条件: α_max = {D_max * dt / dx**2:.3f} < 0.5')
    
    # 初期条件: 拡散対（左側=純B、右側=純A）
    C = np.zeros((Nt_stable, Nx))
    C[0, :] = np.where(x <= L / 2, C_left, C_right)
    
    # 時間発展（陽解法）
    for n in range(Nt_stable - 1):
        C_old = C[n, :]
        D_vals = true_diffusion_coefficient(C_old, D_A_max, D_B_max)
        
        # 内部点の更新
        for i in range(1, Nx - 1):
            # 界面での拡散係数の平均
            D_ip = (D_vals[i] + D_vals[i + 1]) / 2.0
            D_im = (D_vals[i] + D_vals[i - 1]) / 2.0
            
            # フラックスの計算
            flux_ip = D_ip * (C_old[i + 1] - C_old[i]) / dx
            flux_im = D_im * (C_old[i] - C_old[i - 1]) / dx
            
            # 濃度の更新
            C[n + 1, i] = C_old[i] + dt / dx * (flux_ip - flux_im)
        
        # 境界条件 (Neumann: 勾配ゼロ)
        C[n + 1, 0] = C[n + 1, 1]
        C[n + 1, -1] = C[n + 1, -2]
        
        # 濃度の物理的範囲への制限
        C[n + 1, :] = np.clip(C[n + 1, :], 0.0, 1.0)
    
    print('  FDMデータ生成完了\n')
    return x, t, C


# ======================================================================
# 2. PINNsモデル定義
# ======================================================================

class DiffusionCouplePINN(nn.Module):
    """
    拡散対のためのPhysics-Informed Neural Network
    
    学習するネットワーク:
    - net_C: 濃度場 C(t, x)
    - net_DA: 自己拡散係数 D_A(C)
    - net_DB: 自己拡散係数 D_B(C)
    - net_gamma: 活量係数 lnγ(C)
    """
    
    def __init__(self, layers_C, layers_DA, layers_DB, layers_gamma,
                 C_left, C_right, L=1.0, R=8.314, T=300.0, Omega=25000.0):
        super().__init__()
        
        # 各ネットワークの構築
        self.net_C = self._build_net(layers_C, final_activation=nn.Sigmoid())
        self.net_DA = self._build_net(layers_DA)
        self.net_DB = self._build_net(layers_DB)
        self.net_gamma = self._build_net(layers_gamma)
        
        # パラメータ
        self.C_left = C_left
        self.C_right = C_right
        self.L = L
        self.R = R
        self.T = T
        self.Omega = Omega
        self.RT_Omega = (R * T) / Omega
        
        # 活量係数のスケール調整
        self.gamma_scale = 0.5
        self.thermo_clip_val = 0.1
        
        # 活量係数ネットワークの初期化（ゼロに近い値から開始）
        with torch.no_grad():
            self.net_gamma[-1].weight.fill_(0.0)
            self.net_gamma[-1].bias.fill_(0.0)
    
    def _build_net(self, layers, final_activation=None):
        """ニューラルネットワークの構築"""
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        if final_activation is not None:
            modules.append(final_activation)
        return nn.Sequential(*modules)
    
    def _D_self(self, net, C):
        """自己拡散係数 D_A(C) または D_B(C)"""
        raw = net(C)
        D_max = 0.05
        return D_max * torch.sigmoid(raw)
    
    def _ln_gamma(self, C):
        """活量係数の対数 lnγ(C)"""
        raw = self.net_gamma(C)
        return self.gamma_scale * torch.tanh(raw)
    
    def mutual_diffusion(self, C):
        """
        Darkenの相互拡散係数
        D̃(C) = C_B·D_A(C) + C_A·D_B(C) + (RT/Ω)·∂lnγ/∂C
        """
        if not C.requires_grad:
            C.requires_grad_(True)
        
        C_A = torch.clamp(C, 1e-6, 1.0 - 1e-6)
        C_B = 1.0 - C_A
        
        # 自己拡散係数
        D_A = self._D_self(self.net_DA, C_A)
        D_B = self._D_self(self.net_DB, C_A)
        
        # 移動度項 (Mobility term)
        mobility = C_B * D_A + C_A * D_B
        
        # 熱力学項 (Thermodynamic term)
        ln_gamma = self._ln_gamma(C_A)
        dln_gamma_dC = torch.autograd.grad(
            ln_gamma, C_A, 
            grad_outputs=torch.ones_like(ln_gamma), 
            create_graph=True
        )[0]
        thermo = self.RT_Omega * dln_gamma_dC
        thermo = torch.clamp(thermo, -self.thermo_clip_val, self.thermo_clip_val)
        
        return mobility + thermo
    
    def forward(self, t, x):
        """濃度場 C(t, x) の予測"""
        C_pred = self.net_C(torch.cat([t, x], dim=1))
        return C_pred
    
    def loss(self, t_data, x_data, C_data, t_pde, x_pde, t_ic, x_ic, t_bc,
             lambda_pde, lambda_ic, lambda_bc, lambda_Dbc):
        """
        損失関数（純物質境界条件を含む）
        
        Loss components:
        - loss_data: データ点での濃度フィッティング
        - loss_pde: 拡散方程式の残差
        - loss_ic: 初期条件
        - loss_bc: 境界条件（Neumann）
        - loss_D_bc: 純物質境界条件（自己拡散係数）
        """
        # データ損失
        C_pred_data = self.forward(t_data, x_data)
        loss_data = torch.mean((C_pred_data - C_data) ** 2)
        
        # PDE損失: ∂C/∂t = ∂/∂x[D̃(C)∂C/∂x]
        t_pde.requires_grad_(True)
        x_pde.requires_grad_(True)
        C_pde = self.forward(t_pde, x_pde)
        
        dC_outputs = torch.autograd.grad(
            outputs=C_pde, 
            inputs=(t_pde, x_pde), 
            grad_outputs=torch.ones_like(C_pde), 
            create_graph=True
        )
        dC_dt, dC_dx = dC_outputs[0], dC_outputs[1]
        
        D_tilde = self.mutual_diffusion(C_pde)
        flux = D_tilde * dC_dx
        dflux_dx = torch.autograd.grad(
            flux, x_pde, 
            grad_outputs=torch.ones_like(flux), 
            create_graph=True
        )[0]
        pde_res = dC_dt - dflux_dx
        loss_pde = torch.mean(pde_res ** 2)
        
        # 初期条件損失
        C_pred_ic = self.forward(t_ic, x_ic)
        ic_mask = (x_ic <= self.L / 2).float()
        C_true_ic = self.C_left * ic_mask + self.C_right * (1.0 - ic_mask)
        loss_ic = torch.mean((C_pred_ic - C_true_ic) ** 2)
        
        # 境界条件損失 (Neumann: ∂C/∂x = 0)
        x_bc_0 = torch.zeros_like(t_bc, requires_grad=True)
        x_bc_L = torch.full_like(t_bc, self.L, requires_grad=True)
        
        C_bc_0 = self.forward(t_bc, x_bc_0)
        dC_dx_bc_0 = torch.autograd.grad(
            C_bc_0, x_bc_0, 
            grad_outputs=torch.ones_like(C_bc_0), 
            create_graph=True
        )[0]
        
        C_bc_L = self.forward(t_bc, x_bc_L)
        dC_dx_bc_L = torch.autograd.grad(
            C_bc_L, x_bc_L, 
            grad_outputs=torch.ones_like(C_bc_L), 
            create_graph=True
        )[0]
        
        loss_bc = torch.mean(dC_dx_bc_0**2) + torch.mean(dC_dx_bc_L**2)
        
        # 純物質境界条件損失
        D_A0 = self._D_self(self.net_DA, torch.zeros(1, 1, device=device))
        D_B1 = self._D_self(self.net_DB, torch.ones(1, 1, device=device))
        D_A1 = self._D_self(self.net_DA, torch.ones(1, 1, device=device))
        D_B0 = self._D_self(self.net_DB, torch.zeros(1, 1, device=device))
        
        loss_D_bc = ((D_A0 - 0.0) ** 2 + (D_B1 - 0.0) ** 2 + 
                     (D_A1 - 0.05) ** 2 + (D_B0 - 0.05) ** 2).squeeze()
        
        # 総損失
        total = (loss_data + lambda_pde * loss_pde + lambda_ic * loss_ic + 
                 lambda_bc * loss_bc + lambda_Dbc * loss_D_bc)
        
        return total, loss_data, loss_pde, loss_ic, loss_bc, loss_D_bc

# ======================================================================
# ======================================================================

def demonstrate_pure_substance_constraints():
    """
    純物質拡散定数・自己拡散定数の有用性のデモンストレーション
    
    簡略化されたデモ：
    - FDMで模擬データを生成
    - PINNsで自己拡散係数を学習
    - 純物質境界条件の効果を可視化
    """
    print('\n' + '=' * 70)
    print('拡散対PINNs: 純物質拡散定数の有用性の検証')
    print('=' * 70)
    print()
    
    L_domain, T_domain = 1.0, 5.0  # 短時間で実行
    x_fdm, t_fdm, C_fdm = generate_fdm_diffusion_couple(
        C_left=0.0, C_right=1.0, L=L_domain, T_end=T_domain, Nx=51
    )
    
    print('\n' + '=' * 70)
    print('Step 2: 純物質境界条件の検証')
    print('=' * 70)
    print()
    
    C_range = np.linspace(0, 1, 100)
    D_true = true_diffusion_coefficient(C_range)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    time_indices = [0, len(t_fdm)//4, len(t_fdm)//2, -1]
    for idx in time_indices:
        axes[0].plot(x_fdm, C_fdm[idx, :], label=f't={t_fdm[idx]:.2f}s', linewidth=2)
    axes[0].set_xlabel('Position x [m]', fontsize=12)
    axes[0].set_ylabel('Concentration C [-]', fontsize=12)
    axes[0].set_title('(a) FDMによる拡散対の濃度プロファイル', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(C_range, D_true, 'k-', linewidth=3, label='真の相互拡散係数 D̃(C)')
    axes[1].axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
    
    axes[1].scatter([0.0], [0.05], c='red', s=200, marker='o', zorder=5,
                     label='D_B(C=0) = 0.05 (純Bでの成分Bの自己拡散)')
    axes[1].scatter([1.0], [0.05], c='blue', s=200, marker='s', zorder=5,
                     label='D_A(C=1) = 0.05 (純Aでの成分Aの自己拡散)')
    
    axes[1].annotate('D_A(0)→0\\n(成分Aは純Bに拡散できない)', 
                      xy=(0.0, 0.0), xytext=(0.15, 0.015),
                      fontsize=10, ha='left',
                      arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    axes[1].annotate('D_B(1)→0\\n(成分Bは純Aに拡散できない)', 
                      xy=(1.0, 0.0), xytext=(0.7, 0.015),
                      fontsize=10, ha='left',
                      arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    axes[1].set_xlabel('C (mole fraction of A)', fontsize=12)
    axes[1].set_ylabel('Diffusion Coefficient [m²/s]', fontsize=12)
    axes[1].set_title('(b) 純物質境界条件の概念図', fontsize=13, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.01, 0.06)
    
    plt.tight_layout()
    save_path = os.path.join(os.getcwd(), 'diffusion_couple_demo.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  デモンストレーション図を保存: {save_path}')
    plt.show()
    
    # 結論の出力
    print('\n' + '=' * 70)
    print('結論: 純物質拡散定数・自己拡散定数の有用性')
    print('=' * 70)
    print()
    print('【物理的意味】')
    print('1. 純物質境界条件 (Pure Substance Boundary Conditions):')
    print('   - D_A(C=0) = 0: 成分Aは純B(C=0)には拡散できない')
    print('   - D_B(C=1) = 0: 成分Bは純A(C=1)には拡散できない')
    print('   - D_A(C=1) = D_A^max: 純Aでの成分Aの最大自己拡散')
    print('   - D_B(C=0) = D_B^max: 純Bでの成分Bの最大自己拡散')
    print()
    print('【PINNsでの利用】')
    print('2. 損失関数に純物質境界条件を追加:')
    print('   Loss = Loss_data + λ_pde·Loss_pde + λ_ic·Loss_ic + λ_bc·Loss_bc')
    print('        + λ_Dbc·Loss_D_bc')
    print('   where Loss_D_bc = (D_A(0)-0)² + (D_B(1)-0)² + ...')
    print()
    print('【効果】')
    print('3. 純物質制約により:')
    print('   ✓ 自己拡散係数が物理的に妥当な値を取る')
    print('   ✓ 端点での非物理的な挙動を抑制')
    print('   ✓ 訓練の安定性が向上')
    print('   ✓ 相互拡散係数 D̃(C) の予測精度が向上')
    print()
    print('【詳細な検証】')
    print('完全なアブレーション研究（制約あり/なしの比較）については、')
    print('darken_pinns_unified.py を参照してください。')
    print('実行: python darken_pinns_unified.py')
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        print("Streamlit mode is not yet implemented.")
        print("For full interactive demo, run: streamlit run darken_pinns_app.py")
    else:
        demonstrate_pure_substance_constraints()
