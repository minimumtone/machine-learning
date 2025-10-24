"""
Co拡散解析アプリケーションの使用例
プログラマティックに拡散方程式を解く例
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class SimpleDiffusionSolver:
    """
    簡易拡散方程式ソルバー
    Fick's Second Law: ∂C/∂t = D ∂²C/∂x²
    """
    
    def __init__(self, L=600.0, T_final=72.0, nx=200, nt=500):
        """
        Parameters:
        -----------
        L : float
            領域長さ (μm)
        T_final : float
            最終時間 (h)
        nx : int
            空間分割数
        nt : int
            時間ステップ数
        """
        self.L = L
        self.T_final = T_final
        self.nx = nx
        self.nt = nt
        self.dx = L / (nx - 1)
        self.dt = T_final / (nt - 1)
        self.x = np.linspace(-L/2, L/2, nx)
        self.t = np.linspace(0, T_final, nt)
        
    def solve_constant_D(self, D=10.0, C_left=0.7, C_right=0.0):
        """
        定数拡散係数の場合の解法
        
        Parameters:
        -----------
        D : float
            拡散係数 (μm²/h)
        C_left : float
            左側初期濃度
        C_right : float
            右側初期濃度
            
        Returns:
        --------
        C : ndarray
            濃度分布 (nt, nx)
        """
        C = np.zeros((self.nt, self.nx))
        
        C[0, self.x < 0] = C_left
        C[0, self.x >= 0] = C_right
        
        alpha = D * self.dt / (self.dx**2)
        print(f"安定性パラメータ α = {alpha:.4f} (推奨: < 0.5)")
        
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                C[n+1, i] = C[n, i] + alpha * (C[n, i+1] - 2*C[n, i] + C[n, i-1])
            
            C[n+1, 0] = C[n+1, 1]
            C[n+1, -1] = C[n+1, -2]
        
        return C
    
    def solve_concentration_dependent_D(self, D0=5.0, D1=15.0, C_left=0.7, C_right=0.0):
        """
        濃度依存拡散係数の場合の解法
        D(C) = D0 + (D1 - D0) * C
        
        Parameters:
        -----------
        D0 : float
            最小拡散係数 (μm²/h)
        D1 : float
            最大拡散係数 (μm²/h)
        C_left : float
            左側初期濃度
        C_right : float
            右側初期濃度
            
        Returns:
        --------
        C : ndarray
            濃度分布 (nt, nx)
        """
        C = np.zeros((self.nt, self.nx))
        
        C[0, self.x < 0] = C_left
        C[0, self.x >= 0] = C_right
        
        for n in range(self.nt - 1):
            C_old = C[n, :].copy()
            
            for i in range(1, self.nx - 1):
                D_i = D0 + (D1 - D0) * C_old[i]
                
                dC_dx = (C_old[i+1] - C_old[i-1]) / (2 * self.dx)
                d2C_dx2 = (C_old[i+1] - 2*C_old[i] + C_old[i-1]) / (self.dx**2)
                
                dD_dC = D1 - D0
                
                dC_dt = D_i * d2C_dx2 + dD_dC * dC_dx**2
                
                C[n+1, i] = C_old[i] + self.dt * dC_dt
            
            C[n+1, 0] = C[n+1, 1]
            C[n+1, -1] = C[n+1, -2]
        
        return C


def example_1_constant_diffusion():
    """
    例1: 定数拡散係数
    Co-Ni拡散対を模擬
    """
    print("=" * 60)
    print("例1: 定数拡散係数による拡散")
    print("=" * 60)
    
    solver = SimpleDiffusionSolver(L=600.0, T_final=72.0, nx=200, nt=500)
    C = solver.solve_constant_D(D=10.0, C_left=0.7, C_right=0.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    time_indices = [0, len(solver.t)//4, len(solver.t)//2, len(solver.t)-1]
    for idx in time_indices:
        axes[0].plot(solver.x, C[idx, :], label=f't = {solver.t[idx]:.1f} h')
    
    axes[0].set_xlabel('位置 (μm)', fontsize=12)
    axes[0].set_ylabel('濃度', fontsize=12)
    axes[0].set_title('濃度分布の時間発展', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    im = axes[1].contourf(solver.x, solver.t, C, levels=20, cmap='viridis')
    axes[1].set_xlabel('位置 (μm)', fontsize=12)
    axes[1].set_ylabel('時間 (h)', fontsize=12)
    axes[1].set_title('時空間発展', fontsize=14)
    plt.colorbar(im, ax=axes[1], label='濃度')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/machine-learning/example1_constant_diffusion.png', dpi=150)
    print(f"図を保存しました: example1_constant_diffusion.png")
    plt.close()
    
    np.savetxt('/home/ubuntu/repos/machine-learning/example1_data.csv', 
               C[-1, :], delimiter=',', 
               header='Concentration at t=72h')
    print(f"データを保存しました: example1_data.csv")
    print()


def example_2_concentration_dependent():
    """
    例2: 濃度依存拡散係数
    より現実的なCo-Ni-Cr系を模擬
    """
    print("=" * 60)
    print("例2: 濃度依存拡散係数による拡散")
    print("=" * 60)
    
    solver = SimpleDiffusionSolver(L=600.0, T_final=72.0, nx=200, nt=500)
    C = solver.solve_concentration_dependent_D(D0=5.0, D1=15.0, C_left=0.7, C_right=0.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time_indices = [0, len(solver.t)//4, len(solver.t)//2, len(solver.t)-1]
    for idx in time_indices:
        axes[0, 0].plot(solver.x, C[idx, :], label=f't = {solver.t[idx]:.1f} h')
    
    axes[0, 0].set_xlabel('位置 (μm)', fontsize=12)
    axes[0, 0].set_ylabel('濃度', fontsize=12)
    axes[0, 0].set_title('濃度分布の時間発展', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    im = axes[0, 1].contourf(solver.x, solver.t, C, levels=20, cmap='viridis')
    axes[0, 1].set_xlabel('位置 (μm)', fontsize=12)
    axes[0, 1].set_ylabel('時間 (h)', fontsize=12)
    axes[0, 1].set_title('時空間発展', fontsize=14)
    plt.colorbar(im, ax=axes[0, 1], label='濃度')
    
    D0, D1 = 5.0, 15.0
    C_final = C[-1, :]
    D_final = D0 + (D1 - D0) * C_final
    
    axes[1, 0].plot(solver.x, D_final, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('位置 (μm)', fontsize=12)
    axes[1, 0].set_ylabel('拡散係数 (μm²/h)', fontsize=12)
    axes[1, 0].set_title('最終時刻の拡散係数分布', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    dC_dx = np.gradient(C_final, solver.x)
    flux = -D_final * dC_dx
    
    axes[1, 1].plot(solver.x, flux, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('位置 (μm)', fontsize=12)
    axes[1, 1].set_ylabel('フラックス', fontsize=12)
    axes[1, 1].set_title('最終時刻の拡散フラックス', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/machine-learning/example2_concentration_dependent.png', dpi=150)
    print(f"図を保存しました: example2_concentration_dependent.png")
    plt.close()
    
    print()


def example_3_comparison():
    """
    例3: 定数と濃度依存の比較
    """
    print("=" * 60)
    print("例3: 定数拡散係数 vs 濃度依存拡散係数")
    print("=" * 60)
    
    solver = SimpleDiffusionSolver(L=600.0, T_final=72.0, nx=200, nt=500)
    
    C_const = solver.solve_constant_D(D=10.0, C_left=0.7, C_right=0.0)
    
    C_dep = solver.solve_concentration_dependent_D(D0=5.0, D1=15.0, C_left=0.7, C_right=0.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(solver.x, C_const[-1, :], 'b-', linewidth=2, label='定数 D=10')
    axes[0].plot(solver.x, C_dep[-1, :], 'r-', linewidth=2, label='濃度依存 D=5-15')
    axes[0].set_xlabel('位置 (μm)', fontsize=12)
    axes[0].set_ylabel('濃度', fontsize=12)
    axes[0].set_title('最終時刻の濃度分布比較 (t=72h)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    diff = C_dep[-1, :] - C_const[-1, :]
    axes[1].plot(solver.x, diff, 'g-', linewidth=2)
    axes[1].set_xlabel('位置 (μm)', fontsize=12)
    axes[1].set_ylabel('濃度差', fontsize=12)
    axes[1].set_title('濃度依存 - 定数', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/machine-learning/example3_comparison.png', dpi=150)
    print(f"図を保存しました: example3_comparison.png")
    plt.close()
    
    print(f"\n統計情報:")
    print(f"  定数拡散係数:")
    print(f"    最終濃度範囲: [{C_const[-1, :].min():.4f}, {C_const[-1, :].max():.4f}]")
    print(f"    中心濃度: {C_const[-1, len(solver.x)//2]:.4f}")
    print(f"  濃度依存拡散係数:")
    print(f"    最終濃度範囲: [{C_dep[-1, :].min():.4f}, {C_dep[-1, :].max():.4f}]")
    print(f"    中心濃度: {C_dep[-1, len(solver.x)//2]:.4f}")
    print(f"  最大差分: {np.abs(diff).max():.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Co系超合金拡散解析 - 使用例")
    print("=" * 60 + "\n")
    
    example_1_constant_diffusion()
    
    example_2_concentration_dependent()
    
    example_3_comparison()
    
    print("=" * 60)
    print("すべての例が完了しました！")
    print("生成されたファイル:")
    print("  - example1_constant_diffusion.png")
    print("  - example1_data.csv")
    print("  - example2_concentration_dependent.png")
    print("  - example3_comparison.png")
    print("=" * 60)
