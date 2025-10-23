import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def simulate_two_state_process(k01, k10, T, F, initial_state=0):
    t = 0
    state = initial_state
    times = [0]
    states = [state]
    entropy_production = 0
    entropy_history = [0]
    
    while t < T:
        if state == 0:
            rate = k01
            dt = np.random.exponential(1/rate)
            if t + dt < T:
                t += dt
                state = 1
                entropy_production += F
                times.append(t)
                states.append(state)
                entropy_history.append(entropy_production)
        else:
            rate = k10
            dt = np.random.exponential(1/rate)
            if t + dt < T:
                t += dt
                state = 0
                entropy_production -= F
                times.append(t)
                states.append(state)
                entropy_history.append(entropy_production)
        
        if t + dt >= T:
            break
    
    return np.array(times), np.array(states), entropy_production, np.array(entropy_history)


def run_simulation(k0, F, beta, T, n_trajectories):
    k01 = k0 * np.exp(beta * F / 2)
    k10 = k0 * np.exp(-beta * F / 2)
    
    entropy_productions = []
    
    for _ in range(n_trajectories):
        _, _, ep, _ = simulate_two_state_process(k01, k10, T, F)
        entropy_productions.append(ep / T)
    
    return np.array(entropy_productions)


def test_gc_symmetry():
    print("=" * 60)
    print("ガラボッティ―コーエン対称性のテスト")
    print("=" * 60)
    
    k0 = 1.0
    F = 0.2
    beta = 1.0
    T = 200.0
    n_trajectories = 10000
    
    print(f"\nパラメータ:")
    print(f"  基準遷移レート k₀ = {k0}")
    print(f"  外部駆動力 F = {F}")
    print(f"  逆温度 β = {beta}")
    print(f"  観測時間 T = {T}")
    print(f"  軌道数 = {n_trajectories}")
    
    k01 = k0 * np.exp(beta * F / 2)
    k10 = k0 * np.exp(-beta * F / 2)
    
    print(f"\n遷移レート:")
    print(f"  k₀₁ = {k01:.4f}")
    print(f"  k₁₀ = {k10:.4f}")
    
    print(f"\nシミュレーション実行中...")
    entropy_rates = run_simulation(k0, F, beta, T, n_trajectories)
    
    print(f"\n統計情報:")
    print(f"  平均エントロピー生成率: {np.mean(entropy_rates):.4f}")
    print(f"  標準偏差: {np.std(entropy_rates):.4f}")
    print(f"  正のエントロピー生成: {np.sum(entropy_rates > 0) / len(entropy_rates) * 100:.2f}%")
    print(f"  負のエントロピー生成: {np.sum(entropy_rates < 0) / len(entropy_rates) * 100:.2f}%")
    
    positive_rates = entropy_rates[entropy_rates > 0]
    negative_rates = entropy_rates[entropy_rates < 0]
    
    sigma_bins = np.linspace(-np.abs(entropy_rates).max(), np.abs(entropy_rates).max(), 30)
    bin_width = sigma_bins[1] - sigma_bins[0]
    
    hist_pos, _ = np.histogram(positive_rates, bins=sigma_bins[sigma_bins >= 0], density=True)
    hist_neg, _ = np.histogram(-negative_rates, bins=sigma_bins[sigma_bins >= 0], density=True)
    
    valid_indices = (hist_pos > 0) & (hist_neg > 0)
    sigma_values = (sigma_bins[sigma_bins >= 0][:-1] + bin_width/2)[valid_indices]
    log_ratio = np.log(hist_pos[valid_indices] / hist_neg[valid_indices])
    
    if len(sigma_values) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(sigma_values, log_ratio)
        theoretical_slope = beta * T
        
        print(f"\nGC対称性の検証:")
        print(f"  理論傾き (βT): {theoretical_slope:.4f}")
        print(f"  実測傾き: {slope:.4f}")
        print(f"  相対誤差: {abs(slope - theoretical_slope) / theoretical_slope * 100:.2f}%")
        print(f"  決定係数 R²: {r_value**2:.4f}")
        
        if r_value**2 > 0.95 and abs(slope - theoretical_slope) / theoretical_slope < 0.1:
            print(f"\n✓ GC対称性が確認されました！")
            return True
        else:
            print(f"\n✗ GC対称性の検証に問題があります。")
            return False
    else:
        print(f"\n✗ 検証に十分なデータがありません。")
        return False


def create_visualization():
    print("\n" + "=" * 60)
    print("可視化を生成中...")
    print("=" * 60)
    
    k0 = 1.0
    F = 0.2
    beta = 1.0
    T = 200.0
    n_trajectories = 10000
    
    k01 = k0 * np.exp(beta * F / 2)
    k10 = k0 * np.exp(-beta * F / 2)
    
    times_sample, states_sample, ep_sample, entropy_history_sample = simulate_two_state_process(k01, k10, T, F)
    entropy_rates = run_simulation(k0, F, beta, T, n_trajectories)
    
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.step(times_sample, states_sample, where='post', linewidth=1.5)
    ax1.set_xlabel('時間 t', fontsize=11)
    ax1.set_ylabel('状態', fontsize=11)
    ax1.set_yticks([0, 1])
    ax1.set_title('状態遷移の時間発展', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(times_sample, entropy_history_sample, linewidth=1.5, color='red')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('時間 t', fontsize=11)
    ax2.set_ylabel('累積エントロピー生成 Σ', fontsize=11)
    ax2.set_title('エントロピー生成の時間発展', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(entropy_rates, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    mean_entropy_rate = np.mean(entropy_rates)
    ax3.axvline(mean_entropy_rate, color='red', linestyle='--', linewidth=2, 
                label=f'平均値 = {mean_entropy_rate:.3f}')
    ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('エントロピー生成率 Σ/t', fontsize=11)
    ax3.set_ylabel('確率密度 P(Σ/t)', fontsize=11)
    ax3.set_title('エントロピー生成率の確率分布', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    positive_rates = entropy_rates[entropy_rates > 0]
    negative_rates = entropy_rates[entropy_rates < 0]
    
    sigma_bins = np.linspace(-np.abs(entropy_rates).max(), np.abs(entropy_rates).max(), 30)
    bin_width = sigma_bins[1] - sigma_bins[0]
    
    hist_pos, _ = np.histogram(positive_rates, bins=sigma_bins[sigma_bins >= 0], density=True)
    hist_neg, _ = np.histogram(-negative_rates, bins=sigma_bins[sigma_bins >= 0], density=True)
    
    valid_indices = (hist_pos > 0) & (hist_neg > 0)
    sigma_values = (sigma_bins[sigma_bins >= 0][:-1] + bin_width/2)[valid_indices]
    log_ratio = np.log(hist_pos[valid_indices] / hist_neg[valid_indices])
    
    ax4.scatter(sigma_values, log_ratio, s=60, alpha=0.7, color='blue', 
                edgecolors='black', linewidth=1, label='シミュレーション')
    
    theoretical_line = beta * sigma_values * T
    ax4.plot(sigma_values, theoretical_line, 'r-', linewidth=2.5, 
             label=f'理論値: βΣt', zorder=2)
    
    if len(sigma_values) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(sigma_values, log_ratio)
        fitted_line = slope * sigma_values + intercept
        ax4.plot(sigma_values, fitted_line, 'g--', linewidth=2, 
                 label=f'フィット: R²={r_value**2:.3f}')
    
    ax4.set_xlabel('エントロピー生成率 Σ/t', fontsize=11)
    ax4.set_ylabel('ln[P(Σ)/P(-Σ)]', fontsize=11)
    ax4.set_title('GC対称性の検証', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/repos/machine-learning/gc_symmetry_test.png', dpi=150, bbox_inches='tight')
    print(f"\n可視化を保存しました: gc_symmetry_test.png")
    plt.close()


if __name__ == "__main__":
    success = test_gc_symmetry()
    create_visualization()
    
    print("\n" + "=" * 60)
    if success:
        print("テスト完了: すべてのテストに合格しました ✓")
    else:
        print("テスト完了: 一部のテストに失敗しました")
    print("=" * 60)
