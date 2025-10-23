import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
import pandas as pd

st.set_page_config(page_title="ガラボッティ―コーエン対称性", layout="wide")

st.title("ガラボッティ―コーエン（GC）対称性の体感")
st.markdown("## 外部駆動下の二状態マルコフジャンプ過程")

st.markdown("""

二状態系（状態0と状態1）が外部駆動力 $F$ の下で確率的に遷移するマルコフ過程を考えます。

**遷移レート:**
- 状態 0 → 1: $k_{01}(F) = k_0 \exp(\\beta F/2)$
- 状態 1 → 0: $k_{10}(F) = k_0 \exp(-\\beta F/2)$

ここで:
- $k_0$: 基準遷移レート
- $\\beta = 1/(k_B T)$: 逆温度
- $F$: 外部駆動力（熱力学的力）

**エントロピー生成:**

各遷移でエントロピー生成が発生します:
- 0 → 1 の遷移: $\Delta s = +F$
- 1 → 0 の遷移: $\Delta s = -F$

時間 $t$ でのエントロピー生成率は:
$$\Sigma_t = \\frac{1}{t} \sum_{i} \Delta s_i$$

**ガラボッティ―コーエン対称性:**

エントロピー生成率の確率分布 $P(\Sigma)$ は以下の対称性を満たします:

$$\\frac{P(\Sigma)}{P(-\Sigma)} = \exp(\\beta \Sigma t)$$

対数をとると:
$$\ln\\frac{P(\Sigma)}{P(-\Sigma)} = \\beta \Sigma t$$

これは揺らぎの定理（Fluctuation Theorem）の一種で、非平衡統計力学の基本原理です。
""")

st.sidebar.header("パラメータ設定")

k0 = st.sidebar.slider("基準遷移レート k₀", 0.1, 10.0, 1.0, 0.1)
F = st.sidebar.slider("外部駆動力 F", 0.0, 2.0, 0.3, 0.05)
beta = st.sidebar.slider("逆温度 β", 0.1, 5.0, 1.0, 0.1)
T = st.sidebar.slider("観測時間 T", 10.0, 500.0, 100.0, 10.0)
n_trajectories = st.sidebar.slider("軌道数", 100, 20000, 5000, 100)

st.sidebar.markdown("---")
st.sidebar.markdown("### 理論値")
st.sidebar.markdown(f"k₀₁ = {k0 * np.exp(beta * F / 2):.3f}")
st.sidebar.markdown(f"k₁₀ = {k0 * np.exp(-beta * F / 2):.3f}")


def simulate_two_state_process(k01, k10, T, initial_state=0):
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
        _, _, ep, _ = simulate_two_state_process(k01, k10, T)
        entropy_productions.append(ep / T)
    
    return np.array(entropy_productions)


if st.button("シミュレーション実行", type="primary"):
    with st.spinner("シミュレーション実行中..."):
        k01 = k0 * np.exp(beta * F / 2)
        k10 = k0 * np.exp(-beta * F / 2)
        
        times_sample, states_sample, ep_sample, entropy_history_sample = simulate_two_state_process(k01, k10, T)
        
        entropy_rates = run_simulation(k0, F, beta, T, n_trajectories)
        
        st.session_state['simulation_done'] = True
        st.session_state['times_sample'] = times_sample
        st.session_state['states_sample'] = states_sample
        st.session_state['ep_sample'] = ep_sample
        st.session_state['entropy_history_sample'] = entropy_history_sample
        st.session_state['entropy_rates'] = entropy_rates
        st.session_state['k01'] = k01
        st.session_state['k10'] = k10
        st.session_state['T'] = T
        st.session_state['beta'] = beta

if 'simulation_done' in st.session_state and st.session_state['simulation_done']:
    times_sample = st.session_state['times_sample']
    states_sample = st.session_state['states_sample']
    ep_sample = st.session_state['ep_sample']
    entropy_history_sample = st.session_state['entropy_history_sample']
    entropy_rates = st.session_state['entropy_rates']
    k01 = st.session_state['k01']
    k10 = st.session_state['k10']
    T_sim = st.session_state['T']
    beta_sim = st.session_state['beta']
    
    st.success(f"シミュレーション完了！ {n_trajectories} 本の軌道を生成しました。")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("サンプル軌道")
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.step(times_sample, states_sample, where='post', linewidth=1.5)
        ax1.set_xlabel('時間 t', fontsize=12)
        ax1.set_ylabel('状態', fontsize=12)
        ax1.set_yticks([0, 1])
        ax1.set_title('状態遷移の時間発展', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(times_sample, entropy_history_sample, linewidth=1.5, color='red')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('時間 t', fontsize=12)
        ax2.set_ylabel('累積エントロピー生成 Σ', fontsize=12)
        ax2.set_title('エントロピー生成の時間発展', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        st.subheader("エントロピー生成率の分布")
        fig2, ax = plt.subplots(figsize=(10, 8))
        
        counts, bins, _ = ax.hist(entropy_rates, bins=50, density=True, alpha=0.7, 
                                   color='blue', edgecolor='black', label='シミュレーション')
        
        mean_entropy_rate = np.mean(entropy_rates)
        ax.axvline(mean_entropy_rate, color='red', linestyle='--', linewidth=2, 
                   label=f'平均値 = {mean_entropy_rate:.3f}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('エントロピー生成率 Σ/t', fontsize=12)
        ax.set_ylabel('確率密度 P(Σ/t)', fontsize=12)
        ax.set_title('エントロピー生成率の確率分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    
    st.subheader("ガラボッティ―コーエン対称性の検証")
    
    st.markdown("""
    GC対称性: $\ln[P(\Sigma)/P(-\Sigma)] = \\beta \Sigma t$
    
    この関係を検証するため、正のエントロピー生成率と負のエントロピー生成率の確率比の対数をプロットします。
    理論的には、この値は $\\beta \Sigma t$ に比例する直線になるはずです。
    """)
    
    positive_rates = entropy_rates[entropy_rates > 0]
    negative_rates = entropy_rates[entropy_rates < 0]
    
    sigma_bins = np.linspace(-np.abs(entropy_rates).max(), np.abs(entropy_rates).max(), 30)
    bin_width = sigma_bins[1] - sigma_bins[0]
    
    hist_pos, _ = np.histogram(positive_rates, bins=sigma_bins[sigma_bins >= 0], density=True)
    hist_neg, _ = np.histogram(-negative_rates, bins=sigma_bins[sigma_bins >= 0], density=True)
    
    valid_indices = (hist_pos > 0) & (hist_neg > 0)
    sigma_values = (sigma_bins[sigma_bins >= 0][:-1] + bin_width/2)[valid_indices]
    log_ratio = np.log(hist_pos[valid_indices] / hist_neg[valid_indices])
    
    if len(sigma_values) > 0:
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        ax.scatter(sigma_values, log_ratio, s=80, alpha=0.7, color='blue', 
                   edgecolors='black', linewidth=1, label='シミュレーション', zorder=3)
        
        theoretical_line = beta_sim * sigma_values * T_sim
        ax.plot(sigma_values, theoretical_line, 'r-', linewidth=2.5, 
                label=f'理論値: βΣt (β={beta_sim:.2f}, t={T_sim:.1f})', zorder=2)
        
        if len(sigma_values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(sigma_values, log_ratio)
            fitted_line = slope * sigma_values + intercept
            ax.plot(sigma_values, fitted_line, 'g--', linewidth=2, 
                    label=f'フィット: 傾き={slope:.3f} (理論={beta_sim*T_sim:.3f}), R²={r_value**2:.4f}', 
                    zorder=1)
            
            st.markdown(f"""
            **フィッティング結果:**
            - 実測傾き: {slope:.3f}
            - 理論傾き: {beta_sim * T_sim:.3f}
            - 相対誤差: {abs(slope - beta_sim * T_sim) / (beta_sim * T_sim) * 100:.2f}%
            - 決定係数 R²: {r_value**2:.4f}
            """)
        
        ax.set_xlabel('エントロピー生成率 Σ/t', fontsize=13)
        ax.set_ylabel('ln[P(Σ)/P(-Σ)]', fontsize=13)
        ax.set_title('ガラボッティ―コーエン対称性の検証', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    else:
        st.warning("対称性の検証に十分なデータがありません。軌道数を増やすか、観測時間を長くしてください。")
    
    st.subheader("統計情報")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("平均エントロピー生成率", f"{np.mean(entropy_rates):.4f}")
    
    with col2:
        st.metric("標準偏差", f"{np.std(entropy_rates):.4f}")
    
    with col3:
        positive_fraction = np.sum(entropy_rates > 0) / len(entropy_rates) * 100
        st.metric("正のエントロピー生成", f"{positive_fraction:.1f}%")
    
    with col4:
        negative_fraction = np.sum(entropy_rates < 0) / len(entropy_rates) * 100
        st.metric("負のエントロピー生成", f"{negative_fraction:.1f}%")
    
    st.markdown("---")
    st.subheader("データのダウンロード")
    
    df = pd.DataFrame({
        'エントロピー生成率': entropy_rates
    })
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="CSV形式でダウンロード",
        data=csv,
        file_name='gallavotti_cohen_data.csv',
        mime='text/csv',
    )

st.markdown("---")
st.markdown("""

**エントロピー生成の意味:**
- 正のエントロピー生成: 系が外部駆動力の方向に従って進化（典型的）
- 負のエントロピー生成: 系が外部駆動力に逆らって進化（非典型的、稀）

**GC対称性の意義:**
- 非平衡状態でも、揺らぎには厳密な対称性が存在する
- 負のエントロピー生成（第二法則に反する揺らぎ）の確率を定量的に予測できる
- マクロな第二法則とミクロな可逆性を結びつける

**実験での検証:**
- 光ピンセットで捕捉されたコロイド粒子
- 生体分子モーター
- 電気回路の熱雑音

この対称性は、非平衡統計力学の基本原理であり、熱力学第二法則の揺らぎレベルでの精密化を与えます。
""")

st.markdown("---")
st.markdown("""

1. Gallavotti, G., & Cohen, E. G. D. (1995). Dynamical ensembles in stationary states. *Journal of Statistical Physics*, 80(5-6), 931-970.
2. Evans, D. J., Cohen, E. G. D., & Morriss, G. P. (1993). Probability of second law violations in shearing steady states. *Physical Review Letters*, 71(15), 2401.
3. Seifert, U. (2012). Stochastic thermodynamics, fluctuation theorems and molecular machines. *Reports on Progress in Physics*, 75(12), 126001.
""")
