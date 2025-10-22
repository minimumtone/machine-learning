import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

plt.rcParams["font.family"] = ["DejaVu Sans", "Hiragino Sans", "Yu Gothic", "Meiryo", "Takao", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="変分法とDFTの理解", layout="wide", page_icon="⚛️")
st.title("⚛️ 変分法と密度汎関数法(DFT)の理解")
st.markdown("""このアプリケーションは、**変分法**の基本原理を学び、**密度汎関数法(DFT)**による第一原理計算への応用を理解するための教育ツールです。""")

tabs = st.tabs(["📚 変分法の基礎", "🎯 変分原理の実演", "⚛️ DFT基礎", "🔬 簡易DFT計算", "📊 エネルギー最小化"])

with tabs[0]:
    st.header("📚 変分法の基礎理論")
    
    st.markdown(r"""
    
    **変分法(Calculus of Variations)**は、関数の関数(汎関数)を最適化する数学的手法です。
    物理学では、系の真の状態は**作用や自由エネルギーが最小となる状態**として記述されます。
    
    汎関数 $F[y]$ は関数 $y(x)$ を入力として実数値を出力します:
    
    $$F[y] = \int_a^b L(x, y(x), y'(x)) dx$$
    
    ここで $L$ はラグランジアン(被積分関数)です。
    
    汎関数 $F[y]$ を最小化する関数 $y(x)$ は以下を満たします:
    
    $$\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'} = 0$$
    
    これが**オイラー-ラグランジュ方程式**です。
    """)
    
    st.markdown(r"""
    
    粒子の軌道は作用 $S$ を最小化します:
    
    $$S = \int_{t_1}^{t_2} L(q, \dot{q}, t) dt$$
    
    系の基底状態エネルギー $E_0$ は:
    
    $$E_0 = \min_{\psi} \frac{\langle \psi | \hat{H} | \psi \rangle}{\langle \psi | \psi \rangle}$$
    
    任意の試行波動関数 $\psi$ に対して $E[\psi] \geq E_0$ が成立します。
    
    電子密度 $n(\mathbf{r})$ の汎関数としてエネルギーを表現:
    
    $$E[n] = T[n] + V_{ext}[n] + V_{ee}[n]$$
    
    基底状態密度 $n_0$ は $E[n]$ を最小化します。
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **変分法の利点**
        - 近似解の上界を保証
        - 系統的な改善が可能
        - 複雑な多体系に適用可能
        """)
    
    with col2:
        st.success("""
        **DFTへの応用**
        - 多電子系を密度で記述
        - 計算コストの大幅削減
        - 材料科学で広く使用
        """)

with tabs[1]:
    st.header("🎯 変分原理の実演: 最速降下曲線")
    
    st.markdown(r"""
    
    点A(0,0)から点B(1,-1)まで、重力下で最短時間で滑り落ちる曲線を求めます。
    
    **汎関数(時間)**:
    $$T[y] = \int_0^1 \sqrt{\frac{1 + (y')^2}{2gy}} dx$$
    
    変分法により、最適曲線は**サイクロイド曲線**であることが導かれます。
    """)
    
    num_comparisons = st.slider("比較する曲線の数", 3, 10, 5)
    
    x = np.linspace(0, 1, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    g = 9.8
    times = []
    labels = []
    
    for i in range(num_comparisons):
        if i == 0:
            y = -x
            label = "直線"
            color = 'red'
            linewidth = 2
        elif i == 1:
            y = -x**2
            label = "放物線"
            color = 'blue'
            linewidth = 2
        elif i == 2:
            theta = np.linspace(0, np.pi, 100)
            R = 0.5
            x_cycl = R * (theta - np.sin(theta))
            y_cycl = -R * (1 - np.cos(theta))
            x = x_cycl / x_cycl[-1]
            y = y_cycl / abs(y_cycl[-1]) * 1.0
            label = "サイクロイド(最適解)"
            color = 'green'
            linewidth = 3
        else:
            power = 1.2 + 0.3 * i
            y = -(x**power)
            label = f"べき乗 x^{power:.1f}"
            color = 'gray'
            linewidth = 1
        
        ax1.plot(x, y, label=label, color=color, linewidth=linewidth, alpha=0.8)
        
        dy_dx = np.gradient(y, x)
        integrand = np.sqrt((1 + dy_dx**2) / (2 * g * np.abs(y) + 1e-10))
        time = np.trapz(integrand, x)
        times.append(time)
        labels.append(label)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('様々な曲線の比較', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    colors_bar = ['red', 'blue', 'green'] + ['gray'] * (num_comparisons - 3)
    ax2.barh(labels, times, color=colors_bar, alpha=0.7)
    ax2.set_xlabel('時間 (秒)', fontsize=12)
    ax2.set_title('各曲線の降下時間', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    min_idx = np.argmin(times)
    ax2.axvline(times[min_idx], color='green', linestyle='--', linewidth=2, label='最小時間')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.success(f"✅ **最短時間: {times[min_idx]:.4f}秒 ({labels[min_idx]})**")
    
    st.markdown("""
    - **緑色のサイクロイド曲線**が最短時間を実現
    - 直線や単純な放物線よりも効率的
    - 変分法により数学的に最適性が保証される
    """)

with tabs[2]:
    st.header("⚛️ 密度汎関数法(DFT)の基礎")
    
    st.markdown(r"""
    
    **Hohenberg-Kohnの定理**(1964年)により、基底状態のすべての性質は
    電子密度 $n(\mathbf{r})$ のみで決定できることが証明されました。
    
    
    | アプローチ | 変数の数 | 計算コスト |
    |-----------|---------|----------|
    | 波動関数法 | $3N$ (N個の電子座標) | $O(2^N)$ |
    | **DFT** | **3** (空間座標のみ) | **$O(N^3)$** |
    
    
    多体問題を一電子問題に変換:
    
    $$\left[-\frac{\hbar^2}{2m}\nabla^2 + V_{eff}(\mathbf{r})\right]\phi_i(\mathbf{r}) = \varepsilon_i \phi_i(\mathbf{r})$$
    
    電子密度:
    $$n(\mathbf{r}) = \sum_{i=1}^N |\phi_i(\mathbf{r})|^2$$
    
    実効ポテンシャル:
    $$V_{eff}(\mathbf{r}) = V_{ext}(\mathbf{r}) + \int \frac{n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} d\mathbf{r}' + V_{xc}[n](\mathbf{r})$$
    """)
    
    st.markdown(r"""
    
    DFTの精度は交換相関エネルギー $E_{xc}[n]$ の近似に依存します:
    
    $$E_{xc}^{LDA}[n] = \int n(\mathbf{r}) \varepsilon_{xc}(n(\mathbf{r})) d\mathbf{r}$$
    - 最も単純な近似
    - 均一電子ガスの厳密解を利用
    
    $$E_{xc}^{GGA}[n] = \int f(n(\mathbf{r}), \nabla n(\mathbf{r})) d\mathbf{r}$$
    - 密度勾配も考慮
    - 分子・固体の構造計算で標準的
    
    $$E_{xc}^{hybrid} = aE_x^{HF} + (1-a)E_x^{DFT} + E_c^{DFT}$$
    - Hartree-Fock交換を混合
    - バンドギャップの精度向上
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LDA", "最速", delta="精度: 低")
    with col2:
        st.metric("GGA", "標準", delta="精度: 中")
    with col3:
        st.metric("Hybrid", "高精度", delta="コスト: 高")
    
    st.info("""
    💡 **実用上のポイント**
    - 構造最適化: GGA (PBE, PBEsol)
    - バンドギャップ: Hybrid (HSE06, B3LYP)
    - 大規模系: LDA
    """)

with tabs[3]:
    st.header("🔬 簡易的な1次元DFT計算")
    
    st.markdown(r"""
    
    無限井戸ポテンシャル中の電子をKohn-Sham方程式で解きます:
    
    $$-\frac{\hbar^2}{2m}\frac{d^2\phi}{dx^2} + V(x)\phi = \varepsilon \phi$$
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        N_electrons = st.slider("電子数", 1, 10, 4)
        L = st.slider("井戸の幅 (Å)", 5.0, 20.0, 10.0, 0.5)
        V0 = st.slider("外部ポテンシャルの強度 (eV)", 0.0, 5.0, 2.0, 0.5)
    
    x = np.linspace(0, L, 200)
    dx = x[1] - x[0]
    
    V_ext = V0 * (np.cos(2 * np.pi * x / L) + 1) / 2
    
    hbar = 1.054571817e-34
    m_e = 9.1093837015e-31
    eV = 1.602176634e-19
    angstrom = 1e-10
    
    prefactor = hbar**2 / (2 * m_e * angstrom**2 * eV)
    
    H = np.zeros((len(x), len(x)))
    for i in range(1, len(x)-1):
        H[i, i] = 2 * prefactor / dx**2 + V_ext[i]
        H[i, i-1] = -prefactor / dx**2
        H[i, i+1] = -prefactor / dx**2
    
    H[0, 0] = 1e10
    H[-1, -1] = 1e10
    
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    n_density = np.zeros_like(x)
    for i in range(N_electrons):
        n_density += eigenvectors[:, i]**2
    
    n_density = n_density / np.trapz(n_density, x) * N_electrons
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(x, V_ext, 'k-', linewidth=2, label='外部ポテンシャル')
    ax.fill_between(x, 0, V_ext, alpha=0.3, color='gray')
    ax.set_xlabel('位置 (Å)', fontsize=11)
    ax.set_ylabel('エネルギー (eV)', fontsize=11)
    ax.set_title('外部ポテンシャル V(x)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for i in range(min(5, N_electrons)):
        offset = eigenvalues[i]
        ax.plot(x, eigenvectors[:, i]**2 * 5 + offset, label=f'n={i+1}, E={eigenvalues[i]:.2f} eV')
        ax.axhline(offset, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('位置 (Å)', fontsize=11)
    ax.set_ylabel('エネルギー (eV)', fontsize=11)
    ax.set_title('固有状態 (波動関数の2乗)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(x, n_density, 'b-', linewidth=2, label='電子密度')
    ax.fill_between(x, 0, n_density, alpha=0.3, color='blue')
    ax.set_xlabel('位置 (Å)', fontsize=11)
    ax.set_ylabel('密度 (電子/Å)', fontsize=11)
    ax.set_title(f'総電子密度 (N={N_electrons})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    energies = eigenvalues[:min(10, len(eigenvalues))]
    ax.barh(range(len(energies)), energies, color=['red' if i < N_electrons else 'lightgray' for i in range(len(energies))])
    ax.set_yticks(range(len(energies)))
    ax.set_yticklabels([f'n={i+1}' for i in range(len(energies))])
    ax.set_xlabel('エネルギー (eV)', fontsize=11)
    ax.set_title('エネルギー準位 (赤=占有)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    total_energy = sum(eigenvalues[:N_electrons])
    st.metric("総電子エネルギー", f"{total_energy:.2f} eV")
    
    st.markdown("""
    - **外部ポテンシャル**: 周期的なポテンシャルが電子を束縛
    - **固有状態**: 各エネルギー準位の電子分布
    - **電子密度**: 全占有軌道の重ね合わせ
    - **エネルギー準位**: 低いエネルギーから順に占有(赤色)
    """)

with tabs[4]:
    st.header("📊 変分法によるエネルギー最小化")
    
    st.markdown(r"""
    
    変分原理を用いて、ガウス型試行波動関数のパラメータを最適化します:
    
    $$\psi_{trial}(x) = A \exp\left(-\alpha (x-x_0)^2\right)$$
    
    期待値エネルギー:
    $$E[\alpha] = \frac{\langle \psi | \hat{H} | \psi \rangle}{\langle \psi | \psi \rangle}$$
    
    を最小化する $\alpha$ を求めます。
    """)
    
    potential_type = st.selectbox("ポテンシャルの種類", 
                                  ["調和振動子", "二重井戸", "非対称井戸"])
    
    alpha_range = np.linspace(0.1, 5.0, 100)
    x_calc = np.linspace(-5, 5, 500)
    dx_calc = x_calc[1] - x_calc[0]
    
    if potential_type == "調和振動子":
        k = 1.0
        V_pot = 0.5 * k * x_calc**2
        analytical_alpha = np.sqrt(k)
        st.info(f"💡 解析解: α_opt = √k = {analytical_alpha:.3f}")
    elif potential_type == "二重井戸":
        V_pot = (x_calc**2 - 4)**2 / 16
        analytical_alpha = None
    else:
        V_pot = 0.3 * x_calc**2 + 0.5 * x_calc
        analytical_alpha = None
    
    energies = []
    
    for alpha in alpha_range:
        psi = np.exp(-alpha * x_calc**2)
        psi = psi / np.sqrt(np.trapz(psi**2, x_calc))
        
        d2psi_dx2 = np.gradient(np.gradient(psi, dx_calc), dx_calc)
        
        T = -0.5 * np.trapz(psi * d2psi_dx2, x_calc)
        V = np.trapz(psi**2 * V_pot, x_calc)
        E = T + V
        energies.append(E)
    
    optimal_idx = np.argmin(energies)
    optimal_alpha = alpha_range[optimal_idx]
    optimal_energy = energies[optimal_idx]
    
    psi_optimal = np.exp(-optimal_alpha * x_calc**2)
    psi_optimal = psi_optimal / np.sqrt(np.trapz(psi_optimal**2, x_calc))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(alpha_range, energies, 'b-', linewidth=2, label='E(α)')
    ax.axvline(optimal_alpha, color='red', linestyle='--', linewidth=2, 
               label=f'最適α = {optimal_alpha:.3f}')
    if analytical_alpha:
        ax.axvline(analytical_alpha, color='green', linestyle=':', linewidth=2,
                   label=f'解析解α = {analytical_alpha:.3f}')
    ax.scatter([optimal_alpha], [optimal_energy], color='red', s=100, zorder=5)
    ax.set_xlabel('パラメータ α', fontsize=12)
    ax.set_ylabel('エネルギー E(α)', fontsize=12)
    ax.set_title('変分エネルギーの最小化', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(x_calc, V_pot, 'k-', linewidth=2, label='ポテンシャル V(x)', alpha=0.7)
    ax.plot(x_calc, psi_optimal**2 * max(V_pot), 'r-', linewidth=2, 
            label=f'最適波動関数² (α={optimal_alpha:.3f})')
    ax.fill_between(x_calc, 0, psi_optimal**2 * max(V_pot), alpha=0.3, color='red')
    ax.axhline(optimal_energy, color='blue', linestyle='--', linewidth=1.5,
               label=f'基底エネルギー = {optimal_energy:.3f}')
    ax.set_xlabel('位置 x', fontsize=12)
    ax.set_ylabel('エネルギー / 確率密度', fontsize=12)
    ax.set_title('最適化された波動関数', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("最適パラメータ α", f"{optimal_alpha:.4f}")
    with col2:
        st.metric("基底エネルギー", f"{optimal_energy:.4f}")
    with col3:
        if analytical_alpha:
            error = abs(optimal_alpha - analytical_alpha) / analytical_alpha * 100
            st.metric("解析解との誤差", f"{error:.2f}%")
        else:
            st.metric("計算モード", "数値最適化")
    
    st.success("""
    ✅ **変分法の威力**
    - 単純な試行関数でも良い近似が得られる
    - パラメータ最適化により基底エネルギーの上界を与える
    - DFTでも同様の原理で電子密度を最適化
    """)

st.markdown("---")
st.markdown("""

- **Hohenberg-Kohn定理**: P. Hohenberg and W. Kohn, Phys. Rev. **136**, B864 (1964)
- **Kohn-Sham方程式**: W. Kohn and L. J. Sham, Phys. Rev. **140**, A1133 (1965)
- **教科書**: R. M. Martin, "Electronic Structure: Basic Theory and Practical Methods"
- **日本語解説**: 押山淳「密度汎関数法の基礎」講談社サイエンティフィク

---
**開発**: 機械学習・材料科学教育プロジェクト | バージョン: 1.0
""")
