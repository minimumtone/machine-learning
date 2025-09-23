import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="CVM（クラスター変分法）教育アプリ",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 CVM（クラスター変分法）基礎理解アプリ")
st.markdown("""
このアプリは**クラスター変分法（Cluster Variation Method, CVM）**を基礎から理解するための教育ツールです。
二元系合金の統計熱力学的性質を学習し、相図の計算方法を段階的に習得できます。

**学習内容:**
- 格子統計の基礎概念
- クラスター近似の理論
- 二元系での自由エネルギー計算
- 相分離と相図の予測
- 温度・組成依存性の理解
""")

class CVMBinarySystem:
    """二元系CVM計算クラス"""
    
    def __init__(self, interaction_energy=1000, temperature=300):
        self.J = interaction_energy  # 相互作用エネルギー [J/mol]
        self.T = temperature  # 温度 [K]
        self.R = 8.314  # ガス定数 [J/(mol·K)]
        
    def point_approximation_free_energy(self, x):
        """点近似での自由エネルギー"""
        if x <= 0 or x >= 1:
            return np.inf
        
        S_config = -self.R * (x * np.log(x) + (1-x) * np.log(1-x))
        
        H = self.J * x * (1-x)
        
        F = H - self.T * S_config
        return F
    
    def pair_approximation_free_energy(self, x, y_AA=None, y_BB=None, y_AB=None):
        """ペア近似での自由エネルギー"""
        if y_AA is None or y_BB is None or y_AB is None:
            y_AA = x**2
            y_BB = (1-x)**2
            y_AB = 2*x*(1-x)
        
        total = y_AA + y_BB + y_AB
        if total > 0:
            y_AA /= total
            y_BB /= total
            y_AB /= total
        
        S_pair = 0
        if y_AA > 0:
            S_pair += y_AA * np.log(y_AA)
        if y_BB > 0:
            S_pair += y_BB * np.log(y_BB)
        if y_AB > 0:
            S_pair += y_AB * np.log(y_AB)
        S_pair *= -self.R
        
        if x > 0 and x < 1:
            S_point = -self.R * (x * np.log(x) + (1-x) * np.log(1-x))
        else:
            S_point = 0
        
        H = self.J * y_AB
        
        F = H - self.T * (S_pair - S_point)
        return F
    
    def calculate_equilibrium_pairs(self, x):
        """平衡状態でのペア確率を計算"""
        def objective(vars):
            y_AA, y_BB, y_AB = vars
            
            if y_AA < 0 or y_BB < 0 or y_AB < 0:
                return 1e10
            
            total = y_AA + y_BB + y_AB
            if abs(total - 1) > 1e-6:
                return 1e10
            
            x_calc = y_AA + 0.5 * y_AB
            if abs(x_calc - x) > 1e-6:
                return 1e10
            
            return self.pair_approximation_free_energy(x, y_AA, y_BB, y_AB)
        
        y_AA_init = x**2
        y_BB_init = (1-x)**2
        y_AB_init = 2*x*(1-x)
        
        total = y_AA_init + y_BB_init + y_AB_init
        y_AA_init /= total
        y_BB_init /= total
        y_AB_init /= total
        
        try:
            result = minimize(objective, [y_AA_init, y_BB_init, y_AB_init], 
                            method='SLSQP',
                            bounds=[(0, 1), (0, 1), (0, 1)])
            
            if result.success:
                return result.x
            else:
                return [y_AA_init, y_BB_init, y_AB_init]
        except:
            return [y_AA_init, y_BB_init, y_AB_init]
    
    def calculate_spinodal(self, x_range):
        """スピノーダル線の計算"""
        spinodal_points = []
        
        for x in x_range:
            if x <= 0.01 or x >= 0.99:
                continue
                
            dx = 0.001
            f_plus = self.point_approximation_free_energy(x + dx)
            f_minus = self.point_approximation_free_energy(x - dx)
            f_center = self.point_approximation_free_energy(x)
            
            d2f_dx2 = (f_plus - 2*f_center + f_minus) / (dx**2)
            
            if d2f_dx2 < 0:  # 不安定領域
                spinodal_points.append(x)
        
        return spinodal_points

def create_theory_explanation():
    """理論説明セクション"""
    st.header("📚 CVM理論の基礎")
    
    tab1, tab2, tab3, tab4 = st.tabs(["格子統計", "クラスター近似", "自由エネルギー", "相分離"])
    
    with tab1:
        st.subheader("格子統計の基礎")
        st.markdown("""
        **格子統計**は、結晶格子上の原子配置を統計力学的に扱う手法です。
        
        **基本概念:**
        - 格子点上に異なる種類の原子（A, B）が配置
        - 各配置の統計的重みを考慮
        - 熱平衡状態での最安定配置を求める
        
        **二元系での記号:**
        - $x$: A原子の濃度
        - $1-x$: B原子の濃度
        - $N$: 総格子点数
        """)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        for i in range(5):
            for j in range(5):
                if (i + j) % 2 == 0:
                    ax[0].scatter(i, j, c='red', s=100, marker='o')
                else:
                    ax[0].scatter(i, j, c='blue', s=100, marker='s')
        ax[0].set_title('規則配置 (x=0.5)')
        ax[0].set_xlim(-0.5, 4.5)
        ax[0].set_ylim(-0.5, 4.5)
        ax[0].grid(True, alpha=0.3)
        
        np.random.seed(42)
        for i in range(5):
            for j in range(5):
                if np.random.random() < 0.5:
                    ax[1].scatter(i, j, c='red', s=100, marker='o')
                else:
                    ax[1].scatter(i, j, c='blue', s=100, marker='s')
        ax[1].set_title('ランダム配置 (x≈0.5)')
        ax[1].set_xlim(-0.5, 4.5)
        ax[1].set_ylim(-0.5, 4.5)
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("クラスター近似")
        st.markdown("""
        **クラスター変分法**では、格子を小さなクラスターに分割して近似計算を行います。
        
        **近似レベル:**
        1. **点近似**: 各格子点を独立に扱う（平均場近似）
        2. **ペア近似**: 最近接原子対の相関を考慮
        3. **三角形近似**: 三角形クラスターを考慮
        4. **四面体近似**: より大きなクラスターを考慮
        
        **ペア近似での変数:**
        - $y_{AA}$: A-Aペアの確率
        - $y_{BB}$: B-Bペアの確率  
        - $y_{AB}$: A-Bペアの確率
        
        **制約条件:**
        - $y_{AA} + y_{BB} + y_{AB} = 1$
        - $y_{AA} + \\frac{1}{2}y_{AB} = x$
        """)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].scatter(0, 0, c='red', s=200, marker='o')
        axes[0].set_title('点近似')
        axes[0].set_xlim(-1, 1)
        axes[0].set_ylim(-1, 1)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot([0, 1], [0, 0], 'k-', linewidth=3)
        axes[1].scatter([0, 1], [0, 0], c=['red', 'blue'], s=200, marker='o')
        axes[1].set_title('ペア近似')
        axes[1].set_xlim(-0.5, 1.5)
        axes[1].set_ylim(-0.5, 0.5)
        axes[1].grid(True, alpha=0.3)
        
        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [0, 0, np.sqrt(3)/2, 0]
        axes[2].plot(triangle_x, triangle_y, 'k-', linewidth=3)
        axes[2].scatter(triangle_x[:-1], triangle_y[:-1], c=['red', 'blue', 'green'], s=200, marker='o')
        axes[2].set_title('三角形近似')
        axes[2].set_xlim(-0.2, 1.2)
        axes[2].set_ylim(-0.2, 1.0)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.subheader("自由エネルギーの計算")
        st.markdown("""
        **自由エネルギー**は系の安定性を決定する重要な量です。
        
        **点近似での自由エネルギー:**
        $$F = H - TS_{config}$$
        
        **各項の意味:**
        - $H = Jx(1-x)$: エンタルピー項（相互作用エネルギー）
        - $S_{config} = -R[x\\ln x + (1-x)\\ln(1-x)]$: 配置エントロピー
        - $T$: 温度
        - $J$: 相互作用パラメータ
        
        **ペア近似での自由エネルギー:**
        $$F = Jy_{AB} - T(S_{pair} - S_{point})$$
        
        **エントロピー項:**
        - $S_{pair} = -R[y_{AA}\\ln y_{AA} + y_{BB}\\ln y_{BB} + y_{AB}\\ln y_{AB}]$
        - $S_{point} = -R[x\\ln x + (1-x)\\ln(1-x)]$
        """)
    
    with tab4:
        st.subheader("相分離現象")
        st.markdown("""
        **相分離**は、均一な固溶体が二つの異なる組成の相に分かれる現象です。
        
        **重要な概念:**
        - **スピノーダル分解**: 均一相が不安定になる境界
        - **臨界温度**: 相分離が起こる最高温度
        - **混合ギャップ**: 二相共存領域
        
        **スピノーダル条件:**
        $$\\frac{\\partial^2 F}{\\partial x^2} = 0$$
        
        **相分離の駆動力:**
        - 低温: エンタルピー効果が支配的（相分離促進）
        - 高温: エントロピー効果が支配的（混合促進）
        """)

def create_interactive_calculator():
    """インタラクティブ計算セクション"""
    st.header("🧮 インタラクティブCVM計算")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("パラメータ設定")
        
        temperature = st.slider(
            "温度 [K]",
            min_value=200,
            max_value=1000,
            value=500,
            step=50,
            help="系の温度を設定します"
        )
        
        interaction_energy = st.slider(
            "相互作用エネルギー J [J/mol]",
            min_value=-2000,
            max_value=3000,
            value=1000,
            step=100,
            help="正の値：相分離傾向、負の値：規則化傾向"
        )
        
        composition = st.slider(
            "A原子濃度 x",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.01,
            help="A原子のモル分率"
        )
        
        approximation = st.selectbox(
            "近似レベル",
            ["点近似", "ペア近似"],
            help="計算に使用する近似レベル"
        )
    
    with col2:
        st.subheader("計算結果")
        
        cvm = CVMBinarySystem(interaction_energy, temperature)
        
        if approximation == "点近似":
            free_energy = cvm.point_approximation_free_energy(composition)
            st.metric("自由エネルギー", f"{free_energy:.2f} J/mol")
            
            if composition > 0 and composition < 1:
                S_config = -cvm.R * (composition * np.log(composition) + 
                                   (1-composition) * np.log(1-composition))
                st.metric("配置エントロピー", f"{S_config:.2f} J/(mol·K)")
            
            H = interaction_energy * composition * (1-composition)
            st.metric("エンタルピー", f"{H:.2f} J/mol")
            
        else:  # ペア近似
            y_AA, y_BB, y_AB = cvm.calculate_equilibrium_pairs(composition)
            free_energy = cvm.pair_approximation_free_energy(composition, y_AA, y_BB, y_AB)
            
            st.metric("自由エネルギー", f"{free_energy:.2f} J/mol")
            
            st.write("**平衡ペア確率:**")
            pair_df = pd.DataFrame({
                'ペア種類': ['A-A', 'B-B', 'A-B'],
                '確率': [y_AA, y_BB, y_AB],
                '理想値': [composition**2, (1-composition)**2, 2*composition*(1-composition)]
            })
            st.dataframe(pair_df, use_container_width=True)

def create_phase_diagram():
    """相図作成セクション"""
    st.header("📊 相図の計算と可視化")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("相図パラメータ")
        
        J_min = st.number_input("J最小値 [J/mol]", value=500, step=100)
        J_max = st.number_input("J最大値 [J/mol]", value=2000, step=100)
        
        T_min = st.number_input("温度最小値 [K]", value=300, step=50)
        T_max = st.number_input("温度最大値 [K]", value=800, step=50)
        
        resolution = st.selectbox("計算精度", [20, 50, 100], index=1)
        
        calculate_phase_diagram = st.button("相図計算実行", type="primary")
    
    with col2:
        if calculate_phase_diagram:
            st.subheader("相図結果")
            
            temperatures = np.linspace(T_min, T_max, resolution)
            compositions = np.linspace(0.01, 0.99, resolution)
            
            free_energy_map = np.zeros((len(temperatures), len(compositions)))
            spinodal_map = np.zeros((len(temperatures), len(compositions)))
            
            progress_bar = st.progress(0)
            
            for i, T in enumerate(temperatures):
                cvm = CVMBinarySystem(J_max, T)  # 固定相互作用エネルギー使用
                
                for j, x in enumerate(compositions):
                    free_energy_map[i, j] = cvm.point_approximation_free_energy(x)
                    
                    if x > 0.02 and x < 0.98:
                        dx = 0.01
                        f_plus = cvm.point_approximation_free_energy(x + dx)
                        f_minus = cvm.point_approximation_free_energy(x - dx)
                        f_center = cvm.point_approximation_free_energy(x)
                        
                        d2f_dx2 = (f_plus - 2*f_center + f_minus) / (dx**2)
                        spinodal_map[i, j] = 1 if d2f_dx2 < 0 else 0
                
                progress_bar.progress((i + 1) / len(temperatures))
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            im1 = axes[0, 0].imshow(free_energy_map, aspect='auto', origin='lower',
                                   extent=[compositions[0], compositions[-1], temperatures[0], temperatures[-1]],
                                   cmap='viridis')
            axes[0, 0].set_title('自由エネルギー分布')
            axes[0, 0].set_xlabel('組成 x')
            axes[0, 0].set_ylabel('温度 [K]')
            plt.colorbar(im1, ax=axes[0, 0], label='自由エネルギー [J/mol]')
            
            im2 = axes[0, 1].imshow(spinodal_map, aspect='auto', origin='lower',
                                   extent=[compositions[0], compositions[-1], temperatures[0], temperatures[-1]],
                                   cmap='RdBu')
            axes[0, 1].set_title('スピノーダル領域')
            axes[0, 1].set_xlabel('組成 x')
            axes[0, 1].set_ylabel('温度 [K]')
            plt.colorbar(im2, ax=axes[0, 1], label='スピノーダル指標')
            
            for i, T in enumerate(temperatures[::5]):  # 間引いて表示
                spinodal_x = []
                for j, x in enumerate(compositions):
                    if spinodal_map[i*5, j] > 0.5:
                        spinodal_x.append(x)
                
                if len(spinodal_x) > 0:
                    axes[1, 0].scatter(spinodal_x, [T] * len(spinodal_x), 
                                     c='red', s=10, alpha=0.6)
            
            axes[1, 0].set_title('温度-組成相図')
            axes[1, 0].set_xlabel('組成 x')
            axes[1, 0].set_ylabel('温度 [K]')
            axes[1, 0].grid(True, alpha=0.3)
            
            mid_temp_idx = len(temperatures) // 2
            axes[1, 1].plot(compositions, free_energy_map[mid_temp_idx, :], 
                           'b-', linewidth=2, label=f'T={temperatures[mid_temp_idx]:.0f}K')
            axes[1, 1].set_title('自由エネルギー曲線')
            axes[1, 1].set_xlabel('組成 x')
            axes[1, 1].set_ylabel('自由エネルギー [J/mol]')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("結果の解釈")
            st.markdown(f"""
            **計算条件:**
            - 相互作用エネルギー: {J_max} J/mol
            - 温度範囲: {T_min}-{T_max} K
            - 組成範囲: 0.01-0.99
            
            **相図の読み方:**
            - 赤い領域: スピノーダル分解が起こる不安定領域
            - 青い領域: 安定な固溶体領域
            - 境界線: スピノーダル線（相分離の境界）
            
            **物理的意味:**
            - 低温では相分離が起こりやすい
            - 高温では均一混合が安定
            - 組成x=0.5付近で最も不安定
            """)

def create_comparison_section():
    """近似手法比較セクション"""
    st.header("⚖️ 近似手法の比較")
    
    st.markdown("""
    異なる近似レベルでの計算結果を比較し、近似の精度と計算コストを理解します。
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("比較条件")
        
        comp_temperature = st.slider(
            "比較温度 [K]",
            min_value=200,
            max_value=1000,
            value=400,
            step=50
        )
        
        comp_interaction = st.slider(
            "比較用相互作用エネルギー [J/mol]",
            min_value=500,
            max_value=3000,
            value=1500,
            step=100
        )
        
        show_comparison = st.button("比較計算実行", type="primary")
    
    with col2:
        if show_comparison:
            st.subheader("近似手法比較結果")
            
            x_range = np.linspace(0.01, 0.99, 100)
            
            cvm = CVMBinarySystem(comp_interaction, comp_temperature)
            
            point_energies = []
            pair_energies = []
            
            for x in x_range:
                f_point = cvm.point_approximation_free_energy(x)
                point_energies.append(f_point)
                
                y_AA, y_BB, y_AB = cvm.calculate_equilibrium_pairs(x)
                f_pair = cvm.pair_approximation_free_energy(x, y_AA, y_BB, y_AB)
                pair_energies.append(f_pair)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.plot(x_range, point_energies, 'b-', linewidth=2, label='点近似')
            ax1.plot(x_range, pair_energies, 'r--', linewidth=2, label='ペア近似')
            ax1.set_xlabel('組成 x')
            ax1.set_ylabel('自由エネルギー [J/mol]')
            ax1.set_title(f'自由エネルギー比較 (T={comp_temperature}K)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            energy_diff = np.array(pair_energies) - np.array(point_energies)
            ax2.plot(x_range, energy_diff, 'g-', linewidth=2)
            ax2.set_xlabel('組成 x')
            ax2.set_ylabel('エネルギー差 [J/mol]')
            ax2.set_title('ペア近似 - 点近似')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("統計的比較")
            
            comparison_df = pd.DataFrame({
                '近似手法': ['点近似', 'ペア近似'],
                '最小自由エネルギー [J/mol]': [
                    f"{min(point_energies):.2f}",
                    f"{min(pair_energies):.2f}"
                ],
                '最大自由エネルギー [J/mol]': [
                    f"{max(point_energies):.2f}",
                    f"{max(pair_energies):.2f}"
                ],
                '平均自由エネルギー [J/mol]': [
                    f"{np.mean(point_energies):.2f}",
                    f"{np.mean(pair_energies):.2f}"
                ]
            })
            
            st.dataframe(comparison_df, use_container_width=True)
            
            st.markdown(f"""
            **比較結果の解釈:**
            
            - **エネルギー差の平均**: {np.mean(energy_diff):.2f} J/mol
            - **最大エネルギー差**: {max(abs(energy_diff)):.2f} J/mol
            
            **近似の特徴:**
            - **点近似**: 計算が簡単、平均場近似
            - **ペア近似**: より精密、局所相関を考慮
            - **差分**: ペア近似は一般的により低い自由エネルギーを与える
            
            **使い分け:**
            - 定性的理解: 点近似で十分
            - 定量的予測: ペア近似以上が必要
            - 相図計算: 高次近似が重要
            """)

def main():
    """メイン関数"""
    
    st.sidebar.header("📋 学習セクション")
    section = st.sidebar.selectbox(
        "学習したいセクションを選択:",
        [
            "理論の基礎",
            "インタラクティブ計算",
            "相図の計算",
            "近似手法の比較"
        ]
    )
    
    if section == "理論の基礎":
        create_theory_explanation()
    elif section == "インタラクティブ計算":
        create_interactive_calculator()
    elif section == "相図の計算":
        create_phase_diagram()
    elif section == "近似手法の比較":
        create_comparison_section()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **参考文献:**
    - Kikuchi, R. (1951). A theory of cooperative phenomena. Physical Review, 81(6), 988-1003.
    - Sanchez, J. M., & de Fontaine, D. (1978). The fcc Ising model in the cluster variation approximation. Physical Review B, 17(7), 2926-2936.
    
    **学習のポイント:**
    1. 格子統計の基本概念を理解する
    2. 近似レベルの違いを把握する  
    3. 相図の物理的意味を考える
    4. 実際の合金系への応用を考察する
    """)

if __name__ == "__main__":
    main()
