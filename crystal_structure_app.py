"""
結晶構造解析アプリケーション
Crystal Structure Analysis Application

幾何学と対称性に基づく結晶構造の無秩序性解析
Geometric and Symmetry-based Crystal Disorder Analysis

Based on Warren-Cowley Short Range Order (SRO) parameters
with E(3) invariance (Euclidean transformation invariance)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Dict

# Page configuration
st.set_page_config(
    page_title="結晶構造解析アプリ",
    page_icon="💎",
    layout="wide"
)

# ============================================================================
# CrystalGeometry Class: Mathematical Foundation
# ============================================================================


class CrystalGeometry:
    """
    結晶幾何学クラス

    Mathematical Definition:
    - Lattice Space: L = span(a, b, c)
    - Atomic State: s_i = x_i ⊗ f_i  (tensor product)
    - System State: S = ⊕_{i=1}^N s_i  (direct sum)

    where:
    - x_i ∈ ℝ³: position vector
    - f_i ∈ {0, 1}: chemical species (0=A, 1=B)
    """

    def __init__(self, structure_type: str = "FCC", size: int = 2):
        """
        Initialize crystal geometry

        Args:
            structure_type: "SC" (Simple Cubic), "BCC" (Body-Centered Cubic),
                          "FCC" (Face-Centered Cubic)
            size: Supercell size (N×N×N), default=2 for data reduction
        """
        self.structure_type = structure_type
        self.size = size
        self.lattice_constant = 1.0  # Normalized to 1.0

        # Generate lattice vectors (basis vectors)
        self.basis_vectors = self._get_basis_vectors()

        # Generate atomic positions
        self.positions = self._generate_positions()

        # Initialize chemical species (will be set by assign_species)
        self.species = np.zeros(len(self.positions), dtype=int)

    def _get_basis_vectors(self) -> np.ndarray:
        """
        Get basis vectors for lattice space
        L = span(a, b, c)

        Returns:
            3×3 array where each row is a basis vector
        """
        a = self.lattice_constant

        # Standard cubic basis
        return np.array([
            [a, 0, 0],  # a vector
            [0, a, 0],  # b vector
            [0, 0, a]   # c vector
        ])

    def _generate_positions(self) -> np.ndarray:
        """
        Generate atomic positions for supercell

        Returns:
            N×3 array of atomic positions
        """
        positions = []
        a = self.lattice_constant

        # Generate supercell grid
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    base_pos = np.array([i*a, j*a, k*a])

                    if self.structure_type == "SC":
                        # Simple Cubic: 1 atom per unit cell
                        positions.append(base_pos)

                    elif self.structure_type == "BCC":
                        # Body-Centered Cubic: 2 atoms per unit cell
                        positions.append(base_pos)  # Corner
                        positions.append(base_pos + np.array([a / 2, a / 2, a / 2]))  # Center

                    elif self.structure_type == "FCC":
                        # Face-Centered Cubic: 4 atoms per unit cell
                        positions.append(base_pos)  # Corner
                        positions.append(base_pos + np.array([a / 2, a / 2, 0]))  # Face 1
                        positions.append(base_pos + np.array([a / 2, 0, a / 2]))  # Face 2
                        positions.append(base_pos + np.array([0, a / 2, a / 2]))  # Face 3

        return np.array(positions)

    def assign_species(self, concentration_B: float, random_seed: int = None):
        """
        Assign chemical species to atoms

        Args:
            concentration_B: Concentration of B atoms (0.0 to 1.0)
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n_atoms = len(self.positions)
        n_B_atoms = int(n_atoms * concentration_B)

        # Initialize all as A (0)
        self.species = np.zeros(n_atoms, dtype=int)

        # Randomly select B atoms
        if n_B_atoms > 0:
            b_indices = np.random.choice(n_atoms, size=n_B_atoms, replace=False)
            self.species[b_indices] = 1

    def get_coordination_number(self) -> int:
        """
        Get coordination number (number of nearest neighbors)

        Returns:
            Coordination number for the structure type
        """
        coord_numbers = {
            "SC": 6,   # 6 nearest neighbors
            "BCC": 8,  # 8 nearest neighbors
            "FCC": 12  # 12 nearest neighbors
        }
        return coord_numbers[self.structure_type]

    def calculate_neighbor_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate pairwise distances between all atoms

        Returns:
            distance_matrix: N×N matrix of distances
            nearest_neighbor_distance: Distance to nearest neighbor
        """
        # Calculate pairwise distances using scipy
        distance_matrix = squareform(pdist(self.positions, metric='euclidean'))

        # Find nearest neighbor distance (excluding self, distance=0)
        # Set diagonal to infinity to exclude self-distances
        distance_matrix_copy = distance_matrix.copy()
        np.fill_diagonal(distance_matrix_copy, np.inf)
        nearest_neighbor_distance = np.min(distance_matrix_copy)

        return distance_matrix, nearest_neighbor_distance

    def get_state_representation(self) -> Dict:
        """
        Get system state representation
        S = ⊕_{i=1}^N (x_i ⊗ f_i)

        Returns:
            Dictionary containing:
            - positions: x_i ∈ ℝ³
            - species: f_i ∈ {0, 1}
            - state: Combined representation
        """
        return {
            'positions': self.positions,
            'species': self.species,
            'n_atoms': len(self.positions),
            'n_A': np.sum(self.species == 0),
            'n_B': np.sum(self.species == 1),
            'concentration_B': np.mean(self.species)
        }


# ============================================================================
# Warren-Cowley SRO Calculator: E(3) Invariant Order Parameter
# ============================================================================

class WarrenCowleySRO:
    """
    Warren-Cowley Short Range Order Parameter Calculator

    Mathematical Definition:
    α_n = 1 - P_n(B|A) / c_B

    where:
    - P_n(B|A): Conditional probability of finding B at n-th neighbor of A
    - c_B: Overall concentration of B
    - α ≈ 0: Random (disordered)
    - α < 0: Ordered (A-B alternating)
    - α > 0: Clustering (A-A, B-B preference)

    E(3) Invariance:
    α is a scalar quantity invariant under:
    - Rotations
    - Translations
    - Reflections
    """

    def __init__(self, crystal: CrystalGeometry):
        self.crystal = crystal
        self.distance_matrix, self.nn_distance = crystal.calculate_neighbor_distances()

    def calculate_alpha(self, shell: int = 1, tolerance: float = 0.1) -> float:
        """
        Calculate Warren-Cowley SRO parameter for given neighbor shell

        Args:
            shell: Neighbor shell number (1 = nearest neighbors)
            tolerance: Distance tolerance for identifying neighbors

        Returns:
            α: Warren-Cowley SRO parameter
        """
        # Get target distance for this shell
        target_distance = shell * self.nn_distance

        # Find neighbors within tolerance
        neighbor_mask = np.abs(self.distance_matrix - target_distance) < tolerance

        # Calculate overall concentration of B
        c_B = np.mean(self.crystal.species)

        # Handle edge cases
        if c_B == 0.0 or c_B == 1.0:
            return 0.0  # No disorder when all atoms are same type

        # Count A atoms and their B neighbors
        A_atoms = np.where(self.crystal.species == 0)[0]

        if len(A_atoms) == 0:
            return 0.0

        total_neighbors = 0
        B_neighbors = 0

        for a_idx in A_atoms:
            # Get neighbors of this A atom
            neighbors = np.where(neighbor_mask[a_idx])[0]
            total_neighbors += len(neighbors)

            # Count B neighbors
            B_neighbors += np.sum(self.crystal.species[neighbors] == 1)

        if total_neighbors == 0:
            return 0.0

        # Calculate conditional probability P(B|A)
        P_B_given_A = B_neighbors / total_neighbors

        # Calculate Warren-Cowley parameter
        alpha = 1.0 - (P_B_given_A / c_B)

        return alpha

    def interpret_alpha(self, alpha: float) -> str:
        """
        Interpret the meaning of α value

        Args:
            alpha: Warren-Cowley SRO parameter

        Returns:
            Interpretation string
        """
        if abs(alpha) < 0.1:
            return "Random Structure (無秩序構造)"
        elif alpha < -0.1:
            return "Ordered Structure (規則構造: A-B alternating)"
        else:  # alpha > 0.1
            return "Clustered Structure (クラスター構造: A-A, B-B preference)"


# ============================================================================
# Streamlit UI: Single Page Layout
# ============================================================================

def main():
    st.title("💎 幾何学と対称性で見る結晶構造の無秩序性")
    st.markdown("""
    このアプリケーションは、結晶構造の無秩序性（Disorder）を**幾何学的定義**と**対称性**の観点から解析します。
    Warren-Cowley Short Range Order (SRO) パラメータを用いて、E(3)不変な秩序パラメータを計算します。
    """)

    # ========================================================================
    # Section 1: Mathematical Definition
    # ========================================================================
    st.header("1. 数学的定義 (Mathematical Definition)")

    st.markdown("""
    結晶構造は、線形代数の**Span（生成空間）**と**直和・直積**を用いて厳密に記述されます。
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("格子空間 (Lattice Space)")
        st.latex(r"\mathcal{L} = \text{span}(\mathbf{a}, \mathbf{b}, \mathbf{c})")
        st.markdown("""
        基本並進ベクトル $\\mathbf{a}, \\mathbf{b}, \\mathbf{c}$ によって張られる空間。
        """)

        st.subheader("原子の状態 (Atomic State)")
        st.latex(r"s_i = x_i \otimes f_i")
        st.markdown(r"""
        - $x_i \in \mathbb{R}^3$: 位置座標
        - $f_i \in \{0, 1\}$: 化学種属性 (0=A, 1=B)
        - $\otimes$: **直積（Tensor Product）**
        """)

    with col2:
        st.subheader("全系の状態 (System State)")
        st.latex(r"S = \bigoplus_{i=1}^{N} s_i = \bigoplus_{i=1}^{N} (x_i \otimes f_i)")
        st.markdown("""
        結晶全体の状態は、全原子状態の**直和（Direct Sum）**。
        """)

        st.subheader("秩序パラメータ (Order Parameter)")
        st.latex(r"\alpha_n = 1 - \frac{P_n(B|A)}{c_B}")
        st.markdown(r"""
        - $\alpha \approx 0$: ランダム（無秩序）
        - $\alpha < 0$: 秩序化（A-B交互）
        - $\alpha > 0$: クラスタリング（同種凝集）

        **E(3)不変性**: 回転・並進・反射に対して不変なスカラー量
        """)

    st.info("""
    💡 **Data Reduction原則**: 初心者向けに、デフォルトは2×2×2セル（原子数8〜32個）で計算します。
    アルゴリズムの正しさは、データ数Nによらず確認できます。
    """)

    # ========================================================================
    # Section 2: Interactive Mode
    # ========================================================================
    st.header("2. インタラクティブモード (Interactive Mode)")

    # Sidebar controls
    st.sidebar.header("💎 Crystal Geometry App")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Basic Settings")

    structure_type = st.sidebar.selectbox(
        "結晶構造 (Crystal Structure)",
        ["FCC", "BCC", "SC"],
        help="FCC: Face-Centered Cubic, BCC: Body-Centered Cubic, SC: Simple Cubic"
    )

    size = st.sidebar.selectbox(
        "サイズ (Size N×N×N)",
        [2, 3, 4],
        index=0,
        help="2×2×2 is recommended for fast computation (Data Reduction)"
    )

    concentration = st.sidebar.slider(
        "B原子濃度 (Concentration of B)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Concentration of B atoms (0.0 = all A, 1.0 = all B)"
    )

    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="For reproducibility"
    )

    # Generate crystal structure
    crystal = CrystalGeometry(structure_type=structure_type, size=size)
    crystal.assign_species(concentration_B=concentration, random_seed=random_seed)

    # Get state representation
    state = crystal.get_state_representation()

    # Display structure information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総原子数 (Total Atoms)", state['n_atoms'])
    with col2:
        st.metric("A原子数", state['n_A'])
    with col3:
        st.metric("B原子数", state['n_B'])
    with col4:
        st.metric("配位数 (Coordination)", crystal.get_coordination_number())

    # 3D Visualization
    st.subheader("3D可視化 (3D Visualization)")

    # Create 3D scatter plot
    positions = crystal.positions
    species = crystal.species

    # Separate A and B atoms
    A_atoms = positions[species == 0]
    B_atoms = positions[species == 1]

    fig = go.Figure()

    # Add A atoms (blue)
    if len(A_atoms) > 0:
        fig.add_trace(go.Scatter3d(
            x=A_atoms[:, 0],
            y=A_atoms[:, 1],
            z=A_atoms[:, 2],
            mode='markers',
            name='A atoms',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.8
            )
        ))

    # Add B atoms (red)
    if len(B_atoms) > 0:
        fig.add_trace(go.Scatter3d(
            x=B_atoms[:, 0],
            y=B_atoms[:, 1],
            z=B_atoms[:, 2],
            mode='markers',
            name='B atoms',
            marker=dict(
                size=8,
                color='red',
                opacity=0.8
            )
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        height=600,
        title=f"{structure_type} Structure (Size: {size}×{size}×{size})"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    🔄 **E(3)対称性の確認**: マウスで3Dグラフを回転させても、原子のつながりは変わりません。
    これが**E(3)不変性（ユークリッド変換に対する不変性）**です。
    """)

    # Calculate SRO parameter
    st.subheader("秩序パラメータ計算 (Order Parameter Calculation)")

    sro_calculator = WarrenCowleySRO(crystal)
    alpha = sro_calculator.calculate_alpha(shell=1)
    interpretation = sro_calculator.interpret_alpha(alpha)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Warren-Cowley SRO (α)", f"{alpha:.4f}")
    with col2:
        st.metric("判定 (Interpretation)", interpretation)

    st.markdown(f"""
    **計算詳細:**
    - 第1近接殻（Nearest Neighbors）での計算
    - 濃度 c_B = {concentration:.3f}
    - α = {alpha:.4f}
    """)

    # ========================================================================
    # Section 3: Comprehensive Verification Mode (Sweep & Verify)
    # ========================================================================
    st.header("3. 網羅的検証モード (Comprehensive Verification Mode)")

    st.markdown("""
    パラメータ範囲を指定し、**全パターンを総当たりで検証**します。
    リアルタイムでグラフが更新され、統計的な傾向やエッジケースを発見できます。
    """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced: Parameter Sweep")

    enable_sweep = st.sidebar.checkbox("Enable Sweep Mode", value=False)

    if enable_sweep:
        sweep_structures = st.sidebar.multiselect(
            "検証する結晶構造",
            ["SC", "BCC", "FCC"],
            default=["FCC"]
        )

        resolution = st.sidebar.slider(
            "濃度の刻み数 (Resolution)",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of concentration points to test"
        )

        trials = st.sidebar.slider(
            "各点での試行回数 (Trials)",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of random configurations per concentration"
        )

        sweep_size = st.sidebar.selectbox(
            "Sweep用サイズ",
            [2, 3],
            index=0,
            help="Keep small for faster computation"
        )

        if st.sidebar.button("🚀 Run Validation", type="primary"):
            st.subheader("検証実行中... (Running Validation)")

            # Prepare concentration range
            concentrations = np.linspace(0.05, 0.95, resolution)

            # Prepare results storage
            results = {struct: {'conc': [], 'alpha_mean': [], 'alpha_std': []}
                      for struct in sweep_structures}

            # Progress tracking
            total_iterations = len(sweep_structures) * len(concentrations) * trials
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Real-time chart placeholder
            chart_placeholder = st.empty()

            iteration = 0

            # Sweep over structures
            for struct in sweep_structures:
                for conc in concentrations:
                    alpha_values = []

                    # Multiple trials for statistical averaging
                    for trial in range(trials):
                        # Update progress
                        iteration += 1
                        progress = iteration / total_iterations
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Computing {struct} at Conc={conc:.2f}, Trial {trial+1}/{trials}... "
                            f"({iteration}/{total_iterations})"
                        )

                        # Generate crystal and calculate α
                        crystal_sweep = CrystalGeometry(
                            structure_type=struct,
                            size=sweep_size
                        )
                        crystal_sweep.assign_species(
                            concentration_B=conc,
                            random_seed=random_seed + iteration
                        )

                        sro_sweep = WarrenCowleySRO(crystal_sweep)
                        alpha_sweep = sro_sweep.calculate_alpha(shell=1)
                        alpha_values.append(alpha_sweep)

                    # Store statistics
                    results[struct]['conc'].append(conc)
                    results[struct]['alpha_mean'].append(np.mean(alpha_values))
                    results[struct]['alpha_std'].append(np.std(alpha_values))

                    # Update chart in real-time
                    fig_sweep = go.Figure()

                    for s in sweep_structures:
                        if len(results[s]['conc']) > 0:
                            fig_sweep.add_trace(go.Scatter(
                                x=results[s]['conc'],
                                y=results[s]['alpha_mean'],
                                mode='lines+markers',
                                name=s,
                                error_y=dict(
                                    type='data',
                                    array=results[s]['alpha_std'],
                                    visible=True
                                )
                            ))

                    fig_sweep.add_hline(y=0, line_dash="dash", line_color="gray",
                                       annotation_text="α=0 (Random)")

                    fig_sweep.update_layout(
                        title="Warren-Cowley SRO vs Concentration",
                        xaxis_title="Concentration of B",
                        yaxis_title="α (SRO Parameter)",
                        height=500
                    )

                    chart_placeholder.plotly_chart(fig_sweep, use_container_width=True)

            progress_bar.progress(1.0)
            status_text.text("✅ 検証完了！ (Validation Complete!)")

            st.success("全パターンの検証が完了しました。")

            # Display results table
            st.subheader("検証結果サマリー (Validation Summary)")

            for struct in sweep_structures:
                st.markdown(f"**{struct} Structure:**")
                df_results = pd.DataFrame({
                    'Concentration': results[struct]['conc'],
                    'α (mean)': results[struct]['alpha_mean'],
                    'α (std)': results[struct]['alpha_std']
                })
                st.dataframe(df_results, use_container_width=True)

    # ========================================================================
    # Section 4: Testing Strategy & Verification Matrix
    # ========================================================================
    st.header("4. 検証マトリクス (Testing Strategy)")

    st.markdown("""
    以下の挙動を「Sweep Mode」で確認してください：
    """)

    verification_matrix = pd.DataFrame({
        'ケース': [
            '境界値',
            '統計性',
            '構造差',
            'ロバスト性'
        ],
        '設定条件': [
            '濃度 0.0 または 1.0',
            '濃度 0.5, 試行回数 10',
            'SC vs FCC',
            '分割数 50 (高負荷)'
        ],
        '期待される挙動': [
            'エラーが起きない、α=0となる',
            'ランダム配置でα≈0に収束',
            '配位数の違いにより結果に差',
            'アプリが落ちずに完走'
        ]
    })

    st.dataframe(verification_matrix, use_container_width=True)

    # ========================================================================
    # Section 5: Comprehensive Theory and Documentation
    # ========================================================================
    st.header("5. 詳細理論解説 (Comprehensive Theory)")

    with st.expander("📚 理論的背景と数学的基礎", expanded=False):
        st.markdown("""
        ### 5.1 結晶構造の幾何学的記述

        結晶構造は、**線形代数**と**群論**を用いて厳密に記述されます。

        #### 格子空間 (Lattice Space)

        結晶格子は、3つの基本並進ベクトル $\\mathbf{a}, \\mathbf{b}, \\mathbf{c}$ によって張られる空間として定義されます：

        $$\\mathcal{L} = \\text{span}(\\mathbf{a}, \\mathbf{b}, \\mathbf{c}) = \\{n_1\\mathbf{a} + n_2\\mathbf{b} + n_3\\mathbf{c} \\mid n_1, n_2, n_3 \\in \\mathbb{Z}\\}$$

        この定義により、結晶格子は**離散的な並進対称性**を持つことが保証されます。

        #### 直積と直和の意味

        **直積 (Tensor Product) $\\otimes$:**

        原子の状態 $s_i$ は、位置情報 $x_i \\in \\mathbb{R}^3$ と化学種情報 $f_i \\in \\{0, 1\\}$ の直積として表現されます：

        $$s_i = x_i \\otimes f_i$$

        これは、位置と化学種が**独立した自由度**であることを意味します。プログラム実装では、
        これは別々の配列 `positions[i]` と `species[i]` として表現されます。

        **直和 (Direct Sum) $\\bigoplus$:**

        結晶全体の状態 $S$ は、全原子状態の直和として表現されます：

        $$S = \\bigoplus_{i=1}^{N} s_i$$

        これは、各原子が**独立した自由度**を持ち、全系の状態空間が各原子の状態空間の
        直和であることを意味します。プログラム実装では、これは配列のリストとして表現されます。

        ### 5.2 Warren-Cowley SRO パラメータの物理的意味

        #### 定義と導出

        Warren-Cowley Short Range Order (SRO) パラメータ $\\alpha_n$ は、第 $n$ 近接殻における
        化学的短範囲秩序を定量化する指標です：

        $$\\alpha_n = 1 - \\frac{P_n(B|A)}{c_B}$$

        ここで：
        - $P_n(B|A)$: A原子の第 $n$ 近接位置にB原子が存在する条件付き確率
        - $c_B$: B原子の全体濃度

        #### 物理的解釈

        1. **$\\alpha = 0$ (ランダム構造):**
           - $P_n(B|A) = c_B$ となり、A原子の周りのB原子分布が全体濃度と一致
           - 統計的に完全にランダムな配置
           - エントロピーが最大

        2. **$\\alpha < 0$ (規則構造):**
           - $P_n(B|A) > c_B$ となり、A原子の周りにB原子が優先的に配置
           - A-B交互配列（規則合金）
           - エンタルピー的に安定（負の混合エンタルピー）

        3. **$\\alpha > 0$ (クラスター構造):**
           - $P_n(B|A) < c_B$ となり、A原子の周りにA原子が優先的に配置
           - 相分離傾向（クラスタリング）
           - 正の混合エンタルピー

        ### 5.3 E(3) 不変性と対称性

        #### ユークリッド群 E(3)

        E(3)は、3次元ユークリッド空間における**等長変換**（距離を保存する変換）の群です：

        $$E(3) = \\{(R, \\mathbf{t}) \\mid R \\in SO(3), \\mathbf{t} \\in \\mathbb{R}^3\\}$$

        ここで：
        - $R$: 回転行列（$SO(3)$: 特殊直交群）
        - $\\mathbf{t}$: 並進ベクトル

        #### E(3) 不変性の重要性

        Warren-Cowley SRO パラメータ $\\alpha$ は、**E(3)不変なスカラー量**です：

        $$\\alpha(R\\{x_i\\} + \\mathbf{t}) = \\alpha(\\{x_i\\})$$

        これは、結晶を回転・並進しても $\\alpha$ の値が変わらないことを意味します。

        **物理的意義:**
        - 結晶の向きや位置に依存しない本質的な性質
        - 実験測定値との対応が明確
        - 機械学習における**幾何学的深層学習 (Geometric Deep Learning)** の基礎

        ### 5.4 結晶構造タイプと配位数

        #### Simple Cubic (SC)
        - **配位数**: 6
        - **充填率**: 52.4%
        - **例**: Po (ポロニウム)

        #### Body-Centered Cubic (BCC)
        - **配位数**: 8 (第1近接), 6 (第2近接)
        - **充填率**: 68.0%
        - **例**: Fe, Cr, W

        #### Face-Centered Cubic (FCC)
        - **配位数**: 12
        - **充填率**: 74.0% (最密充填)
        - **例**: Al, Cu, Au, Ni

        ### 5.5 Data Reduction の原理

        本アプリでは、**Data Reduction原則**を採用しています：

        **原理:**
        - アルゴリズムの正しさは、データサイズ $N$ に依存しない
        - 小さなサンプル（$2\\times2\\times2$）で本質的な挙動を確認できる
        - 計算コストを削減し、学習効率を最大化

        **数学的根拠:**
        - SROパラメータは**局所的な相関**を測定
        - 統計的性質は、十分なサンプリングで収束
        - 境界効果は、周期境界条件で緩和可能

        ### 5.6 実装上の注意点

        #### 距離計算の効率化

        本アプリでは、`scipy.spatial.distance.pdist` を使用して効率的に距離行列を計算：

        ```python
        distance_matrix = squareform(pdist(positions, metric='euclidean'))
        ```

        これにより、$O(N^2)$ の計算を高速化しています。

        #### 境界値処理

        濃度 $c_B = 0$ または $c_B = 1$ の場合、$\\alpha$ の計算でゼロ除算が発生する可能性があります。
        本実装では、これらのケースを明示的に処理し、$\\alpha = 0$ を返します。

        #### 統計的平均化

        ランダム配置の場合、単一の配置では統計的ゆらぎが大きくなります。
        **Sweep Mode** では、複数回の試行を平均化することで、信頼性の高い結果を得ます。
        """)

    with st.expander("🔬 使用方法とベストプラクティス", expanded=False):
        st.markdown("""
        ### 使用方法ガイド

        #### 1. Interactive Mode（インタラクティブモード）

        **目的:** 単一の条件で詳細な解析を行う

        **手順:**
        1. サイドバーで結晶構造タイプを選択（FCC推奨）
        2. サイズを選択（2×2×2推奨：高速計算）
        3. B原子濃度をスライダーで調整
        4. 3D可視化で原子配置を確認
        5. SROパラメータの値と解釈を確認

        **ヒント:**
        - マウスで3Dグラフを回転させ、E(3)不変性を体感
        - Random Seedを変更して、異なる配置を試す
        - 濃度0.5付近でランダム構造を観察

        #### 2. Sweep & Verify Mode（網羅的検証モード）

        **目的:** パラメータ空間を網羅的に探索し、統計的傾向を把握

        **手順:**
        1. サイドバーで「Enable Sweep Mode」をチェック
        2. 検証する結晶構造を選択（複数可）
        3. 濃度の刻み数を設定（20推奨）
        4. 各点での試行回数を設定（5〜10推奨）
        5. 「Run Validation」ボタンをクリック
        6. リアルタイムでグラフが更新されるのを観察

        **ヒント:**
        - 計算時間は、(構造数 × 刻み数 × 試行回数) に比例
        - 高解像度（刻み数50）は、最終確認時のみ使用
        - 複数構造を同時に比較し、配位数の影響を観察

        #### 3. 検証マトリクスの活用

        以下の項目を必ず確認してください：

        | 検証項目 | 設定 | 期待される結果 |
        |---------|------|---------------|
        | 境界値テスト | 濃度 0.0, 1.0 | エラーなし、α=0 |
        | ランダム性確認 | 濃度 0.5, 試行10回 | α≈0に収束 |
        | 構造依存性 | SC vs FCC | 配位数の違いによる差 |
        | ロバスト性 | 刻み数50 | アプリが完走 |

        ### ベストプラクティス

        #### 学習段階

        1. **まず小さく始める:** 2×2×2、濃度0.5から開始
        2. **可視化を活用:** 3Dグラフで直感的理解を深める
        3. **パラメータを変化:** 濃度を0.1刻みで変更し、αの変化を観察
        4. **統計性を確認:** Random Seedを変更し、ゆらぎを体感

        #### 研究段階

        1. **Sweep Modeで全体像把握:** 濃度全域でのαの振る舞いを確認
        2. **複数構造の比較:** SC, BCC, FCCの違いを定量化
        3. **統計的信頼性:** 試行回数を増やし、誤差範囲を評価
        4. **結果のエクスポート:** データフレームをCSVで保存

        ### トラブルシューティング

        **Q: αの値が常に0になる**
        - A: 濃度が0.0または1.0の場合、これは正常です
        - A: Random Seedを変更して、異なる配置を試してください

        **Q: Sweep Modeが遅い**
        - A: サイズを2×2×2に設定してください
        - A: 刻み数と試行回数を減らしてください

        **Q: 3D可視化が表示されない**
        - A: ブラウザを更新してください
        - A: Plotlyがサポートされているブラウザを使用してください
        """)

    with st.expander("📖 参考文献と発展的学習", expanded=False):
        st.markdown("""
        ### 主要参考文献

        #### 結晶学・材料科学

        1. **Warren, B. E. (1969).** *X-Ray Diffraction.* Dover Publications.
           - X線回折による結晶構造解析の古典的名著
           - Warren-Cowley SROパラメータの原典

        2. **Cowley, J. M. (1950).** "An Approximate Theory of Order in Alloys."
           *Physical Review*, 77(5), 669-675.
           - SROパラメータの理論的基礎

        3. **Kittel, C. (2004).** *Introduction to Solid State Physics* (8th ed.). Wiley.
           - 結晶構造と対称性の基礎

        #### 幾何学的深層学習

        4. **Bronstein, M. M., et al. (2021).** "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges."
           *arXiv:2104.13478*
           - E(3)不変性と幾何学的深層学習の包括的レビュー

        5. **Thomas, N., et al. (2018).** "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds."
           *arXiv:1802.08219*
           - E(3)等変ニューラルネットワークの基礎

        6. **Batzner, S., et al. (2022).** "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials."
           *Nature Communications*, 13, 2453.
           - 材料科学への応用例

        ### 発展的学習トピック

        #### 1. より高度な秩序パラメータ

        - **多体相関関数:** 3体、4体相関の計算
        - **動径分布関数 (RDF):** 連続的な距離依存性
        - **構造因子 S(q):** 逆格子空間での解析

        #### 2. 統計力学との接続

        - **Ising模型:** 格子上のスピン系との対応
        - **Cluster Variation Method (CVM):** より精密な自由エネルギー計算
        - **Monte Carlo法:** 熱平衡状態のシミュレーション

        #### 3. 機械学習への応用

        - **Graph Neural Networks (GNN):** 結晶構造の表現学習
        - **E(3)等変ネットワーク:** 対称性を保存するニューラルネットワーク
        - **Materials Informatics:** 物性予測と材料設計

        ### オンラインリソース

        - **Materials Project:** https://materialsproject.org/
          - 結晶構造データベース

        - **Crystallography Open Database:** http://www.crystallography.net/
          - オープンアクセスの結晶構造データ

        - **E3NN (PyTorch):** https://e3nn.org/
          - E(3)等変ニューラルネットワークのライブラリ
        """)

    # ========================================================================
    # Footer
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    **開発コンセプト:**
    - **Math-First**: 数学的定義をコードに反映
    - **Data Reduction**: 計算コストを抑え、本質の理解を優先
    - **Visual Verification**: 数値だけでなく、GUI上で網羅的に検証

    **バージョン:** 1.0.0
    **開発:** Devin AI
    **ライセンス:** MIT License
    """)


if __name__ == "__main__":
    main()
