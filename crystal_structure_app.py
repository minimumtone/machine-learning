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

    def __init__(self, structure_type: str = "FCC", size: int = 2, 
                 positional_sigma: float = 0.0, random_seed: int = None):
        """
        Initialize crystal geometry

        Args:
            structure_type: "SC" (Simple Cubic), "BCC" (Body-Centered Cubic),
                          "FCC" (Face-Centered Cubic)
            size: Supercell size (N×N×N), default=2 for data reduction
            positional_sigma: Positional disorder parameter (σ/a), 0.0 = perfect lattice
            random_seed: Random seed for positional disorder
        """
        self.structure_type = structure_type
        self.size = size
        self.lattice_constant = 1.0  # Normalized to 1.0
        self.positional_sigma = positional_sigma
        self.random_seed = random_seed

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

        positions = np.array(positions)
        
        # Add positional disorder if requested
        if self.positional_sigma > 0:
            rng = np.random.default_rng(self.random_seed)
            sigma_abs = self.positional_sigma * self.lattice_constant
            noise = rng.normal(scale=sigma_abs, size=positions.shape)
            positions = positions + noise
            
        return positions

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
    α_n = 1 - P_n(B|A) / c_B  (shell-based)
    α_k = 1 - P_k(B|A) / c_B  (k-NN based)

    where:
    - P_n(B|A): Conditional probability of finding B at n-th neighbor shell of A
    - P_k(B|A): Conditional probability of finding B among k nearest neighbors of A
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
        self.n_atoms = len(crystal.positions)
        self.species = crystal.species

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
    
    def _compute_knn_indices(self, k: int) -> np.ndarray:
        """
        Compute k-nearest neighbor indices for each atom
        
        Args:
            k: Number of nearest neighbors
            
        Returns:
            N×k array of neighbor indices
        """
        knn_indices = np.zeros((self.n_atoms, k), dtype=int)
        
        for i in range(self.n_atoms):
            # Get distances from atom i to all others
            distances = self.distance_matrix[i].copy()
            # Set self-distance to infinity to exclude it
            distances[i] = np.inf
            # Get indices of k nearest neighbors
            knn_indices[i] = np.argsort(distances)[:k]
            
        return knn_indices
    
    def calculate_alpha_knn(self, k: int) -> float:
        """
        Calculate Warren-Cowley SRO parameter using k-nearest neighbors
        
        Mathematical Definition:
        For each A atom i, consider its k nearest neighbors N_k(i).
        p_i(B|A; k) = (# of B atoms in N_k(i)) / k
        P_k(B|A) = (1/N_A) Σ_{i ∈ A} p_i(B|A; k)
        α_k = 1 - P_k(B|A) / c_B
        
        Args:
            k: Number of nearest neighbors (e.g., k=5 for "5nn")
            
        Returns:
            α_k: k-NN Warren-Cowley SRO parameter
        """
        # Calculate overall concentration of B
        c_B = np.mean(self.species)
        
        # Handle edge cases
        if c_B == 0.0 or c_B == 1.0:
            return 0.0  # No disorder when all atoms are same type
        
        # Get A atom indices
        A_atoms = np.where(self.species == 0)[0]
        
        if len(A_atoms) == 0:
            return 0.0
        
        # Compute k-NN indices for all atoms
        knn_indices = self._compute_knn_indices(k)
        
        # Calculate P_k(B|A)
        p_values = []
        for a_idx in A_atoms:
            neighbors = knn_indices[a_idx]
            # Count B atoms among k nearest neighbors
            n_B_neighbors = np.sum(self.species[neighbors] == 1)
            p_i = n_B_neighbors / k
            p_values.append(p_i)
        
        P_k_B_given_A = np.mean(p_values)
        
        # Calculate α_k
        alpha_k = 1.0 - (P_k_B_given_A / c_B)
        
        return alpha_k
    
    def calculate_alpha_knn_multi(self, k_values: list) -> dict:
        """
        Calculate α_k for multiple k values
        
        Args:
            k_values: List of k values (e.g., [1, 2, 3, 4, 5])
            
        Returns:
            Dictionary mapping k -> α_k
        """
        results = {}
        for k in k_values:
            results[k] = self.calculate_alpha_knn(k)
        return results


# ============================================================================
# Structural SRO Calculator: Geometric Correlation
# ============================================================================

class StructuralSRO:
    """
    Structural Short Range Order Calculator
    
    Unlike chemical SRO (which measures species correlations on a fixed lattice),
    structural SRO measures correlations of geometric descriptors (bond lengths,
    coordination environments) that arise from positional disorder.
    
    Mathematical Definition:
    For a geometric descriptor q_i (e.g., mean distance to k nearest neighbors):
    β_k = ⟨(q_i - ⟨q⟩)(q_j - ⟨q⟩)⟩_{pairs} / Var(q)
    
    where pairs are k-nearest neighbors.
    
    Interpretation:
    - β ≈ 0: No structural correlation (random positional disorder)
    - β > 0: Similar environments cluster together
    - β < 0: Dissimilar environments are neighbors (anti-correlation)
    """
    
    def __init__(self, crystal: CrystalGeometry):
        self.crystal = crystal
        self.distance_matrix, self.nn_distance = crystal.calculate_neighbor_distances()
        self.n_atoms = len(crystal.positions)
        
    def _compute_geometric_descriptor(self, k: int) -> np.ndarray:
        """
        Compute geometric descriptor q_i for each atom
        
        q_i = mean distance to k nearest neighbors
        
        Args:
            k: Number of nearest neighbors
            
        Returns:
            Array of q_i values (length N)
        """
        q_values = np.zeros(self.n_atoms)
        
        for i in range(self.n_atoms):
            # Get distances from atom i to all others
            distances = self.distance_matrix[i].copy()
            # Set self-distance to infinity to exclude it
            distances[i] = np.inf
            # Get k nearest neighbor distances
            knn_distances = np.sort(distances)[:k]
            # Compute mean distance
            q_values[i] = np.mean(knn_distances)
            
        return q_values
    
    def calculate_structural_sro_knn(self, k: int) -> float:
        """
        Calculate structural SRO parameter using k-nearest neighbors
        
        Mathematical Definition:
        1. Compute geometric descriptor: q_i = mean distance to k nearest neighbors
        2. Compute mean and variance: ⟨q⟩, Var(q)
        3. For each atom i and its k nearest neighbors j:
           Compute (q_i - ⟨q⟩)(q_j - ⟨q⟩)
        4. Average over all pairs and normalize:
           β_k = ⟨(q_i - ⟨q⟩)(q_j - ⟨q⟩)⟩ / Var(q)
        
        Args:
            k: Number of nearest neighbors
            
        Returns:
            β_k: Structural SRO parameter
        """
        # Compute geometric descriptor for all atoms
        q_values = self._compute_geometric_descriptor(k)
        
        # Compute mean and variance
        q_mean = np.mean(q_values)
        q_var = np.var(q_values)
        
        # Handle edge case: no variance (perfect lattice)
        if q_var < 1e-10:
            return 0.0
        
        # Compute centered values
        q_centered = q_values - q_mean
        
        # Compute k-NN indices
        knn_indices = np.zeros((self.n_atoms, k), dtype=int)
        for i in range(self.n_atoms):
            distances = self.distance_matrix[i].copy()
            distances[i] = np.inf
            knn_indices[i] = np.argsort(distances)[:k]
        
        # Compute correlation over all neighbor pairs
        correlations = []
        for i in range(self.n_atoms):
            for j in knn_indices[i]:
                correlation = q_centered[i] * q_centered[j]
                correlations.append(correlation)
        
        # Average and normalize
        beta_k = np.mean(correlations) / q_var
        
        return beta_k
    
    def calculate_structural_sro_multi(self, k_values: list) -> dict:
        """
        Calculate β_k for multiple k values
        
        Args:
            k_values: List of k values (e.g., [1, 2, 3, 4, 5])
            
        Returns:
            Dictionary mapping k -> β_k
        """
        results = {}
        for k in k_values:
            results[k] = self.calculate_structural_sro_knn(k)
        return results
    
    def interpret_beta(self, beta: float) -> str:
        """
        Interpret the meaning of β value
        
        Args:
            beta: Structural SRO parameter
            
        Returns:
            Interpretation string
        """
        if abs(beta) < 0.1:
            return "No Structural Correlation (構造相関なし)"
        elif beta > 0.1:
            return "Positive Structural Correlation (類似環境がクラスター化)"
        else:  # beta < -0.1
            return "Negative Structural Correlation (異なる環境が隣接)"


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

    st.sidebar.markdown("---")
    st.sidebar.subheader("Structural Disorder")
    
    positional_sigma = st.sidebar.slider(
        "位置ゆらぎ σ/a (Positional Disorder)",
        min_value=0.0,
        max_value=0.1,
        value=0.0,
        step=0.01,
        help="Gaussian positional disorder parameter (σ/a). 0.0 = perfect lattice, >0 = structural disorder"
    )

    # Generate crystal structure
    crystal = CrystalGeometry(
        structure_type=structure_type, 
        size=size,
        positional_sigma=positional_sigma,
        random_seed=random_seed
    )
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

    # Calculate SRO parameters
    st.subheader("秩序パラメータ計算 (Order Parameter Calculation)")

    sro_calculator = WarrenCowleySRO(crystal)
    
    # Traditional shell-based SRO
    alpha_shell = sro_calculator.calculate_alpha(shell=1)
    interpretation_shell = sro_calculator.interpret_alpha(alpha_shell)
    
    # k-NN Chemical SRO (for k=1,2,3,4,5)
    k_values = [1, 2, 3, 4, 5]
    alpha_knn_results = sro_calculator.calculate_alpha_knn_multi(k_values)
    
    # Structural SRO (if positional disorder is present)
    if positional_sigma > 0:
        structural_sro = StructuralSRO(crystal)
        beta_knn_results = structural_sro.calculate_structural_sro_multi(k_values)
    
    # Display Chemical SRO
    st.markdown("### 化学的短範囲秩序 (Chemical SRO)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Shell-based α₁", f"{alpha_shell:.4f}")
        st.caption("Traditional nearest neighbor shell")
    with col2:
        st.metric("判定 (Interpretation)", interpretation_shell)
    
    # Display k-NN Chemical SRO
    st.markdown("**k-NN Chemical SRO (α_k for k=1,2,3,4,5):**")
    
    knn_cols = st.columns(5)
    for idx, k in enumerate(k_values):
        with knn_cols[idx]:
            st.metric(f"α_{k}", f"{alpha_knn_results[k]:.4f}")
    
    # Plot k-NN Chemical SRO
    fig_chem = go.Figure()
    fig_chem.add_trace(go.Scatter(
        x=k_values,
        y=[alpha_knn_results[k] for k in k_values],
        mode='lines+markers',
        name='Chemical SRO (α_k)',
        line=dict(color='blue', width=2),
        marker=dict(size=10)
    ))
    fig_chem.add_hline(y=0, line_dash="dash", line_color="gray", 
                       annotation_text="Random (α=0)")
    fig_chem.update_layout(
        title="k-NN Chemical SRO vs k",
        xaxis_title="k (number of nearest neighbors)",
        yaxis_title="α_k",
        height=400
    )
    st.plotly_chart(fig_chem, use_container_width=True)
    
    # Display Structural SRO if applicable
    if positional_sigma > 0:
        st.markdown("### 構造的短範囲秩序 (Structural SRO)")
        st.info("""
        構造SROは、位置ゆらぎによる幾何学的記述子（平均結合長など）の相関を測定します。
        化学SROとは異なり、格子の位置的無秩序に起因する構造相関を捉えます。
        """)
        
        # Display k-NN Structural SRO
        st.markdown("**k-NN Structural SRO (β_k for k=1,2,3,4,5):**")
        
        struct_cols = st.columns(5)
        for idx, k in enumerate(k_values):
            with struct_cols[idx]:
                st.metric(f"β_{k}", f"{beta_knn_results[k]:.4f}")
        
        # Plot k-NN Structural SRO
        fig_struct = go.Figure()
        fig_struct.add_trace(go.Scatter(
            x=k_values,
            y=[beta_knn_results[k] for k in k_values],
            mode='lines+markers',
            name='Structural SRO (β_k)',
            line=dict(color='green', width=2),
            marker=dict(size=10)
        ))
        fig_struct.add_hline(y=0, line_dash="dash", line_color="gray", 
                           annotation_text="No Correlation (β=0)")
        fig_struct.update_layout(
            title="k-NN Structural SRO vs k",
            xaxis_title="k (number of nearest neighbors)",
            yaxis_title="β_k",
            height=400
        )
        st.plotly_chart(fig_struct, use_container_width=True)
        
        st.markdown("""
        **解釈:**
        - β ≈ 0: 構造相関なし（ランダムな位置ゆらぎ）
        - β > 0: 類似環境がクラスター化
        - β < 0: 異なる環境が隣接（反相関）
        """)
    else:
        st.info("""
        💡 **構造SROを計算するには**: サイドバーの「位置ゆらぎ σ/a」を0より大きく設定してください。
        位置的無秩序がない場合、構造SROは定義されません。
        """)
    
    st.markdown(f"""
    **計算詳細:**
    - 濃度 c_B = {concentration:.3f}
    - 位置ゆらぎ σ/a = {positional_sigma:.3f}
    - Shell-based α₁ = {alpha_shell:.4f}
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

        ### 5.6 SQS と無秩序指標の設計：化学SRO vs 構造SRO

        #### Special Quasirandom Structure (SQS) とは

        **SQS（Special Quasirandom Structure）**は、有限サイズの超格子で完全ランダム合金の統計的性質を
        最もよく模倣する原子配置です。Zunger et al. (1990) によって提案され、第一原理計算における
        ランダム合金のモデル化に広く使用されています。

        #### 完全ランダム合金の数学的定義

        完全ランダム合金では、任意の近接殻 $n$ において：

        $$P_n(B|A) = c_B$$

        これは、A原子の周りのB原子の条件付き確率が、全体濃度 $c_B$ に等しいことを意味します。
        Warren-Cowley SROパラメータで表現すると：

        $$\\alpha_n = 1 - \\frac{P_n(B|A)}{c_B} = 1 - \\frac{c_B}{c_B} = 0$$

        つまり、**すべての近接殻で $\\alpha_n = 0$** が完全ランダムの条件です。

        #### 有限サイズの制約とSQSの必要性

        しかし、有限サイズの超格子では、すべての殻で同時に $\\alpha_n = 0$ を達成することは
        **数学的に不可能**です。これは以下の理由によります：

        1. **離散性の制約**: 原子数が有限のため、濃度を厳密に実現できない
        2. **幾何学的制約**: 周期境界条件により、遠方の殻で相関が生じる
        3. **組み合わせ的制約**: 配置の自由度が限られている

        #### SQSの最適化問題としての定式化

        SQSは、以下の評価関数 $J$ を最小化する配置として定義されます：

        $$J(\\text{config}) = \\sum_{n=1}^{N_{\\text{shell}}} w_n \\alpha_n^2(\\text{config})$$

        ここで：
        - $w_n$: 第 $n$ 近接殻の重み（通常、近い殻ほど大きい）
        - $\\alpha_n(\\text{config})$: 配置 config における第 $n$ 殻のSROパラメータ
        - $N_{\\text{shell}}$: 考慮する近接殻の数

        **最適なSQS**は：

        $$\\text{SQS} = \\arg\\min_{\\text{config}} J(\\text{config})$$

        この定式化により、SQSは「Warren-Cowley SROパラメータ空間における原点 $(0,0,\\ldots,0)$ に
        最も近い配置」として幾何学的に理解できます。

        #### 化学SRO vs 構造SRO：本質的な違い

        本アプリでは、**2種類の短範囲秩序**を区別します：

        **1. 化学的短範囲秩序（Chemical SRO）**

        - **定義**: 固定された格子上での化学種（A vs B）の配置相関
        - **パラメータ**: Warren-Cowley $\\alpha_k$
        - **測定対象**: 占有変数 $\\sigma_i \\in \\{0, 1\\}$ の相関
        - **物理的意味**: 「どの原子がどこにいるか」の秩序
        - **数式**: 
          $$\\alpha_k = 1 - \\frac{P_k(B|A)}{c_B}$$
        - **応用**: 合金の相分離、規則-不規則転移、SQS設計

        **2. 構造的短範囲秩序（Structural SRO）**

        - **定義**: 位置的無秩序による幾何学的記述子の相関
        - **パラメータ**: 構造相関係数 $\\beta_k$
        - **測定対象**: 幾何学的記述子 $q_i$ （例: 平均結合長）の相関
        - **物理的意味**: 「原子の局所環境の類似性」の秩序
        - **数式**: 
          $$\\beta_k = \\frac{\\langle (q_i - \\langle q \\rangle)(q_j - \\langle q \\rangle) \\rangle_{\\text{pairs}}}{\\text{Var}(q)}$$
        - **応用**: アモルファス材料、液体、高エントロピー合金の構造解析

        **重要な区別**:
        - 化学SROは**離散的**（A or B）、構造SROは**連続的**（距離、角度など）
        - 化学SROは理想格子で定義可能、構造SROは位置ゆらぎが必要
        - 両者は独立に存在可能（例: 化学的にランダムだが構造的に相関がある）

        #### 本アプリとSQSの関係

        本アプリで計算するWarren-Cowley SROパラメータ $\\alpha_k$ は、SQSの品質を評価する
        **直接的かつ定量的な指標**として機能します：

        **1. アンサンブル平均アプローチ（本アプリ）**

        - 複数のランダム配置を生成: $\\{\\text{config}_1, \\text{config}_2, \\ldots, \\text{config}_M\\}$
        - 各配置で $\\alpha_k$ を計算
        - アンサンブル平均: $\\langle \\alpha_k \\rangle_{\\text{ensemble}} \\approx 0$
        - **利点**: 統計的信頼性、実装が容易
        - **欠点**: 多数の配置が必要、各配置は最適ではない

        **2. 単一最適配置アプローチ（SQS）**

        - 最適化により単一配置を生成: $\\text{config}_{\\text{SQS}}$
        - その配置で $\\alpha_k(\\text{SQS}) \\approx 0$ （すべての $k$ で）
        - **利点**: 単一配置で完全ランダムを近似、DFT計算に最適
        - **欠点**: 最適化が必要、実装が複雑

        **相補的な関係**:
        - 本アプリ: $\\alpha_k$ の統計的振る舞いを理解 → SQSの理論的基礎
        - SQSツール（ATAT等）: 最適配置を生成 → 実用的な材料設計

        #### k-NN拡張の意義

        従来のWarren-Cowley SROは**近接殻ベース**（第1殻、第2殻...）でしたが、
        本アプリでは**k-NN（k nearest neighbors）ベース**に拡張しました：

        **k-NN拡張の利点**:
        1. **位置ゆらぎへの頑健性**: 殻の境界が曖昧な場合でも定義可能
        2. **マルチスケール解析**: $k=1,2,3,4,5$ で異なる長さスケールの秩序を捉える
        3. **機械学習との親和性**: k-NN記述子はGNNの自然な入力
        4. **構造SROとの統一**: 同じk-NN枠組みで化学・構造SROを計算可能

        **数式**:
        $$\\alpha_k = 1 - \\frac{1}{N_A c_B} \\sum_{i \\in A} \\frac{\\#\\{j \\in \\text{kNN}(i) : j \\in B\\}}{k}$$

        ここで、$\\text{kNN}(i)$ は原子 $i$ の $k$ 個の最近接原子の集合です。

        #### SQS生成アルゴリズムの概要

        実用的なSQS生成には、以下のアルゴリズムが使用されます：

        **1. Monte Carlo法（ATAT/mcsqs）**
        - ランダムな原子交換を繰り返し、$J$ を最小化
        - Metropolis基準で配置を受理/棄却
        - 最も広く使用されている手法

        **2. 遺伝的アルゴリズム**
        - 配置を「遺伝子」として扱い、交叉・突然変異で進化
        - 並列化が容易

        **3. クラスター展開法**
        - 配置エネルギーをクラスター相互作用で展開
        - 高精度だが計算コストが高い

        #### 実用的応用例

        **1. 第一原理計算（DFT）**
        - ランダム合金の電子状態計算
        - 格子定数、弾性定数の予測
        - 例: Al-Cu, Ni-Fe合金

        **2. 高エントロピー合金（HEA）**
        - 多成分系（5元素以上）のモデル化
        - 相安定性の評価
        - 例: CoCrFeNiMn（Cantor合金）

        **3. 材料設計**
        - 新規合金の探索
        - 物性予測（硬度、耐食性など）
        - 機械学習との組み合わせ

        #### 本アプリでのSQS概念の確認方法

        **手順**:
        1. **Interactive Mode**で濃度0.5、FCC、サイズ2×2×2を設定
        2. Random Seedを変更しながら、複数の配置で $\\alpha_1$ を計算
        3. **Sweep Mode**で濃度全域（0.05〜0.95）をスキャン
        4. $\\alpha$ の平均値が0付近に収束することを確認
        5. **k-NN SRO**で $\\alpha_1, \\alpha_2, \\ldots, \\alpha_5$ を観察
        6. 位置ゆらぎを追加して**構造SRO** $\\beta_k$ を計算

        **期待される結果**:
        - ランダム配置のアンサンブル平均: $\\langle \\alpha_k \\rangle \\approx 0$
        - 個々の配置: $\\alpha_k \\neq 0$ （統計的ゆらぎ）
        - 標準偏差: $\\sigma(\\alpha_k) \\sim 1/\\sqrt{N}$ （$N$: 原子数）

        #### 既存SQSツールとの比較

        | ツール | 目的 | 手法 | 本アプリとの関係 |
        |--------|------|------|------------------|
        | **ATAT/mcsqs** | SQS生成 | Monte Carlo | 本アプリで理論理解→ATATで実用生成 |
        | **SOD** | 秩序度計算 | 統計解析 | 本アプリと類似（教育的実装） |
        | **icet** | クラスター展開 | 機械学習 | 本アプリのk-NN SROが入力特徴量 |
        | **sqsgenerator** | Python実装 | 遺伝的 | 本アプリで評価関数を理解 |

        #### まとめ：無秩序指標の設計指針

        **化学SRO（$\\alpha_k$）の使い方**:
        - 合金の化学的無秩序度の定量化
        - SQS配置の品質評価
        - 相分離・規則化の検出

        **構造SRO（$\\beta_k$）の使い方**:
        - アモルファス・液体の構造相関
        - 位置的無秩序の定量化
        - 高エントロピー合金の局所環境解析

        **両者の統合**:
        - 化学的にランダムだが構造的に相関がある系（例: 液体合金）
        - 化学的に秩序化しているが構造的に無秩序な系（例: 規則合金の格子欠陥）
        - k-NN枠組みで統一的に扱える

        ### 5.7 実装上の注意点

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
