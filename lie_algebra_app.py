import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sympy as sp
from sympy import symbols, Matrix, simplify, latex, sqrt, cos, sin, pi, exp, I
import pandas as pd
from scipy.linalg import expm
import seaborn as sns

st.set_page_config(
    page_title="Lie代数学習アプリ",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Lie代数学習アプリケーション")
st.markdown("""
このアプリケーションは学部レベルでのLie代数の理解を深めるための教育ツールです。
古典的・低次元Lie代数の表現論に焦点を当て、計算ツール、可視化、演習問題を提供します。
""")

class LieAlgebraCalculator:
    """Lie代数の基本計算を行うクラス"""
    
    @staticmethod
    def commutator(A, B):
        """交換子 [A,B] = AB - BA を計算"""
        return A * B - B * A
    
    @staticmethod
    def adjoint_representation(X, basis):
        """随伴表現 ad(X) を計算"""
        n = len(basis)
        ad_matrix = sp.zeros(n, n)
        
        for i in range(n):
            ad_X_ei = LieAlgebraCalculator.commutator(X, basis[i])
            
            for j in range(n):
                coeff = 0
                for k in range(n):
                    if ad_X_ei.has(basis[k]):
                        coeff += ad_X_ei.coeff(basis[k], 1) if j == k else 0
                ad_matrix[i, j] = coeff
        
        return ad_matrix
    
    @staticmethod
    def structure_constants(basis):
        """構造定数を計算"""
        n = len(basis)
        structure_constants = {}
        
        for i in range(n):
            for j in range(n):
                comm = LieAlgebraCalculator.commutator(basis[i], basis[j])
                structure_constants[(i, j)] = []
                
                for k in range(n):
                    if comm.has(basis[k]):
                        coeff = comm.coeff(basis[k], 1)
                        structure_constants[(i, j)].append(float(coeff))
                    else:
                        structure_constants[(i, j)].append(0.0)
        
        return structure_constants

class ClassicalLieAlgebras:
    """古典的Lie代数の定義と性質"""
    
    @staticmethod
    def sl2_basis():
        """sl(2,R)の標準基底"""
        h = sp.Matrix([[1, 0], [0, -1]])
        e = sp.Matrix([[0, 1], [0, 0]])
        f = sp.Matrix([[0, 0], [1, 0]])
        return {'h': h, 'e': e, 'f': f}
    
    @staticmethod
    def so3_basis():
        """so(3)の標準基底（反対称行列）"""
        L1 = sp.Matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        L2 = sp.Matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        L3 = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        return {'L1': L1, 'L2': L2, 'L3': L3}
    
    @staticmethod
    def su2_basis():
        """su(2)のパウリ行列基底"""
        sigma1 = sp.Matrix([[0, 1], [1, 0]])
        sigma2 = sp.Matrix([[0, -I], [I, 0]])
        sigma3 = sp.Matrix([[1, 0], [0, -1]])
        return {'σ1': sigma1, 'σ2': sigma2, 'σ3': sigma3}

def main():
    st.sidebar.title("📚 学習メニュー")
    
    menu_options = [
        "🏠 ホーム",
        "📖 基本概念",
        "🔢 計算ツール", 
        "📊 可視化",
        "🎯 演習問題",
        "📈 表現論"
    ]
    
    selected_menu = st.sidebar.selectbox("学習内容を選択", menu_options)
    
    if selected_menu == "🏠 ホーム":
        show_home()
    elif selected_menu == "📖 基本概念":
        show_basic_concepts()
    elif selected_menu == "🔢 計算ツール":
        show_calculation_tools()
    elif selected_menu == "📊 可視化":
        show_visualization()
    elif selected_menu == "🎯 演習問題":
        show_exercises()
    elif selected_menu == "📈 表現論":
        show_representation_theory()

def show_home():
    st.header("🏠 Lie代数学習アプリへようこそ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 学習目標")
        st.markdown("""
        - Lie代数の基本概念の理解
        - 古典的Lie代数（sl(2), so(3), su(2)）の性質
        - 交換子と構造定数の計算
        - 表現論の基礎
        - 随伴表現の理解
        """)
        
    with col2:
        st.subheader("🛠️ 機能")
        st.markdown("""
        - **計算ツール**: 交換子、構造定数の自動計算
        - **可視化**: Lie群の軌道と表現の視覚化
        - **演習問題**: インタラクティブな問題演習
        - **表現論**: 既約表現の分析
        """)
    
    st.subheader("📚 推奨学習順序")
    st.markdown("""
    1. **基本概念** - Lie代数の定義と基本性質
    2. **計算ツール** - 具体的な計算に慣れる
    3. **可視化** - 幾何学的直観を養う
    4. **表現論** - より高度な理論の理解
    5. **演習問題** - 知識の定着と応用
    """)

def show_basic_concepts():
    st.header("📖 Lie代数の基本概念")
    
    concept_tabs = st.tabs(["定義", "古典的Lie代数", "交換子", "構造定数"])
    
    with concept_tabs[0]:
        st.subheader("Lie代数の定義")
        st.markdown("""
        **Lie代数**は、双線形演算（Lie括弧）を持つベクトル空間です。
        
        体 $K$ 上のベクトル空間 $\\mathfrak{g}$ がLie代数であるとは、
        双線形写像 $[\\cdot, \\cdot]: \\mathfrak{g} \\times \\mathfrak{g} \\to \\mathfrak{g}$ が存在して、
        以下の性質を満たすことです：
        
        1. **反対称性**: $[x, y] = -[y, x]$
        2. **Jacobi恒等式**: $[x, [y, z]] + [y, [z, x]] + [z, [x, y]] = 0$
        """)
        
        st.latex(r"""
        \text{Jacobi恒等式: } [[x, y], z] + [[y, z], x] + [[z, x], y] = 0
        """)
        
    with concept_tabs[1]:
        st.subheader("古典的Lie代数")
        
        algebra_type = st.selectbox("Lie代数を選択", ["sl(2,R)", "so(3)", "su(2)"])
        
        if algebra_type == "sl(2,R)":
            st.markdown("""
            **sl(2,R)**: トレースが0の2×2実行列のLie代数
            """)
            basis = ClassicalLieAlgebras.sl2_basis()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.latex("h = " + latex(basis['h']))
            with col2:
                st.latex("e = " + latex(basis['e']))
            with col3:
                st.latex("f = " + latex(basis['f']))
                
            st.markdown("**交換関係:**")
            st.latex(r"[h, e] = 2e, \quad [h, f] = -2f, \quad [e, f] = h")
            
        elif algebra_type == "so(3)":
            st.markdown("""
            **so(3)**: 3×3反対称行列のLie代数（3次元回転群の無限小生成元）
            """)
            basis = ClassicalLieAlgebras.so3_basis()
            
            for name, matrix in basis.items():
                st.latex(f"{name} = " + latex(matrix))
                
            st.markdown("**交換関係:**")
            st.latex(r"[L_1, L_2] = L_3, \quad [L_2, L_3] = L_1, \quad [L_3, L_1] = L_2")
            
        elif algebra_type == "su(2)":
            st.markdown("""
            **su(2)**: トレースが0の2×2エルミート行列のLie代数
            """)
            basis = ClassicalLieAlgebras.su2_basis()
            
            for name, matrix in basis.items():
                st.latex(f"{name} = " + latex(matrix))
                
            st.markdown("**交換関係:**")
            st.latex(r"[\sigma_1, \sigma_2] = 2i\sigma_3, \quad [\sigma_2, \sigma_3] = 2i\sigma_1, \quad [\sigma_3, \sigma_1] = 2i\sigma_2")
    
    with concept_tabs[2]:
        st.subheader("交換子（Lie括弧）")
        st.markdown("""
        行列Lie代数では、交換子は通常の行列の交換子として定義されます：
        """)
        st.latex(r"[A, B] = AB - BA")
        
        st.markdown("**例**: sl(2,R)での計算")
        basis = ClassicalLieAlgebras.sl2_basis()
        h, e, f = basis['h'], basis['e'], basis['f']
        
        comm_he = LieAlgebraCalculator.commutator(h, e)
        comm_hf = LieAlgebraCalculator.commutator(h, f)
        comm_ef = LieAlgebraCalculator.commutator(e, f)
        
        st.latex(f"[h, e] = {latex(comm_he)} = 2e")
        st.latex(f"[h, f] = {latex(comm_hf)} = -2f")
        st.latex(f"[e, f] = {latex(comm_ef)} = h")
    
    with concept_tabs[3]:
        st.subheader("構造定数")
        st.markdown("""
        Lie代数の基底 $\\{e_1, e_2, \\ldots, e_n\\}$ に対して、構造定数 $C_{ij}^k$ は：
        """)
        st.latex(r"[e_i, e_j] = \sum_{k=1}^n C_{ij}^k e_k")
        
        st.markdown("構造定数はLie代数の構造を完全に決定します。")

def show_calculation_tools():
    st.header("🔢 計算ツール")
    
    tool_tabs = st.tabs(["交換子計算", "構造定数", "随伴表現"])
    
    with tool_tabs[0]:
        st.subheader("交換子計算器")
        
        algebra_choice = st.selectbox("Lie代数を選択", ["sl(2,R)", "so(3)", "su(2)"], key="comm_calc")
        
        if algebra_choice == "sl(2,R)":
            basis = ClassicalLieAlgebras.sl2_basis()
            basis_names = list(basis.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                elem1 = st.selectbox("第1要素", basis_names, key="elem1_sl2")
            with col2:
                elem2 = st.selectbox("第2要素", basis_names, key="elem2_sl2")
            
            if st.button("交換子を計算", key="calc_comm_sl2"):
                A = basis[elem1]
                B = basis[elem2]
                comm = LieAlgebraCalculator.commutator(A, B)
                
                st.latex(f"[{elem1}, {elem2}] = {latex(comm)}")
                
                coeffs = []
                for name, base_elem in basis.items():
                    if comm.has(base_elem):
                        coeff = comm.coeff(base_elem, 1)
                        if coeff != 0:
                            coeffs.append(f"{coeff}{name}")
                
                if coeffs:
                    result_str = " + ".join(coeffs).replace("+ -", "- ")
                    st.markdown(f"**基底表現**: {result_str}")
                else:
                    st.markdown("**結果**: 0")
    
    with tool_tabs[1]:
        st.subheader("構造定数表")
        
        algebra_choice = st.selectbox("Lie代数を選択", ["sl(2,R)", "so(3)"], key="struct_const")
        
        if algebra_choice == "sl(2,R)":
            basis = ClassicalLieAlgebras.sl2_basis()
            basis_list = list(basis.values())
            basis_names = list(basis.keys())
            
            if st.button("構造定数を計算", key="calc_struct_sl2"):
                struct_const = LieAlgebraCalculator.structure_constants(basis_list)
                
                n = len(basis_names)
                for k in range(n):
                    st.markdown(f"**{basis_names[k]} 成分:**")
                    
                    data = []
                    for i in range(n):
                        row = []
                        for j in range(n):
                            coeff = struct_const[(i, j)][k]
                            row.append(f"{coeff:.1f}")
                        data.append(row)
                    
                    df = pd.DataFrame(data, 
                                    columns=[f"[·,{name}]" for name in basis_names],
                                    index=[f"[{name},·]" for name in basis_names])
                    st.dataframe(df)
    
    with tool_tabs[2]:
        st.subheader("随伴表現")
        st.markdown("""
        随伴表現 $\\text{ad}: \\mathfrak{g} \\to \\text{End}(\\mathfrak{g})$ は：
        """)
        st.latex(r"\text{ad}(x)(y) = [x, y]")
        
        algebra_choice = st.selectbox("Lie代数を選択", ["sl(2,R)"], key="adjoint")
        
        if algebra_choice == "sl(2,R)":
            basis = ClassicalLieAlgebras.sl2_basis()
            basis_names = list(basis.keys())
            
            element = st.selectbox("要素を選択", basis_names, key="adj_elem")
            
            if st.button("随伴表現を計算", key="calc_adj"):
                X = basis[element]
                basis_list = list(basis.values())
                
                ad_matrix = LieAlgebraCalculator.adjoint_representation(X, basis_list)
                
                st.latex(f"\\text{{ad}}({element}) = {latex(ad_matrix)}")
                
                ad_numeric = np.array(ad_matrix).astype(float)
                df = pd.DataFrame(ad_numeric, 
                                columns=basis_names, 
                                index=basis_names)
                st.dataframe(df)

def show_visualization():
    st.header("📊 可視化")
    
    viz_tabs = st.tabs(["群軌道", "表現空間", "ルート系"])
    
    with viz_tabs[0]:
        st.subheader("SO(3)の群軌道可視化")
        
        col1, col2 = st.columns(2)
        with col1:
            theta = st.slider("回転角θ (度)", 0, 360, 45)
            phi = st.slider("回転軸φ (度)", 0, 180, 90)
        with col2:
            psi = st.slider("回転軸ψ (度)", 0, 360, 0)
            num_points = st.slider("軌道点数", 10, 100, 50)
        
        if st.button("軌道を可視化"):
            t_vals = np.linspace(0, 2*np.pi, num_points)
            
            initial_point = np.array([0, 0, 1])
            
            axis = np.array([
                np.sin(np.radians(phi)) * np.cos(np.radians(psi)),
                np.sin(np.radians(phi)) * np.sin(np.radians(psi)),
                np.cos(np.radians(phi))
            ])
            
            orbit_points = []
            for t in t_vals:
                cos_t = np.cos(t * np.radians(theta) / 360)
                sin_t = np.sin(t * np.radians(theta) / 360)
                
                rotated = (initial_point * cos_t + 
                          np.cross(axis, initial_point) * sin_t +
                          axis * np.dot(axis, initial_point) * (1 - cos_t))
                orbit_points.append(rotated)
            
            orbit_points = np.array(orbit_points)
            
            fig = go.Figure()
            
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
            
            fig.add_trace(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.3, colorscale='Blues',
                showscale=False, name="単位球面"
            ))
            
            fig.add_trace(go.Scatter3d(
                x=orbit_points[:, 0],
                y=orbit_points[:, 1], 
                z=orbit_points[:, 2],
                mode='markers+lines',
                marker=dict(size=5, color='red'),
                line=dict(color='red', width=3),
                name="群軌道"
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[initial_point[0]],
                y=[initial_point[1]],
                z=[initial_point[2]],
                mode='markers',
                marker=dict(size=10, color='green'),
                name="初期点"
            ))
            
            fig.update_layout(
                title="SO(3)による単位球面上の点の軌道",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y", 
                    zaxis_title="Z",
                    aspectmode='cube'
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        st.subheader("SU(2)表現の可視化")
        
        spin = st.selectbox("スピン", ["1/2", "1", "3/2"])
        
        if spin == "1/2":
            st.markdown("**スピン1/2表現** (基本表現)")
            
            basis = ClassicalLieAlgebras.su2_basis()
            
            pauli_choice = st.selectbox("パウリ行列", ["σ1", "σ2", "σ3"])
            
            if st.button("固有値・固有ベクトルを可視化"):
                matrix = basis[pauli_choice]
                matrix_numeric = np.array(matrix).astype(complex)
                
                eigenvals, eigenvecs = np.linalg.eig(matrix_numeric)
                
                st.markdown(f"**{pauli_choice}の固有値**: {eigenvals}")
                
                fig = go.Figure()
                
                u = np.linspace(0, 2 * np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                x = np.outer(np.cos(u), np.sin(v))
                y = np.outer(np.sin(u), np.sin(v))
                z = np.outer(np.ones(np.size(u)), np.cos(v))
                
                fig.add_trace(go.Surface(
                    x=x, y=y, z=z,
                    opacity=0.3, colorscale='Blues',
                    showscale=False
                ))
                
                for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
                    bloch_vec = np.array([
                        2 * np.real(vec[0] * np.conj(vec[1])),
                        2 * np.imag(vec[0] * np.conj(vec[1])),
                        np.abs(vec[0])**2 - np.abs(vec[1])**2
                    ])
                    
                    fig.add_trace(go.Scatter3d(
                        x=[0, bloch_vec[0]],
                        y=[0, bloch_vec[1]],
                        z=[0, bloch_vec[2]],
                        mode='lines+markers',
                        marker=dict(size=[5, 8]),
                        name=f"固有値 {val:.2f}"
                    ))
                
                fig.update_layout(
                    title=f"{pauli_choice}の固有ベクトル (ブロッホ球面)",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        aspectmode='cube'
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[2]:
        st.subheader("A1型ルート系 (sl(2))")
        
        st.markdown("""
        sl(2)のルート系は最も単純なA1型です。
        カルタン部分代数の双対空間において、ルートは ±α の2つです。
        """)
        
        if st.button("ルート系を可視化"):
            fig, ax = plt.subplots(figsize=(8, 6))
            
            roots = np.array([[1], [-1]])
            
            for i, root in enumerate(roots):
                ax.arrow(0, 0, root[0], 0, 
                        head_width=0.1, head_length=0.1,
                        fc='red' if i == 0 else 'blue',
                        ec='red' if i == 0 else 'blue',
                        linewidth=2)
                
                ax.text(root[0] + 0.1, 0.1, 
                       f"{'α' if i == 0 else '-α'}",
                       fontsize=14, fontweight='bold')
            
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-0.5, 0.5)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_title("A1型ルート系 (sl(2))", fontsize=16, fontweight='bold')
            ax.set_xlabel("カルタン部分代数の双対", fontsize=12)
            
            st.pyplot(fig)
            
            st.markdown("""
            **ルート系の性質:**
            - 正ルート: α
            - 負ルート: -α  
            - 単純ルート: α
            - ワイル群: {1, s_α} ≅ Z/2Z
            """)

def show_exercises():
    st.header("🎯 演習問題")
    
    exercise_tabs = st.tabs(["基本計算", "構造定数", "表現論"])
    
    with exercise_tabs[0]:
        st.subheader("基本計算演習")
        
        problem_type = st.selectbox("問題タイプ", ["交換子計算", "Jacobi恒等式"])
        
        if problem_type == "交換子計算":
            st.markdown("**問題**: sl(2,R)において、以下の交換子を計算してください。")
            
            if 'exercise_problem' not in st.session_state:
                st.session_state.exercise_problem = None
            
            if st.button("新しい問題を生成"):
                basis_names = ['h', 'e', 'f']
                elem1, elem2 = np.random.choice(basis_names, 2, replace=False)
                st.session_state.exercise_problem = (elem1, elem2)
            
            if st.session_state.exercise_problem:
                elem1, elem2 = st.session_state.exercise_problem
                st.latex(f"[{elem1}, {elem2}] = ?")
                
                options = ["0", "h", "e", "f", "2e", "-2f", "2h", "-h"]
                user_answer = st.selectbox("答えを選択", options)
                
                if st.button("答えを確認"):
                    basis = ClassicalLieAlgebras.sl2_basis()
                    A = basis[elem1]
                    B = basis[elem2]
                    comm = LieAlgebraCalculator.commutator(A, B)
                    
                    correct_answers = {
                        ('h', 'e'): '2e',
                        ('e', 'h'): '-2e', 
                        ('h', 'f'): '-2f',
                        ('f', 'h'): '2f',
                        ('e', 'f'): 'h',
                        ('f', 'e'): '-h'
                    }
                    
                    correct = correct_answers.get((elem1, elem2), "0")
                    
                    if user_answer == correct:
                        st.success("✅ 正解です！")
                    else:
                        st.error(f"❌ 不正解です。正解は {correct} です。")
                    
                    st.latex(f"[{elem1}, {elem2}] = {latex(comm)}")
        
        elif problem_type == "Jacobi恒等式":
            st.markdown("**問題**: Jacobi恒等式を確認してください。")
            st.latex(r"[[x, y], z] + [[y, z], x] + [[z, x], y] = 0")
            
            if st.button("sl(2,R)で確認"):
                basis = ClassicalLieAlgebras.sl2_basis()
                h, e, f = basis['h'], basis['e'], basis['f']
                
                term1 = LieAlgebraCalculator.commutator(
                    LieAlgebraCalculator.commutator(h, e), f)
                term2 = LieAlgebraCalculator.commutator(
                    LieAlgebraCalculator.commutator(e, f), h)
                term3 = LieAlgebraCalculator.commutator(
                    LieAlgebraCalculator.commutator(f, h), e)
                
                total = term1 + term2 + term3
                
                st.latex(f"[[h, e], f] = {latex(term1)}")
                st.latex(f"[[e, f], h] = {latex(term2)}")
                st.latex(f"[[f, h], e] = {latex(term3)}")
                st.latex(f"\\text{{合計}} = {latex(total)}")
                
                if total == sp.zeros(2, 2):
                    st.success("✅ Jacobi恒等式が成立しています！")
                else:
                    st.error("❌ 計算エラーがあります。")
    
    with exercise_tabs[1]:
        st.subheader("構造定数演習")
        
        st.markdown("**問題**: so(3)の構造定数を求めてください。")
        st.markdown("交換関係: $[L_i, L_j] = \\epsilon_{ijk} L_k$")
        
        if st.button("解答を表示", key="exercise_answer_so3"):
            st.markdown("**解答**:")
            
            epsilon = np.zeros((3, 3, 3))
            epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
            epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
            
            st.markdown("構造定数 $C_{ij}^k = \\epsilon_{ijk}$:")
            
            for k in range(3):
                st.markdown(f"**$L_{k+1}$ 成分:**")
                data = []
                for i in range(3):
                    row = []
                    for j in range(3):
                        row.append(f"{epsilon[i, j, k]:.0f}")
                    data.append(row)
                
                df = pd.DataFrame(data,
                                columns=[f"[·,L{j+1}]" for j in range(3)],
                                index=[f"[L{i+1},·]" for i in range(3)])
                st.dataframe(df)
    
    with exercise_tabs[2]:
        st.subheader("表現論演習")
        
        st.markdown("**問題**: SU(2)の基本表現における重み空間を求めてください。")
        
        if st.button("解答を表示", key="exercise_answer_su2"):
            st.markdown("""
            **解答**:
            
            SU(2)の基本表現（スピン1/2）では：
            - 最高重み: $\\lambda = 1/2$
            - 重み: $\\{1/2, -1/2\\}$
            - 重み空間の次元: 各1次元
            
            重みベクトル:
            - $|1/2\\rangle$: 重み $1/2$
            - $|-1/2\\rangle$: 重み $-1/2$
            """)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            weights = [0.5, -0.5]
            colors = ['red', 'blue']
            
            for i, (weight, color) in enumerate(zip(weights, colors)):
                ax.scatter(weight, 0, s=200, c=color, alpha=0.7)
                ax.text(weight, 0.1, f"|{weight}⟩", 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.3, 0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax.set_title("SU(2)基本表現の重み図", fontsize=14, fontweight='bold')
            ax.set_xlabel("重み", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

def show_representation_theory():
    st.header("📈 表現論")
    
    rep_tabs = st.tabs(["基本概念", "既約表現", "重み空間", "テンソル積"])
    
    with rep_tabs[0]:
        st.subheader("表現論の基本概念")
        
        st.markdown("""
        **Lie代数の表現**とは、Lie代数の準同型写像 $\\rho: \\mathfrak{g} \\to \\text{End}(V)$ です。
        
        **重要な概念:**
        - **既約表現**: 非自明な不変部分空間を持たない表現
        - **重み**: カルタン部分代数の固有値
        - **最高重み**: 表現を特徴づける最大の重み
        - **重み空間**: 各重みに対応する固有空間
        """)
        
        st.markdown("**例: sl(2,R)の表現**")
        st.latex(r"""
        \rho(h) = \begin{pmatrix} \lambda & 0 \\ 0 & -\lambda \end{pmatrix}, \quad
        \rho(e) = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad  
        \rho(f) = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}
        """)
    
    with rep_tabs[1]:
        st.subheader("SU(2)の既約表現")
        
        spin_value = st.selectbox("スピン値", ["1/2", "1", "3/2", "2"])
        
        if st.button("表現を分析", key="analyze_representation"):
            j = float(spin_value)
            dim = int(2*j + 1)
            
            st.markdown(f"**スピン {spin_value} 表現**")
            st.markdown(f"- 次元: {dim}")
            st.markdown(f"- 重み: {{{', '.join([str(j-k) for k in range(dim)])}}}")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            weights = [j - k for k in range(dim)]
            colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, dim))
            
            for i, (weight, color) in enumerate(zip(weights, colors)):
                ax.scatter(weight, 0, s=200, c=[color], alpha=0.8)
                ax.text(weight, 0.1, f"|{weight}⟩", 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xlim(min(weights)-0.5, max(weights)+0.5)
            ax.set_ylim(-0.3, 0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax.set_title(f"スピン {spin_value} 表現の重み図", fontsize=14, fontweight='bold')
            ax.set_xlabel("重み", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            st.markdown("**昇降演算子の作用:**")
            for k in range(dim):
                m = j - k
                if k < dim - 1:
                    next_m = j - (k + 1)
                    coeff = np.sqrt(j*(j+1) - m*(m-1))
                    st.latex(f"J_- |{m}\\rangle = {coeff:.2f} |{next_m}\\rangle")
                if k > 0:
                    prev_m = j - (k - 1)
                    coeff = np.sqrt(j*(j+1) - m*(m+1))
                    st.latex(f"J_+ |{m}\\rangle = {coeff:.2f} |{prev_m}\\rangle")
    
    with rep_tabs[2]:
        st.subheader("重み空間の分解")
        
        st.markdown("""
        カルタン部分代数 $\\mathfrak{h}$ に対して、表現空間は重み空間に分解されます：
        """)
        st.latex(r"V = \bigoplus_{\lambda \in \mathfrak{h}^*} V_\lambda")
        st.latex(r"V_\lambda = \{v \in V : h \cdot v = \lambda(h) v \text{ for all } h \in \mathfrak{h}\}")
        
        if st.button("sl(2)での例を表示", key="show_sl2_example"):
            st.markdown("**例: sl(2)のスピン1表現**")
            
            weights_data = {
                "重み": [1, 0, -1],
                "重み空間": ["|1⟩", "|0⟩", "|-1⟩"],
                "次元": [1, 1, 1],
                "h の固有値": [1, 0, -1]
            }
            
            df = pd.DataFrame(weights_data)
            st.dataframe(df)
            
            st.markdown("**重み空間の基底:**")
            st.latex(r"""
            V_1 = \text{span}\{|1\rangle\}, \quad
            V_0 = \text{span}\{|0\rangle\}, \quad  
            V_{-1} = \text{span}\{|-1\rangle\}
            """)
    
    with rep_tabs[3]:
        st.subheader("テンソル積表現")
        
        st.markdown("""
        2つの表現のテンソル積は新しい表現を与えます。
        SU(2)では、Clebsch-Gordan係数により既約分解されます。
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            j1 = st.selectbox("第1表現のスピン", ["1/2", "1", "3/2"], key="j1")
        with col2:
            j2 = st.selectbox("第2表現のスピン", ["1/2", "1", "3/2"], key="j2")
        
        if st.button("テンソル積を分解", key="decompose_tensor_product"):
            j1_val = float(j1)
            j2_val = float(j2)
            
            j_min = abs(j1_val - j2_val)
            j_max = j1_val + j2_val
            
            j_values = []
            j = j_min
            while j <= j_max:
                j_values.append(j)
                j += 1
            
            st.markdown(f"**{j1} ⊗ {j2} の既約分解:**")
            
            decomposition = " ⊕ ".join([f"{j}" for j in j_values])
            st.latex(f"{j1} \\otimes {j2} = {decomposition}")
            
            dim1 = int(2*j1_val + 1)
            dim2 = int(2*j2_val + 1)
            total_dim = dim1 * dim2
            
            decomp_dim = sum(int(2*j + 1) for j in j_values)
            
            st.markdown("**次元の確認:**")
            st.markdown(f"- 左辺: {dim1} × {dim2} = {total_dim}")
            st.markdown(f"- 右辺: {' + '.join([str(int(2*j + 1)) for j in j_values])} = {decomp_dim}")
            
            if total_dim == decomp_dim:
                st.success("✅ 次元が一致しています！")
            else:
                st.error("❌ 次元が一致しません。")

if __name__ == "__main__":
    main()
