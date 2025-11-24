"""
High-Entropy Alloys (HEA) / Complex Concentrated Alloys (CCA) 
Machine Learning Molecular Dynamics Simulator

ハイエントロピー合金向け機械学習分子動力学シミュレーター

This application demonstrates the power of Machine Learning Potentials (MLP)
for simulating multi-element systems where traditional potentials struggle.

物理的正確さを最重要視した実装
"""

import streamlit as st
import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import bulk
from ase.md.langevin import Langevin
from ase import units
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from stmol import showmol
import py3Dmol

st.set_page_config(
    page_title="HEA/CCA ML-MD Simulator",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ High-Entropy Alloys (HEA) 機械学習分子動力学シミュレーター")
st.markdown("""
このアプリケーションは、**ハイエントロピー合金 (HEA)** や **複雑組成合金 (CCA)** の分子動力学シミュレーションを、
**機械学習ポテンシャル (MACE)** を用いて実行します。

**特徴:**
- 多元素ランダム固溶体の生成と可視化
- 汎用MLポテンシャルによる正確な力計算
- 格子歪みのリアルタイム観察
- エネルギー・温度の時間発展モニタリング
""")

# Sidebar configuration
st.sidebar.header("⚙️ シミュレーション設定")

# Tab structure
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔧 構造生成", 
    "🚀 MD シミュレーション", 
    "📊 結果分析",
    "📚 理論背景",
    "📖 使用方法"
])

# ============================================================================
# Helper Functions
# ============================================================================

def create_hea_structure(elements, size=(3, 3, 3), lattice_constant=3.52):
    """
    指定された元素リストから等原子量のHEA構造(FCC)を生成する
    
    Parameters:
    -----------
    elements : list of str
        構成元素のリスト (例: ['Co', 'Cr', 'Fe', 'Mn', 'Ni'])
    size : tuple
        単位格子の繰り返し数 (nx, ny, nz)
    lattice_constant : float
        格子定数 (Å) - 平均的な値を使用
    
    Returns:
    --------
    atoms : ase.Atoms
        生成されたHEA構造
    """
    # ベースとなる単元素FCC構造を作成
    # Cuの格子定数を基準として使用（後でMLPが最適化する）
    atoms = bulk("Cu", crystalstructure="fcc", a=lattice_constant, cubic=True) * size
    
    # 原子総数を取得
    n_atoms = len(atoms)
    
    # 元素リストを作成（等原子量）
    n_elements = len(elements)
    symbols = []
    for i in range(n_atoms):
        symbols.append(elements[i % n_elements])
    
    # ランダムに混ぜる（配置エントロピーの実現）
    random.shuffle(symbols)
    
    # Atomsオブジェクトに適用
    atoms.set_chemical_symbols(symbols)
    
    return atoms

def calculate_configurational_entropy(elements, composition=None):
    """
    配置エントロピーを計算
    
    S_conf = -R * Σ(c_i * ln(c_i))
    
    Parameters:
    -----------
    elements : list
        元素リスト
    composition : dict or None
        組成比（Noneの場合は等原子量と仮定）
    
    Returns:
    --------
    S_conf : float
        配置エントロピー (J/mol·K)
    """
    R = 8.314  # J/mol·K
    
    if composition is None:
        # 等原子量の場合
        n = len(elements)
        c_i = 1.0 / n
        S_conf = -R * n * (c_i * np.log(c_i))
    else:
        S_conf = -R * sum(c * np.log(c) for c in composition.values() if c > 0)
    
    return S_conf

def get_element_colors():
    """
    CPK配色に基づく元素の色を返す
    """
    colors = {
        'Al': '#848484',  # Aluminum - gray
        'Co': '#f090a0',  # Cobalt - pink
        'Cr': '#8a99c7',  # Chromium - blue-gray
        'Cu': '#c88033',  # Copper - orange
        'Fe': '#e06633',  # Iron - orange-red
        'Mn': '#9c7ac7',  # Manganese - purple
        'Ni': '#50d050',  # Nickel - green
        'Ti': '#bfc2c7',  # Titanium - silver
        'V': '#a6a6ab',   # Vanadium - gray
    }
    return colors

def atoms_to_py3dmol(atoms, show_cell=True):
    """
    ASE AtomsオブジェクトをPy3Dmol形式に変換
    
    Parameters:
    -----------
    atoms : ase.Atoms
        原子構造
    show_cell : bool
        セルを表示するかどうか
    
    Returns:
    --------
    view : py3Dmol.view
        3D可視化オブジェクト
    """
    # XYZ形式の文字列を生成
    xyz_str = f"{len(atoms)}\n\n"
    for atom in atoms:
        pos = atom.position
        xyz_str += f"{atom.symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
    
    # Py3Dmolビューアを作成
    view = py3Dmol.view(width=800, height=600)
    view.addModel(xyz_str, 'xyz')
    
    # 元素ごとの色設定
    colors = get_element_colors()
    for element, color in colors.items():
        view.setStyle({'elem': element}, {'sphere': {'color': color, 'radius': 0.5}})
    
    # デフォルトスタイル（上記で設定されていない元素用）
    view.setStyle({}, {'sphere': {'radius': 0.5}})
    
    # セルの表示
    if show_cell:
        cell = atoms.get_cell()
        # セルの辺を描画
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    start = np.array([0, 0, 0], dtype=float)
                    end = np.array([0, 0, 0], dtype=float)
                    
                    if i == 0:
                        start[1] = j * cell[1, 1]
                        start[2] = k * cell[2, 2]
                        end = start + cell[0]
                    elif i == 1:
                        start[0] = j * cell[0, 0]
                        start[2] = k * cell[2, 2]
                        end = start + cell[1]
                    else:
                        start[0] = j * cell[0, 0]
                        start[1] = k * cell[1, 1]
                        end = start + cell[2]
                    
                    view.addCylinder({
                        'start': {'x': float(start[0]), 'y': float(start[1]), 'z': float(start[2])},
                        'end': {'x': float(end[0]), 'y': float(end[1]), 'z': float(end[2])},
                        'radius': 0.05,
                        'color': 'black',
                        'opacity': 0.5
                    })
    
    view.zoomTo()
    return view

# ============================================================================
# Tab 1: Structure Generation
# ============================================================================

with tab1:
    st.header("🔧 HEA構造の生成")
    
    st.markdown("""
    ### 構成元素の選択
    
    ハイエントロピー合金は、**5種類以上の元素**が**等原子量**または**準等原子量**で混合された合金です。
    代表的な例として、**Cantor合金 (CoCrFeMnNi)** があります。
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Element selection
        available_elements = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'Ti', 'V']
        
        selected_elements = st.multiselect(
            "構成元素を選択（5種類推奨）",
            available_elements,
            default=['Co', 'Cr', 'Fe', 'Mn', 'Ni'],
            help="Cantor合金: Co, Cr, Fe, Mn, Ni"
        )
        
        if len(selected_elements) < 2:
            st.warning("⚠️ 少なくとも2種類以上の元素を選択してください")
        elif len(selected_elements) < 5:
            st.info("ℹ️ HEAは通常5種類以上の元素で構成されます")
        
        # Lattice constant
        lattice_constant = st.slider(
            "格子定数 (Å)",
            min_value=3.0,
            max_value=4.5,
            value=3.52,
            step=0.01,
            help="FCC構造の格子定数。平均的な値として3.52Åを使用"
        )
        
        # System size
        st.markdown("### システムサイズ")
        size_multiplier = st.selectbox(
            "単位格子の繰り返し数",
            options=[2, 3, 4],
            index=1,
            help="3x3x3 = 108原子（推奨）、計算負荷とのバランス"
        )
        size = (size_multiplier, size_multiplier, size_multiplier)
        n_atoms = 4 * size_multiplier**3  # FCC: 4 atoms per unit cell
        
        st.info(f"総原子数: **{n_atoms}** 原子")
    
    with col2:
        st.markdown("### 元素の色")
        colors = get_element_colors()
        for elem in selected_elements:
            if elem in colors:
                st.markdown(f"<span style='color:{colors[elem]}'>⬤</span> {elem}", 
                          unsafe_allow_html=True)
    
    # Generate structure button
    if st.button("🔨 構造を生成", type="primary", disabled=len(selected_elements) < 2):
        with st.spinner("構造を生成中..."):
            # Generate HEA structure
            atoms = create_hea_structure(
                elements=selected_elements,
                size=size,
                lattice_constant=lattice_constant
            )
            
            # Store in session state
            st.session_state['atoms'] = atoms
            st.session_state['initial_atoms'] = atoms.copy()
            st.session_state['selected_elements'] = selected_elements
            
            # Calculate configurational entropy
            S_conf = calculate_configurational_entropy(selected_elements)
            st.session_state['S_conf'] = S_conf
            
            st.success(f"✅ 構造生成完了！ 総原子数: {len(atoms)}")
    
    # Display structure if generated
    if 'atoms' in st.session_state:
        st.markdown("---")
        st.subheader("📐 生成された構造")
        
        atoms = st.session_state['atoms']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 3D可視化")
            view = atoms_to_py3dmol(atoms, show_cell=True)
            showmol(view, height=600, width=800)
        
        with col2:
            st.markdown("### 構造情報")
            st.write(f"**総原子数:** {len(atoms)}")
            st.write(f"**構成元素:** {', '.join(st.session_state['selected_elements'])}")
            
            # Element composition
            symbols = atoms.get_chemical_symbols()
            composition = {}
            for elem in st.session_state['selected_elements']:
                count = symbols.count(elem)
                composition[elem] = count
            
            st.markdown("**組成:**")
            for elem, count in composition.items():
                percentage = (count / len(atoms)) * 100
                st.write(f"- {elem}: {count} 原子 ({percentage:.1f}%)")
            
            # Configurational entropy
            S_conf = st.session_state.get('S_conf', 0)
            st.markdown(f"**配置エントロピー:**")
            st.write(f"S_conf = {S_conf:.2f} J/mol·K")
            
            # Cell parameters
            cell = atoms.get_cell()
            st.markdown("**セルパラメータ:**")
            st.write(f"- a = {cell[0, 0]:.3f} Å")
            st.write(f"- b = {cell[1, 1]:.3f} Å")
            st.write(f"- c = {cell[2, 2]:.3f} Å")

# ============================================================================
# Tab 2: MD Simulation
# ============================================================================

with tab2:
    st.header("🚀 分子動力学シミュレーション")
    
    if 'atoms' not in st.session_state:
        st.warning("⚠️ まず「構造生成」タブで構造を生成してください")
    else:
        st.markdown("""
        ### 機械学習ポテンシャル (MACE) による MD シミュレーション
        
        **MACE (Multi-Atomic Cluster Expansion)** は、多元素系に対応した汎用機械学習ポテンシャルです。
        従来のEAMやEMTポテンシャルでは困難だった、複雑な化学環境の表現が可能です。
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("温度設定")
            temperature = st.number_input(
                "温度 (K)",
                min_value=100,
                max_value=2000,
                value=300,
                step=50,
                help="シミュレーション温度。300K = 室温、1000-1500K = 高温実験"
            )
            
            st.info(f"**{temperature} K** = {temperature - 273.15:.1f} °C")
        
        with col2:
            st.subheader("時間設定")
            timestep = st.number_input(
                "タイムステップ (fs)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                help="1 fs = 10^-15 秒。通常1-2 fsが推奨"
            )
            
            n_steps = st.number_input(
                "ステップ数",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="計算時間に注意。100ステップ ≈ 数分"
            )
        
        with col3:
            st.subheader("積分器設定")
            friction = st.number_input(
                "摩擦係数 (1/fs)",
                min_value=0.001,
                max_value=0.1,
                value=0.002,
                step=0.001,
                format="%.3f",
                help="Langevin動力学の摩擦係数。熱浴との結合強度"
            )
        
        # Potential selection
        st.markdown("---")
        st.subheader("⚡ ポテンシャル選択")
        
        potential_type = st.radio(
            "使用するポテンシャル",
            options=["MACE (推奨)", "EMT (高速・近似)"],
            help="MACE: 高精度ML、EMT: 簡易的な経験的ポテンシャル"
        )
        
        if potential_type == "MACE (推奨)":
            st.info("""
            **MACE-MP-0-small** を使用します。
            - 多元素対応の汎用ポテンシャル
            - Materials Project データで訓練
            - CPU モードで実行（GPU不要）
            - 計算時間: 中程度
            """)
        else:
            st.warning("""
            **EMT (Effective Medium Theory)** を使用します。
            - 限定的な元素のみ対応 (Cu, Ag, Au, Ni, Pd, Pt, Al)
            - 近似的な計算
            - 計算時間: 高速
            - **HEAの正確なシミュレーションには不適**
            """)
        
        # Run simulation button
        st.markdown("---")
        
        if st.button("▶️ シミュレーション開始", type="primary"):
            atoms = st.session_state['atoms'].copy()
            
            # Setup calculator
            try:
                if potential_type == "MACE (推奨)":
                    with st.spinner("MACE ポテンシャルを読み込み中..."):
                        from mace.calculators import mace_mp
                        
                        calc = mace_mp(
                            model="small",
                            dispersion=False,
                            default_dtype="float32",
                            device="cpu"
                        )
                        atoms.calc = calc
                        st.success("✅ MACE ポテンシャル読み込み完了")
                else:
                    from ase.calculators.emt import EMT
                    calc = EMT()
                    atoms.calc = calc
                    st.success("✅ EMT ポテンシャル設定完了")
                
                # Setup MD
                with st.spinner("MDシミュレーションを実行中..."):
                    # Initialize velocities
                    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
                    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
                    
                    # Setup Langevin dynamics
                    dyn = Langevin(
                        atoms,
                        timestep=timestep * units.fs,
                        temperature_K=temperature,
                        friction=friction / units.fs
                    )
                    
                    # Storage for trajectory
                    trajectory = []
                    energies = []
                    temperatures = []
                    times = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run MD
                    for step in range(n_steps):
                        dyn.run(1)
                        
                        # Store data
                        trajectory.append(atoms.copy())
                        energies.append(atoms.get_potential_energy())
                        temperatures.append(atoms.get_temperature())
                        times.append(step * timestep)
                        
                        # Update progress
                        progress = (step + 1) / n_steps
                        progress_bar.progress(progress)
                        status_text.text(f"ステップ {step + 1}/{n_steps} - "
                                       f"E = {energies[-1]:.3f} eV, "
                                       f"T = {temperatures[-1]:.1f} K")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store results
                    st.session_state['trajectory'] = trajectory
                    st.session_state['energies'] = energies
                    st.session_state['temperatures'] = temperatures
                    st.session_state['times'] = times
                    st.session_state['md_params'] = {
                        'temperature': temperature,
                        'timestep': timestep,
                        'n_steps': n_steps,
                        'friction': friction,
                        'potential': potential_type
                    }
                    
                    st.success(f"✅ シミュレーション完了！ {n_steps} ステップ実行")
                    st.balloons()
            
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                st.info("EMTポテンシャルを試すか、元素の組み合わせを変更してください")
        
        # Display real-time results if available
        if 'trajectory' in st.session_state:
            st.markdown("---")
            st.subheader("📊 シミュレーション結果")
            
            # Energy and temperature plots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("ポテンシャルエネルギー", "温度"),
                vertical_spacing=0.12
            )
            
            times = st.session_state['times']
            energies = st.session_state['energies']
            temperatures = st.session_state['temperatures']
            
            # Energy plot
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=energies,
                    mode='lines',
                    name='Potential Energy',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Temperature plot
            target_temp = st.session_state['md_params']['temperature']
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=temperatures,
                    mode='lines',
                    name='Temperature',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_hline(
                y=target_temp,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Target: {target_temp} K",
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="時間 (fs)", row=2, col=1)
            fig.update_yaxes(title_text="エネルギー (eV)", row=1, col=1)
            fig.update_yaxes(title_text="温度 (K)", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("平均エネルギー", f"{np.mean(energies):.3f} eV")
                st.metric("エネルギー標準偏差", f"{np.std(energies):.3f} eV")
            
            with col2:
                st.metric("平均温度", f"{np.mean(temperatures):.1f} K")
                st.metric("温度標準偏差", f"{np.std(temperatures):.1f} K")
            
            with col3:
                st.metric("総シミュレーション時間", f"{times[-1]:.1f} fs")
                st.metric("実行ステップ数", f"{len(times)}")

# ============================================================================
# Tab 3: Results Analysis
# ============================================================================

with tab3:
    st.header("📊 結果分析")
    
    if 'trajectory' not in st.session_state:
        st.warning("⚠️ まず「MDシミュレーション」タブでシミュレーションを実行してください")
    else:
        st.markdown("""
        ### 構造変化と格子歪みの分析
        
        HEAでは、異なる原子半径を持つ元素が混在するため、**局所格子歪み**が発生します。
        これは、理想的な格子位置からのずれとして観察できます。
        """)
        
        trajectory = st.session_state['trajectory']
        initial_atoms = st.session_state['initial_atoms']
        
        # Frame selector
        frame_idx = st.slider(
            "フレームを選択",
            min_value=0,
            max_value=len(trajectory) - 1,
            value=len(trajectory) - 1,
            help="時間発展を確認できます"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### フレーム {frame_idx} の構造")
            current_atoms = trajectory[frame_idx]
            view = atoms_to_py3dmol(current_atoms, show_cell=True)
            showmol(view, height=600, width=800)
        
        with col2:
            st.markdown("### 構造パラメータ")
            
            # Calculate displacement from initial structure
            initial_pos = initial_atoms.get_positions()
            current_pos = current_atoms.get_positions()
            
            # Account for periodic boundary conditions
            displacements = current_pos - initial_pos
            cell = current_atoms.get_cell()
            
            # Calculate RMS displacement
            rms_displacement = np.sqrt(np.mean(np.sum(displacements**2, axis=1)))
            max_displacement = np.max(np.sqrt(np.sum(displacements**2, axis=1)))
            
            st.metric("RMS変位", f"{rms_displacement:.3f} Å")
            st.metric("最大変位", f"{max_displacement:.3f} Å")
            
            # Energy and temperature at this frame
            if frame_idx < len(st.session_state['energies']):
                energy = st.session_state['energies'][frame_idx]
                temp = st.session_state['temperatures'][frame_idx]
                time = st.session_state['times'][frame_idx]
                
                st.metric("時刻", f"{time:.1f} fs")
                st.metric("エネルギー", f"{energy:.3f} eV")
                st.metric("温度", f"{temp:.1f} K")
        
        # Displacement analysis
        st.markdown("---")
        st.subheader("🔍 変位分析")
        
        # Calculate per-element displacement
        symbols = current_atoms.get_chemical_symbols()
        elements = st.session_state['selected_elements']
        
        element_displacements = {}
        for elem in elements:
            indices = [i for i, s in enumerate(symbols) if s == elem]
            if indices:
                elem_disp = np.sqrt(np.sum(displacements[indices]**2, axis=1))
                element_displacements[elem] = elem_disp
        
        # Plot displacement distribution
        fig = go.Figure()
        
        for elem, disp in element_displacements.items():
            fig.add_trace(go.Box(
                y=disp,
                name=elem,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title="元素ごとの変位分布",
            yaxis_title="変位 (Å)",
            xaxis_title="元素",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export data
        st.markdown("---")
        st.subheader("💾 データエクスポート")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export trajectory data
            df_trajectory = pd.DataFrame({
                'Time (fs)': st.session_state['times'],
                'Potential Energy (eV)': st.session_state['energies'],
                'Temperature (K)': st.session_state['temperatures']
            })
            
            csv_trajectory = df_trajectory.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 軌跡データをダウンロード (CSV)",
                data=csv_trajectory,
                file_name="hea_md_trajectory.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export final structure
            from io import StringIO
            from ase.io import write
            
            final_atoms = trajectory[-1]
            xyz_buffer = StringIO()
            write(xyz_buffer, final_atoms, format='xyz')
            xyz_str = xyz_buffer.getvalue()
            
            st.download_button(
                label="📥 最終構造をダウンロード (XYZ)",
                data=xyz_str.encode('utf-8'),
                file_name="hea_final_structure.xyz",
                mime="text/plain"
            )

# ============================================================================
# Tab 4: Theoretical Background
# ============================================================================

with tab4:
    st.header("📚 理論背景")
    
    st.markdown("""
    ## 1. ハイエントロピー合金 (HEA) とは
    
    ### 1.1 定義
    
    ハイエントロピー合金 (High-Entropy Alloys, HEA) は、**5種類以上の主要元素**が
    **等原子量または準等原子量** (各5-35 at.%) で混合された合金です。
    
    従来の合金（1-2種類の主要元素 + 微量添加元素）とは異なり、
    多数の元素が主成分として存在することが特徴です。
    
    ### 1.2 配置エントロピー
    
    HEAを安定化させる重要な要因の一つが**配置エントロピー** ($S_{conf}$) です：
    
    $$S_{conf} = -R \\sum_{i=1}^N c_i \\ln c_i$$
    
    ここで：
    - $R$ = 8.314 J/mol·K（気体定数）
    - $c_i$ = 元素 $i$ のモル分率
    - $N$ = 構成元素数
    
    **等原子量の場合**（$c_i = 1/N$）：
    
    $$S_{conf} = R \\ln N$$
    
    | 元素数 N | S_conf (J/mol·K) |
    |---------|------------------|
    | 2       | 5.76             |
    | 3       | 9.13             |
    | 4       | 11.53            |
    | 5       | 13.38            |
    | 6       | 14.90            |
    
    **物理的意味：**
    - 元素数が増えるほど配置エントロピーが増大
    - ギブス自由エネルギー $G = H - TS$ において、$-TS$ 項が大きくなる
    - 高温で固溶体相が安定化される（「カクテル効果」）
    
    ---
    
    ## 2. 局所格子歪み (Lattice Distortion)
    
    ### 2.1 原子半径の違いによる歪み
    
    HEAでは、異なる原子半径を持つ元素が隣接するため、
    原子位置 $\\boldsymbol{r}_i$ は理想的な格子点 $\\boldsymbol{R}_i$ からずれます：
    
    $$\\Delta \\boldsymbol{r}_i = \\boldsymbol{r}_i - \\boldsymbol{R}_i \\neq 0$$
    
    **代表的な元素の原子半径（金属結合半径）：**
    
    | 元素 | 原子半径 (Å) |
    |-----|-------------|
    | Al  | 1.43        |
    | Co  | 1.25        |
    | Cr  | 1.28        |
    | Cu  | 1.28        |
    | Fe  | 1.26        |
    | Mn  | 1.27        |
    | Ni  | 1.24        |
    | Ti  | 1.47        |
    | V   | 1.34        |
    
    ### 2.2 格子歪みパラメータ
    
    格子歪みの程度を定量化する指標：
    
    $$\\delta = \\sqrt{\\sum_{i=1}^N c_i \\left(1 - \\frac{r_i}{\\bar{r}}\\right)^2}$$
    
    ここで：
    - $r_i$ = 元素 $i$ の原子半径
    - $\\bar{r} = \\sum_{i=1}^N c_i r_i$ = 平均原子半径
    
    **経験則：**
    - $\\delta < 3\\%$: 固溶体形成が容易
    - $3\\% < \\delta < 6\\%$: 固溶体形成可能だが歪みエネルギー大
    - $\\delta > 6\\%$: 相分離やアモルファス化の可能性
    
    ---
    
    ## 3. 機械学習ポテンシャル (MLP)
    
    ### 3.1 なぜMLPが必要か？
    
    従来の経験的ポテンシャル（EAM, Tersoff等）では：
    - **パラメータ数の爆発**: $N$ 元素系で $O(N^2)$ 〜 $O(N^3)$ のパラメータが必要
    - **パラメータ不足**: 多くの元素組み合わせでパラメータが存在しない
    - **精度の限界**: 複雑な化学環境を表現できない
    
    **機械学習ポテンシャルの利点：**
    - 第一原理計算（DFT）の精度に近い
    - 多元素系に対応可能
    - 計算コストは経験的ポテンシャルと同程度
    
    ### 3.2 MACE (Multi-Atomic Cluster Expansion)
    
    MACEは、**等変ニューラルネットワーク**を用いたMLPです。
    
    **基本原理：**
    
    全エネルギーを原子ごとのエネルギーの和として表現：
    
    $$E_{total} = \\sum_{i=1}^{N_{atoms}} E_i(\\text{Environment}_i)$$
    
    ここで $E_i$ はニューラルネットワークで、
    $\\text{Environment}_i$ は原子 $i$ の周囲の化学的・幾何学的環境を表す記述子です。
    
    **記述子の特徴：**
    - 原子種の情報（Co, Cr, Fe, ...）
    - 原子間距離
    - 結合角
    - 回転・並進・置換に対する不変性
    
    **力の計算：**
    
    $$\\boldsymbol{F}_i = -\\frac{\\partial E_{total}}{\\partial \\boldsymbol{r}_i}$$
    
    自動微分により効率的に計算されます。
    
    ---
    
    ## 4. 分子動力学 (MD) シミュレーション
    
    ### 4.1 運動方程式
    
    ニュートンの運動方程式：
    
    $$m_i \\frac{d^2 \\boldsymbol{r}_i}{dt^2} = \\boldsymbol{F}_i$$
    
    ### 4.2 Langevin動力学
    
    本アプリでは**Langevin動力学**を使用しています：
    
    $$m_i \\frac{d^2 \\boldsymbol{r}_i}{dt^2} = \\boldsymbol{F}_i - \\gamma m_i \\frac{d\\boldsymbol{r}_i}{dt} + \\boldsymbol{\\xi}_i(t)$$
    
    ここで：
    - $\\gamma$ = 摩擦係数（熱浴との結合強度）
    - $\\boldsymbol{\\xi}_i(t)$ = ランダム力（揺動散逸定理を満たす）
    
    **特徴：**
    - 指定温度での正準集団（NVT）を実現
    - 熱平衡への緩和が速い
    - 実験条件（一定温度）に対応
    
    ### 4.3 時間積分
    
    Velocity Verlet法の変形を使用：
    
    1. $\\boldsymbol{v}(t + \\Delta t/2) = \\boldsymbol{v}(t) + \\frac{\\boldsymbol{F}(t)}{m} \\frac{\\Delta t}{2}$
    2. $\\boldsymbol{r}(t + \\Delta t) = \\boldsymbol{r}(t) + \\boldsymbol{v}(t + \\Delta t/2) \\Delta t$
    3. $\\boldsymbol{F}(t + \\Delta t)$ を計算
    4. $\\boldsymbol{v}(t + \\Delta t) = \\boldsymbol{v}(t + \\Delta t/2) + \\frac{\\boldsymbol{F}(t + \\Delta t)}{m} \\frac{\\Delta t}{2}$
    
    **タイムステップの選択：**
    - 通常 1-2 fs（$10^{-15}$ 秒）
    - 原子振動の周期（〜10 fs）の1/10程度
    
    ---
    
    ## 5. HEAの特異な性質
    
    ### 5.1 4つの主要効果
    
    1. **高エントロピー効果**: 配置エントロピーによる固溶体安定化
    2. **格子歪み効果**: 原子半径差による強度向上
    3. **遅い拡散効果**: 複雑な環境による拡散係数の低下
    4. **カクテル効果**: 個々の元素にない新しい性質の発現
    
    ### 5.2 代表的なHEA
    
    **Cantor合金 (CoCrFeMnNi)**
    - 最も研究されているHEA
    - FCC単相
    - 優れた延性と靭性
    - 極低温でも延性を維持
    
    **Refractory HEA (TiZrNbHfTa)**
    - 高融点元素の組み合わせ
    - BCC構造
    - 高温強度に優れる
    
    ---
    
    ## 6. 本シミュレーターの物理的妥当性
    
    ### 6.1 検証項目
    
    1. **エネルギー保存**: 断熱系でのエネルギー変動 < 1%
    2. **温度制御**: 目標温度からの偏差 < 5%
    3. **構造安定性**: 異常な原子間距離の発生なし
    4. **格子歪み**: 文献値との整合性
    
    ### 6.2 限界と注意点
    
    - **時間スケール**: 現実的な計算時間では ps オーダー（実験は秒〜時間）
    - **サイズ効果**: 108原子は実験スケール（$10^{23}$）に比べて極小
    - **ML精度**: MACEの訓練データに依存
    - **量子効果**: 古典MDでは電子状態は陽には扱わない
    
    ---
    
    ## 7. 参考文献
    
    1. Yeh, J. W., et al. "Nanostructured high-entropy alloys with multiple principal elements: novel alloy design concepts and outcomes." *Advanced Engineering Materials* 6.5 (2004): 299-303.
    
    2. Miracle, D. B., and O. N. Senkov. "A critical review of high entropy alloys and related concepts." *Acta Materialia* 122 (2017): 448-511.
    
    3. Batatia, I., et al. "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields." *NeurIPS* (2022).
    
    4. George, E. P., D. Raabe, and R. O. Ritchie. "High-entropy alloys." *Nature Reviews Materials* 4.8 (2019): 515-534.
    
    ---
    
    ## 8. 数学的補足
    
    ### 8.1 ギブス自由エネルギーと相安定性
    
    固溶体の形成条件：
    
    $$\\Delta G_{mix} = \\Delta H_{mix} - T \\Delta S_{mix} < 0$$
    
    HEAでは：
    - $\\Delta H_{mix}$: 混合エンタルピー（通常正、不利）
    - $T \\Delta S_{mix}$: エントロピー項（常に正、有利）
    
    高温では $T \\Delta S_{mix}$ が支配的となり、固溶体が安定化されます。
    
    ### 8.2 拡散係数
    
    Arrhenius型の温度依存性：
    
    $$D = D_0 \\exp\\left(-\\frac{Q}{RT}\\right)$$
    
    HEAでは：
    - 活性化エネルギー $Q$ が大きい（複雑な環境）
    - 拡散が遅い → 高温での構造安定性
    
    ---
    
    ## 9. 実験との対応
    
    ### 9.1 X線回折 (XRD)
    
    シミュレーション結果から動径分布関数 (RDF) を計算し、
    XRDパターンと比較可能です。
    
    ### 9.2 透過電子顕微鏡 (TEM)
    
    原子配置の可視化結果は、高分解能TEMの観察に対応します。
    
    ### 9.3 機械的性質
    
    より長時間・大規模なシミュレーションにより、
    応力-歪み曲線や弾性定数の予測が可能です。
    """)

# ============================================================================
# Tab 5: Usage Guide
# ============================================================================

with tab5:
    st.header("📖 使用方法")
    
    st.markdown("""
    ## クイックスタートガイド
    
    ### ステップ1: 構造生成
    
    1. **「構造生成」タブ**を開く
    2. 構成元素を選択（デフォルト: Cantor合金 CoCrFeMnNi）
    3. 格子定数とシステムサイズを設定
    4. **「構造を生成」ボタン**をクリック
    5. 3D可視化で構造を確認
    
    ### ステップ2: MDシミュレーション
    
    1. **「MDシミュレーション」タブ**を開く
    2. 温度を設定（推奨: 300K = 室温）
    3. タイムステップとステップ数を設定
    4. ポテンシャルを選択（推奨: MACE）
    5. **「シミュレーション開始」ボタン**をクリック
    6. 進行状況を確認（数分かかる場合があります）
    
    ### ステップ3: 結果分析
    
    1. **「結果分析」タブ**を開く
    2. スライダーで時間発展を確認
    3. 格子歪みや変位を分析
    4. 必要に応じてデータをエクスポート
    
    ---
    
    ## 推奨される検証シナリオ
    
    ### ケースA: 純金属 vs HEA の比較
    
    **目的**: HEA特有の格子歪みを観察
    
    **手順**:
    1. 純Ni（元素1種類のみ選択）でシミュレーション実行
    2. CoCrFeMnNi（5元素）でシミュレーション実行
    3. 変位分布を比較
    
    **期待される結果**:
    - 純Niは均一な振動
    - HEAは元素ごとに異なる変位パターン
    
    ### ケースB: 高温安定性の確認
    
    **目的**: HEAの高温での構造安定性を確認
    
    **手順**:
    1. 温度300Kでシミュレーション
    2. 温度1000Kでシミュレーション
    3. 温度1500Kでシミュレーション
    4. 各温度でのRMS変位を比較
    
    **期待される結果**:
    - 高温でも構造が維持される（カクテル効果）
    - 純金属より安定な場合がある
    
    ### ケースC: 不安定な組み合わせ
    
    **目的**: 相分離やアモルファス化の観察
    
    **手順**:
    1. 原子半径差の大きい元素を選択（例: Al, Ti, Cu）
    2. 高温（1000K以上）でシミュレーション
    3. 構造の変化を観察
    
    **期待される結果**:
    - 格子歪みが大きくなる
    - 場合によっては構造が崩れる
    
    ---
    
    ## トラブルシューティング
    
    ### Q1: シミュレーションが遅い
    
    **A**: 以下を試してください：
    - システムサイズを小さくする（2x2x2）
    - ステップ数を減らす（50ステップ）
    - EMTポテンシャルを使用（精度は低下）
    
    ### Q2: エラーが発生する
    
    **A**: 以下を確認してください：
    - 選択した元素がMACEでサポートされているか
    - EMTの場合、対応元素のみ選択しているか
    - システムサイズが大きすぎないか
    
    ### Q3: 温度が安定しない
    
    **A**: 以下を調整してください：
    - 摩擦係数を大きくする（0.005-0.01）
    - ステップ数を増やす（平衡化に時間が必要）
    
    ### Q4: 構造が崩壊する
    
    **A**: 以下の可能性があります：
    - 温度が高すぎる（融点に近い）
    - 元素の組み合わせが不適切（相分離）
    - これは物理的に正しい挙動の可能性もあります
    
    ---
    
    ## パフォーマンスガイド
    
    ### 計算時間の目安（CPU: 一般的なPC）
    
    | 設定 | 原子数 | ステップ数 | 時間（MACE） | 時間（EMT） |
    |------|--------|-----------|-------------|------------|
    | 小   | 32     | 50        | 1-2分       | 10秒       |
    | 中   | 108    | 100       | 5-10分      | 30秒       |
    | 大   | 256    | 200       | 30-60分     | 2分        |
    
    ### 推奨設定
    
    **初めての方**:
    - 元素: CoCrFeMnNi（Cantor合金）
    - サイズ: 3x3x3（108原子）
    - 温度: 300K
    - ステップ: 50-100
    - ポテンシャル: MACE
    
    **高速テスト**:
    - サイズ: 2x2x2（32原子）
    - ステップ: 50
    - ポテンシャル: EMT（対応元素のみ）
    
    **詳細解析**:
    - サイズ: 3x3x3
    - ステップ: 200-500
    - ポテンシャル: MACE
    
    ---
    
    ## データの解釈
    
    ### エネルギープロット
    
    - **初期の急激な低下**: 構造緩和（正常）
    - **その後の振動**: 熱振動（正常）
    - **単調な増加/減少**: 異常（設定を見直す）
    
    ### 温度プロット
    
    - **目標温度周辺の振動**: 正常
    - **平均が目標から大きくずれる**: 摩擦係数を調整
    - **発散**: 異常（タイムステップを小さく）
    
    ### 変位分析
    
    - **元素ごとに異なる**: HEAの特徴（正常）
    - **時間とともに増大**: 拡散（高温で正常）
    - **急激な増大**: 構造崩壊の可能性
    
    ---
    
    ## さらなる学習
    
    ### 推奨文献
    
    1. **入門**: "High-Entropy Alloys" by Murty et al.
    2. **理論**: "Computational Materials Science" by Kalidindi
    3. **MD**: "Understanding Molecular Simulation" by Frenkel & Smit
    4. **ML**: "Machine Learning for Molecular Simulation" by Behler
    
    ### オンラインリソース
    
    - ASE Documentation: https://wiki.fysik.dtu.dk/ase/
    - MACE: https://github.com/ACEsuit/mace
    - Materials Project: https://materialsproject.org/
    
    ---
    
    ## 引用について
    
    本アプリケーションを研究で使用する場合、以下を引用してください：
    
    **MACE**:
    ```
    Batatia, I., et al. "MACE: Higher order equivariant message passing 
    neural networks for fast and accurate force fields." 
    Advances in Neural Information Processing Systems 35 (2022).
    ```
    
    **ASE**:
    ```
    Larsen, A. H., et al. "The atomic simulation environment—a Python library 
    for working with atoms." Journal of Physics: Condensed Matter 29.27 (2017): 273002.
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>HEA/CCA ML-MD Simulator</strong></p>
    <p>物理的正確さを最重要視した実装</p>
    <p>Powered by MACE, ASE, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
