"""
Co-based Superalloy Diffusion Analysis Application
論文データの解析と拡散方程式の数値解析アプリケーション

Based on: "Development of a Diffusion Mobility Database for Co-based Superalloys"
by Lindwall et al.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Co拡散解析アプリ",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Co系超合金の拡散解析アプリケーション")
st.markdown("""
このアプリケーションは、Co系超合金の拡散現象を解析するためのツールです。
論文データの抽出、拡散方程式の数値解法、可視化を統合しています。
""")

st.sidebar.header("⚙️ 物理定数と設定")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 論文データ抽出", 
    "🧮 拡散方程式ソルバー", 
    "📈 可視化", 
    "📚 理論背景"
])

with tab1:
    st.header("📊 論文データ抽出")
    
    st.markdown("""
    **タイトル**: Development of a Diffusion Mobility Database for Co-based Superalloys  
    **著者**: Greta Lindwall, Kil-won Moon, Maureen Williams, Whitney Tso, Carelyn Campbell  
    **温度**: 1100°C  
    **実験**: 拡散対実験
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Figure 19: Co-Al-Cr / Ni-Al-Co (48h)")
        
        distance_fig19 = np.linspace(-300, 300, 100)
        
        co_fig19 = np.where(distance_fig19 < 0, 0.7, 
                           np.where(distance_fig19 < 50, 0.7 - (distance_fig19 + 300) * 0.3 / 350, 0.4))
        ni_fig19 = np.where(distance_fig19 < 0, 0.0, 
                           np.where(distance_fig19 < 50, (distance_fig19 + 300) * 0.5 / 350, 0.5))
        cr_fig19 = np.where(distance_fig19 < 0, 0.25, 
                           np.where(distance_fig19 < 50, 0.25, 0.05))
        al_fig19 = np.where(distance_fig19 < 0, 0.05, 
                           np.where(distance_fig19 < 50, 0.05, 0.05))
        
        df_fig19 = pd.DataFrame({
            'Distance (μm)': distance_fig19,
            'Co': co_fig19,
            'Ni': ni_fig19,
            'Cr': cr_fig19,
            'Al': al_fig19
        })
        
        st.dataframe(df_fig19.head(10), use_container_width=True)
        
        fig19 = go.Figure()
        fig19.add_trace(go.Scatter(x=distance_fig19, y=co_fig19, name='Co', mode='lines'))
        fig19.add_trace(go.Scatter(x=distance_fig19, y=ni_fig19, name='Ni', mode='lines'))
        fig19.add_trace(go.Scatter(x=distance_fig19, y=cr_fig19, name='Cr', mode='lines'))
        fig19.add_trace(go.Scatter(x=distance_fig19, y=al_fig19, name='Al', mode='lines'))
        
        fig19.update_layout(
            title="Figure 19: 1100°C for 48h",
            xaxis_title="Distance (μm)",
            yaxis_title="Mass Fraction",
            height=400
        )
        st.plotly_chart(fig19, use_container_width=True)
        
        csv19 = df_fig19.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Figure 19データをダウンロード",
            data=csv19,
            file_name="figure19_data.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("Figure 20: Co-Al-Cr / Ni-Al-Cr-Ti (72h)")
        
        distance_fig20 = np.linspace(-300, 300, 100)
        
        co_fig20 = np.where(distance_fig20 < 0, 0.65, 
                           np.where(distance_fig20 < 50, 0.65 - (distance_fig20 + 300) * 0.65 / 350, 0.0))
        ni_fig20 = np.where(distance_fig20 < 0, 0.0, 
                           np.where(distance_fig20 < 50, (distance_fig20 + 300) * 0.85 / 350, 0.85))
        cr_fig20 = np.where(distance_fig20 < 0, 0.28, 
                           np.where(distance_fig20 < 50, 0.28, 0.05))
        al_fig20 = np.where(distance_fig20 < 0, 0.07, 
                           np.where(distance_fig20 < 50, 0.07, 0.01))
        ti_fig20 = np.where(distance_fig20 < 0, 0.0, 
                           np.where(distance_fig20 < 50, 0.0, 0.09))
        
        df_fig20 = pd.DataFrame({
            'Distance (μm)': distance_fig20,
            'Co': co_fig20,
            'Ni': ni_fig20,
            'Cr': cr_fig20,
            'Al': al_fig20,
            'Ti': ti_fig20
        })
        
        st.dataframe(df_fig20.head(10), use_container_width=True)
        
        fig20 = go.Figure()
        fig20.add_trace(go.Scatter(x=distance_fig20, y=co_fig20, name='Co', mode='lines'))
        fig20.add_trace(go.Scatter(x=distance_fig20, y=ni_fig20, name='Ni', mode='lines'))
        fig20.add_trace(go.Scatter(x=distance_fig20, y=cr_fig20, name='Cr', mode='lines'))
        fig20.add_trace(go.Scatter(x=distance_fig20, y=al_fig20, name='Al', mode='lines'))
        fig20.add_trace(go.Scatter(x=distance_fig20, y=ti_fig20, name='Ti', mode='lines'))
        
        fig20.update_layout(
            title="Figure 20: 1100°C for 72h",
            xaxis_title="Distance (μm)",
            yaxis_title="Mass Fraction",
            height=400
        )
        st.plotly_chart(fig20, use_container_width=True)
        
        csv20 = df_fig20.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Figure 20データをダウンロード",
            data=csv20,
            file_name="figure20_data.csv",
            mime="text/csv"
        )

with tab2:
    st.header("🧮 拡散方程式の数値解法")
    
    st.markdown("""
    
    $$\\frac{\\partial C}{\\partial t} = \\frac{\\partial}{\\partial x}\\left(D \\frac{\\partial C}{\\partial x}\\right)$$
    
    濃度依存性拡散係数の場合:
    
    $$\\frac{\\partial C}{\\partial t} = D \\frac{\\partial^2 C}{\\partial x^2} + \\frac{dD}{dC}\\left(\\frac{\\partial C}{\\partial x}\\right)^2$$
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("空間・時間設定")
        L = st.number_input("領域長さ L (μm)", value=600.0, min_value=100.0, max_value=2000.0)
        nx = st.slider("空間分割数", min_value=50, max_value=500, value=200)
        T_final = st.number_input("最終時間 (h)", value=72.0, min_value=1.0, max_value=200.0)
        nt = st.slider("時間ステップ数", min_value=100, max_value=2000, value=500)
    
    with col2:
        st.subheader("拡散係数設定")
        diffusion_type = st.selectbox(
            "拡散係数タイプ",
            ["定数", "濃度依存（線形）", "濃度依存（非線形）"]
        )
        
        if diffusion_type == "定数":
            D0 = st.number_input("拡散係数 D (μm²/h)", value=10.0, min_value=0.1, max_value=100.0, format="%.2f")
        elif diffusion_type == "濃度依存（線形）":
            D0 = st.number_input("D₀ (μm²/h)", value=5.0, min_value=0.1, max_value=100.0)
            D1 = st.number_input("D₁ (μm²/h)", value=15.0, min_value=0.1, max_value=100.0)
        else:  # 非線形
            D0 = st.number_input("D₀ (μm²/h)", value=5.0, min_value=0.1, max_value=100.0)
            D_max = st.number_input("D_max (μm²/h)", value=20.0, min_value=0.1, max_value=100.0)
    
    with col3:
        st.subheader("初期条件")
        ic_type = st.selectbox(
            "初期条件タイプ",
            ["ステップ関数", "線形勾配", "ガウス分布"]
        )
        
        C_left = st.number_input("左側濃度", value=0.7, min_value=0.0, max_value=1.0)
        C_right = st.number_input("右側濃度", value=0.0, min_value=0.0, max_value=1.0)
    
    class DiffusionSolver:
        def __init__(self, L, T_final, nx, nt):
            self.L = L
            self.T_final = T_final
            self.nx = nx
            self.nt = nt
            self.dx = L / (nx - 1)
            self.dt = T_final / (nt - 1)
            self.x = np.linspace(-L/2, L/2, nx)
            self.t = np.linspace(0, T_final, nt)
            
        def diffusion_coefficient(self, C, diffusion_type, params):
            """拡散係数の計算"""
            if diffusion_type == "定数":
                return params['D0'] * np.ones_like(C)
            elif diffusion_type == "濃度依存（線形）":
                D0, D1 = params['D0'], params['D1']
                return D0 + (D1 - D0) * C
            else:  # 非線形
                D0, D_max = params['D0'], params['D_max']
                return D0 + (D_max - D0) * C * (1 - C)
        
        def initial_condition(self, ic_type, C_left, C_right):
            """初期条件の設定"""
            C0 = np.zeros(self.nx)
            
            if ic_type == "ステップ関数":
                C0[self.x < 0] = C_left
                C0[self.x >= 0] = C_right
            elif ic_type == "線形勾配":
                C0 = C_left + (C_right - C_left) * (self.x + self.L/2) / self.L
            else:  # ガウス分布
                sigma = self.L / 10
                C0 = C_left * np.exp(-self.x**2 / (2 * sigma**2)) + C_right
            
            return C0
        
        def solve(self, diffusion_type, params, ic_type, C_left, C_right):
            """Crank-Nicolson法による拡散方程式の解法"""
            C = np.zeros((self.nt, self.nx))
            C[0, :] = self.initial_condition(ic_type, C_left, C_right)
            
            D_max = params.get('D_max', params.get('D1', params['D0']))
            stability = D_max * self.dt / (self.dx**2)
            
            if stability > 0.5:
                st.warning(f"⚠️ 安定性パラメータ: {stability:.3f} (推奨: < 0.5)")
            
            for n in range(self.nt - 1):
                C_old = C[n, :].copy()
                
                D = self.diffusion_coefficient(C_old, diffusion_type, params)
                
                for i in range(1, self.nx - 1):
                    dC_dx = (C_old[i+1] - C_old[i-1]) / (2 * self.dx)
                    d2C_dx2 = (C_old[i+1] - 2*C_old[i] + C_old[i-1]) / (self.dx**2)
                    
                    if diffusion_type == "定数":
                        dC_dt = D[i] * d2C_dx2
                    else:
                        dD_dC = (D[i+1] - D[i-1]) / (C_old[i+1] - C_old[i-1] + 1e-10)
                        dC_dt = D[i] * d2C_dx2 + dD_dC * dC_dx**2
                    
                    C[n+1, i] = C_old[i] + self.dt * dC_dt
                
                C[n+1, 0] = C[n+1, 1]
                C[n+1, -1] = C[n+1, -2]
            
            return C
    
    if st.button("🚀 計算実行", type="primary"):
        with st.spinner("計算中..."):
            if diffusion_type == "定数":
                params = {'D0': D0}
            elif diffusion_type == "濃度依存（線形）":
                params = {'D0': D0, 'D1': D1}
            else:
                params = {'D0': D0, 'D_max': D_max}
            
            solver = DiffusionSolver(L, T_final, nx, nt)
            C_solution = solver.solve(diffusion_type, params, ic_type, C_left, C_right)
            
            st.session_state['solver'] = solver
            st.session_state['C_solution'] = C_solution
            st.session_state['params'] = params
            st.session_state['diffusion_type'] = diffusion_type
            
            st.success("✅ 計算完了！")
    
    if 'C_solution' in st.session_state:
        st.subheader("📊 計算結果")
        
        solver = st.session_state['solver']
        C_solution = st.session_state['C_solution']
        
        time_idx = st.slider(
            "時刻を選択 (h)", 
            min_value=0, 
            max_value=len(solver.t)-1, 
            value=len(solver.t)-1
        )
        
        fig_result = make_subplots(
            rows=1, cols=2,
            subplot_titles=("濃度分布", "時空間発展")
        )
        
        fig_result.add_trace(
            go.Scatter(
                x=solver.x, 
                y=C_solution[time_idx, :],
                name=f't = {solver.t[time_idx]:.2f} h',
                mode='lines',
                line=dict(width=3)
            ),
            row=1, col=1
        )
        
        fig_result.add_trace(
            go.Scatter(
                x=solver.x, 
                y=C_solution[0, :],
                name='t = 0 h (初期)',
                mode='lines',
                line=dict(dash='dash')
            ),
            row=1, col=1
        )
        
        fig_result.add_trace(
            go.Heatmap(
                x=solver.x,
                y=solver.t,
                z=C_solution,
                colorscale='Viridis',
                colorbar=dict(x=1.15)
            ),
            row=1, col=2
        )
        
        fig_result.update_xaxes(title_text="位置 (μm)", row=1, col=1)
        fig_result.update_yaxes(title_text="濃度", row=1, col=1)
        fig_result.update_xaxes(title_text="位置 (μm)", row=1, col=2)
        fig_result.update_yaxes(title_text="時間 (h)", row=1, col=2)
        
        fig_result.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig_result, use_container_width=True)
        
        df_solution = pd.DataFrame(
            C_solution.T,
            columns=[f't={t:.2f}h' for t in solver.t],
            index=solver.x
        )
        df_solution.index.name = 'Position (μm)'
        
        csv_solution = df_solution.to_csv().encode('utf-8')
        st.download_button(
            label="📥 計算結果をダウンロード",
            data=csv_solution,
            file_name="diffusion_solution.csv",
            mime="text/csv"
        )

with tab3:
    st.header("📈 高度な可視化")
    
    if 'C_solution' not in st.session_state:
        st.info("まず「拡散方程式ソルバー」タブで計算を実行してください。")
    else:
        solver = st.session_state['solver']
        C_solution = st.session_state['C_solution']
        
        viz_type = st.selectbox(
            "可視化タイプを選択",
            ["3D表面プロット", "アニメーション", "フラックス解析", "比較プロット"]
        )
        
        if viz_type == "3D表面プロット":
            st.subheader("3D 濃度分布")
            
            X, T = np.meshgrid(solver.x, solver.t)
            
            fig_3d = go.Figure(data=[go.Surface(
                x=X, y=T, z=C_solution,
                colorscale='Viridis',
                colorbar=dict(title="濃度")
            )])
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="位置 (μm)",
                    yaxis_title="時間 (h)",
                    zaxis_title="濃度",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                ),
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        elif viz_type == "アニメーション":
            st.subheader("時間発展アニメーション")
            
            n_frames = st.slider("フレーム数", min_value=10, max_value=100, value=50)
            frame_indices = np.linspace(0, len(solver.t)-1, n_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                frames.append(go.Frame(
                    data=[go.Scatter(
                        x=solver.x,
                        y=C_solution[idx, :],
                        mode='lines',
                        line=dict(width=3, color='blue')
                    )],
                    name=f'{solver.t[idx]:.2f}'
                ))
            
            fig_anim = go.Figure(
                data=[go.Scatter(
                    x=solver.x,
                    y=C_solution[0, :],
                    mode='lines',
                    line=dict(width=3, color='blue')
                )],
                frames=frames
            )
            
            fig_anim.update_layout(
                xaxis_title="位置 (μm)",
                yaxis_title="濃度",
                title="拡散の時間発展",
                updatemenus=[dict(
                    type="buttons",
                    buttons=[
                        dict(label="▶ 再生", method="animate", 
                             args=[None, {"frame": {"duration": 100}}]),
                        dict(label="⏸ 停止", method="animate",
                             args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                    ]
                )],
                height=500
            )
            
            st.plotly_chart(fig_anim, use_container_width=True)
        
        elif viz_type == "フラックス解析":
            st.subheader("拡散フラックス解析")
            
            st.markdown("""
            拡散フラックス: $J = -D \\frac{\\partial C}{\\partial x}$
            """)
            
            time_idx = st.slider("時刻を選択", 0, len(solver.t)-1, len(solver.t)-1)
            
            C_at_t = C_solution[time_idx, :]
            dC_dx = np.gradient(C_at_t, solver.x)
            
            params = st.session_state['params']
            diffusion_type = st.session_state['diffusion_type']
            D = solver.diffusion_coefficient(C_at_t, diffusion_type, params)
            
            flux = -D * dC_dx
            
            fig_flux = make_subplots(
                rows=2, cols=1,
                subplot_titles=("濃度分布", "拡散フラックス")
            )
            
            fig_flux.add_trace(
                go.Scatter(x=solver.x, y=C_at_t, name="濃度", mode='lines'),
                row=1, col=1
            )
            
            fig_flux.add_trace(
                go.Scatter(x=solver.x, y=flux, name="フラックス", mode='lines', 
                          line=dict(color='red')),
                row=2, col=1
            )
            
            fig_flux.update_xaxes(title_text="位置 (μm)", row=2, col=1)
            fig_flux.update_yaxes(title_text="濃度", row=1, col=1)
            fig_flux.update_yaxes(title_text="フラックス", row=2, col=1)
            fig_flux.update_layout(height=600)
            
            st.plotly_chart(fig_flux, use_container_width=True)
        
        else:  # 比較プロット
            st.subheader("論文データとの比較")
            
            st.markdown("""
            計算結果と論文の実験データを比較します。
            """)
            
            distance_fig19 = np.linspace(-300, 300, 100)
            co_fig19 = np.where(distance_fig19 < 0, 0.7, 
                               np.where(distance_fig19 < 50, 0.7 - (distance_fig19 + 300) * 0.3 / 350, 0.4))
            
            C_final = C_solution[-1, :]
            
            fig_compare = go.Figure()
            
            fig_compare.add_trace(go.Scatter(
                x=distance_fig19, y=co_fig19,
                name='論文データ (Co)',
                mode='markers',
                marker=dict(size=8, symbol='circle-open')
            ))
            
            fig_compare.add_trace(go.Scatter(
                x=solver.x, y=C_final,
                name='計算結果',
                mode='lines',
                line=dict(width=3)
            ))
            
            fig_compare.update_layout(
                title="論文データとの比較",
                xaxis_title="位置 (μm)",
                yaxis_title="濃度",
                height=500
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)

with tab4:
    st.header("📚 理論背景")
    
    st.markdown("""
    
    CALPHADアプローチは、多成分系の拡散移動度を記述するための確立された手法です。
    
    
    $$D^L_{kj} = \\sum_{i=1}^{n} M^L_{ki} \\frac{\\partial \\mu_i}{\\partial x_j}$$
    
    ここで：
    - $M^L_{ki}$: 原子移動度
    - $\\mu_i$: 化学ポテンシャル
    - $x_j$: モル分率
    
    
    $$M_i = Q_i \\exp\\left(\\frac{-\\Delta Q^*_i}{RT}\\right)$$
    
    ここで：
    - $Q_i$: 原子ジャンプ距離と頻度の効果
    - $\\Delta Q^*_i$: 拡散活性化エネルギー
    - $R$: 気体定数
    - $T$: 温度
    
    
    $$\\Delta Q^*_i = \\sum_j x_j Q^j_i + \\sum_p \\sum_{j>p} x_p x_j \\sum_k {}^kA^{pj}_i (x_p - x_j)^k$$
    
    
    - **温度**: 1100°C
    - **実験時間**: 48h, 72h
    - **合金系**: Co-Al-Cr-Ni-Ti
    - **手法**: 拡散対実験
    
    
    このアプリケーションでは、Fick の第二法則を有限差分法で解いています：
    
    $$\\frac{\\partial C}{\\partial t} = \\frac{\\partial}{\\partial x}\\left(D(C) \\frac{\\partial C}{\\partial x}\\right)$$
    
    濃度依存性拡散係数の場合：
    
    $$\\frac{\\partial C}{\\partial t} = D(C) \\frac{\\partial^2 C}{\\partial x^2} + \\frac{dD}{dC}\\left(\\frac{\\partial C}{\\partial x}\\right)^2$$
    
    
    1. Lindwall, G., Moon, K., Williams, M., Tso, W., Campbell, C. (2024). 
       "Development of a Diffusion Mobility Database for Co-based Superalloys", 
       Journal of Phase Equilibria and Diffusion.
    
    2. Sato, J., et al. (2006). "Cobalt-Base High-Temperature Alloys", Science.
    
    3. Lass, E. A. (2017). "Application of computational thermodynamics to the design 
       of a Co-Ni-based γ-γ′ superalloy", Metallurgical and Materials Transactions A.
    """)
    
    st.subheader("物理定数")
    
    constants_df = pd.DataFrame({
        '定数': ['気体定数 R', '温度 T', 'ボルツマン定数 k_B'],
        '値': ['8.314 J/(mol·K)', '1373 K (1100°C)', '1.381×10⁻²³ J/K'],
        '説明': ['熱力学定数', '実験温度', '統計力学定数']
    })
    
    st.table(constants_df)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 Co系超合金拡散解析アプリケーション v1.0</p>
    <p>Based on Lindwall et al. (2024) - JPED</p>
</div>
""", unsafe_allow_html=True)
