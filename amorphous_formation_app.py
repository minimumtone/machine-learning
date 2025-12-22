"""
Amorphous Formation Model Validation Application
非晶質形成モデル妥当性検証アプリケーション

This Streamlit application provides an interactive interface for validating
amorphous (glass) formation models using CALPHAD thermodynamics and
Davis-Uhlmann kinetics.

Based on the validation instruction document for students.

Usage:
    streamlit run amorphous_formation_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from amorphous_formation.materials_database import MaterialsDatabase, Material
from amorphous_formation.calphad_thermodynamics import CALPHADThermodynamics
from amorphous_formation.doolittle_viscosity import DoolittleViscosity
from amorphous_formation.davis_uhlmann_model import DavisUhlmannModel
from amorphous_formation.sensitivity_analysis import SensitivityAnalysis
from amorphous_formation.visualization import AmorphousVisualization

st.set_page_config(
    page_title="非晶質形成モデル検証",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 非晶質形成モデルの妥当性検証")
st.markdown("""
このアプリケーションは、**CALPHAD熱力学**と**Davis-Uhlmannモデル**を組み合わせて、
液体の急冷時にガラス（非晶質）が形成されるか、結晶化するかを予測するモデルの検証を行います。

**検証の目的:**
1. 熱力学的妥当性: CALPHADから得たエネルギー差が温度低下とともに正しく増大しているか
2. 動力学的妥当性: Doolittle式に基づく粘性が $T_g$ 付近で急増しているか
3. 予測精度: 既知の材料の臨界冷却速度 $R_c$ の実験値と一致するか
""")

st.sidebar.header("⚙️ パラメータ設定")

db = MaterialsDatabase()

input_mode = st.sidebar.radio(
    "入力モード",
    ["既存材料を選択", "カスタムパラメータ"]
)

if input_mode == "既存材料を選択":
    material_name = st.sidebar.selectbox(
        "材料を選択",
        db.list_materials(),
        index=2
    )
    material = db.get_material(material_name)
    
    st.sidebar.markdown(f"""
    **選択した材料:** {material.composition}
    - $T_m$ = {material.T_m:.0f} K
    - $T_g$ = {material.T_g:.0f} K
    - $\\Delta H_f$ = {material.delta_H_f:.0f} J/mol
    - $R_c$ (実験) = {material.R_c_exp:.1e} K/s
    """)
    
    T_m = material.T_m
    T_g = material.T_g
    delta_H_f = material.delta_H_f
    sigma = material.sigma
    V_m = material.V_m
    eta_0 = material.eta_0
    D_star = material.D_star
    T_0 = material.T_0
    R_c_exp = material.R_c_exp
    
else:
    st.sidebar.subheader("熱力学パラメータ")
    T_m = st.sidebar.number_input("融点 $T_m$ [K]", value=937.0, min_value=300.0, max_value=3000.0)
    T_g = st.sidebar.number_input("ガラス転移点 $T_g$ [K]", value=625.0, min_value=200.0, max_value=2000.0)
    delta_H_f = st.sidebar.number_input("融解熱 $\\Delta H_f$ [J/mol]", value=8200.0, min_value=1000.0, max_value=50000.0)
    
    st.sidebar.subheader("界面・体積パラメータ")
    sigma = st.sidebar.number_input("界面エネルギー $\\sigma$ [J/m²]", value=0.08, min_value=0.01, max_value=0.5, format="%.4f")
    V_m = st.sidebar.number_input("モル体積 $V_m$ [m³/mol]", value=1.1e-5, min_value=1e-6, max_value=1e-4, format="%.2e")
    
    st.sidebar.subheader("粘度パラメータ")
    eta_0 = st.sidebar.number_input("前指数因子 $\\eta_0$ [Pa·s]", value=1e-5, min_value=1e-8, max_value=1e-2, format="%.2e")
    D_star = st.sidebar.number_input("脆弱性パラメータ $D^*$", value=18.5, min_value=5.0, max_value=100.0)
    T_0 = st.sidebar.number_input("VFT温度 $T_0$ [K]", value=T_g - 50.0, min_value=100.0, max_value=1500.0)
    
    R_c_exp = st.sidebar.number_input("実験臨界冷却速度 $R_c$ [K/s] (比較用)", value=1.0, min_value=1e-6, max_value=1e8, format="%.2e")
    material = None

thermo = CALPHADThermodynamics(T_m=T_m, delta_H_f=delta_H_f, T_g=T_g)
visc = DoolittleViscosity(T_m=T_m, T_g=T_g, eta_0=eta_0, D_star=D_star, T_0=T_0)
model = DavisUhlmannModel(T_m=T_m, T_g=T_g, delta_H_f=delta_H_f, sigma=sigma,
                          V_m=V_m, eta_0=eta_0, D_star=D_star, T_0=T_0)
analysis = SensitivityAnalysis(T_m=T_m, T_g=T_g, delta_H_f=delta_H_f, sigma=sigma,
                               V_m=V_m, eta_0=eta_0, D_star=D_star, T_0=T_0)
viz = AmorphousVisualization(figsize=(8, 5))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 ステップ1: 熱力学",
    "📈 ステップ2: 粘度",
    "🔄 ステップ3: TTT曲線",
    "🎯 ステップ4: 感度解析",
    "✅ 最終課題: 実測値比較",
    "📚 理論背景"
])

with tab1:
    st.header("ステップ1: 熱力学データのチェック（CALPHADの検証）")
    
    st.markdown("""
    ### 指示
    $T_m$（融点）付近から $T_g$（ガラス転移点）まで、$\\Delta G_m = G^L - G^S$ をプロットしなさい。
    
    ### 合格基準
    - $T = T_m$ で $\\Delta G_m = 0$ になっていること
    - 温度が下がるにつれて $\\Delta G_m$ が右肩上がりに増加していること
    - 計算した融解エントロピー $\\Delta S_f = \\Delta G_m / (T_m - T)$ が、文献値に近い一定値を示していること
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig1, ax1 = viz.plot_calphad_driving_force(thermo, show_validation=True)
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("検証結果")
        
        delta_G_at_Tm = thermo.delta_G(T_m)
        check1 = abs(delta_G_at_Tm) < 1e-6
        st.markdown(f"**1. $\\Delta G(T_m) = 0$:** {'✓ PASS' if check1 else '✗ FAIL'}")
        st.markdown(f"   計算値: {delta_G_at_Tm:.2e} J/mol")
        
        T_test = np.linspace(T_g, T_m, 10)
        delta_G_test = thermo.delta_G(T_test)
        monotonic = np.all(np.diff(delta_G_test) <= 0)
        st.markdown(f"**2. 単調増加:** {'✓ PASS' if monotonic else '✗ FAIL'}")
        
        delta_S_f = thermo.get_entropy_of_fusion()
        ratio, classification = thermo.richard_rule_check()
        st.markdown(f"**3. 融解エントロピー:**")
        st.markdown(f"   $\\Delta S_f$ = {delta_S_f:.2f} J/(mol·K)")
        st.markdown(f"   $\\Delta S_f / R$ = {ratio:.2f}")
        st.markdown(f"   {classification}")
    
    st.subheader("計算データ")
    result = thermo.calculate_all(n_points=20)
    df_thermo = pd.DataFrame({
        'Temperature [K]': result.temperature,
        'ΔG [J/mol]': result.delta_G,
        'ΔS [J/(mol·K)]': result.delta_S,
        'ΔG/(RT)': result.driving_force_normalized
    })
    st.dataframe(df_thermo.style.format({
        'Temperature [K]': '{:.1f}',
        'ΔG [J/mol]': '{:.1f}',
        'ΔS [J/(mol·K)]': '{:.2f}',
        'ΔG/(RT)': '{:.4f}'
    }), use_container_width=True)

with tab2:
    st.header("ステップ2: 粘度モデルのチェック（Doolittle式の検証）")
    
    st.markdown("""
    ### 指示
    算出された粘度 $\\eta(T)$ を縦軸（ログスケール）、温度 $T$ を横軸にしてプロットしなさい。
    
    ### 合格基準
    - $T_g$ 付近で粘度が $10^{12} \\sim 10^{13}$ Pa·s 程度まで上昇しているか
    - 高温（$T_m$ 以上）で $10^{-3} \\sim 10^{0}$ Pa·s 程度の低い値になっているか
    - **Angellプロット**（$\\log \\eta$ vs $T_g/T$）を作成し、曲線が「強い液体」か「弱い液体」かの特徴を捉えているか確認せよ
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("粘度の温度依存性")
        fig2, ax2 = viz.plot_viscosity_temperature(visc, show_validation=True)
        st.pyplot(fig2)
        plt.close(fig2)
    
    with col2:
        st.subheader("Angellプロット")
        fig3, ax3 = viz.plot_angell(visc, show_references=True)
        st.pyplot(fig3)
        plt.close(fig3)
    
    st.subheader("検証結果")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_eta_Tg = visc.log_viscosity(T_g)
        check_Tg = 11 <= log_eta_Tg <= 14
        st.metric("$\\log_{10}\\eta(T_g)$", f"{log_eta_Tg:.1f}", 
                 delta="PASS" if check_Tg else "FAIL",
                 delta_color="normal" if check_Tg else "inverse")
        st.caption("目標: 12-13")
    
    with col2:
        log_eta_Tm = visc.log_viscosity(T_m)
        check_Tm = -4 <= log_eta_Tm <= 1
        st.metric("$\\log_{10}\\eta(T_m)$", f"{log_eta_Tm:.1f}",
                 delta="PASS" if check_Tm else "FAIL",
                 delta_color="normal" if check_Tm else "inverse")
        st.caption("目標: -3 to 0")
    
    with col3:
        m = visc.fragility_index_m()
        st.metric("脆弱性指数 m", f"{m:.1f}")
        st.caption(visc.classify_fragility())

with tab3:
    st.header("ステップ3: TTT曲線の「ノーズ」の特定")
    
    st.markdown("""
    ### 指示
    横軸を時間 $\\log t$、縦軸を温度 $T$ としたTTT曲線を描画しなさい。
    
    ### 合格基準
    - グラフが「C」の形（ノーズを持つ形）になっていること
    - **ノーズ温度 $T_n$ の確認**: 一般に $T_n$ は $0.7T_m \\sim 0.8T_m$ 付近に現れることが多い
    - 極端に $T_g$ に近かったり $T_m$ に近かったりする場合は、界面エネルギー $\\sigma$ の設定ミスを疑いなさい
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig4, ax4 = viz.plot_ttt_curve(model, show_nose=True)
        st.pyplot(fig4)
        plt.close(fig4)
    
    with col2:
        st.subheader("TTT曲線パラメータ")
        
        ttt = model.calculate_ttt_curve()
        
        st.metric("ノーズ温度 $T_n$", f"{ttt.nose_temperature:.1f} K")
        st.metric("ノーズ時間 $t_n$", f"{ttt.nose_time:.2e} s")
        st.metric("$T_n / T_m$", f"{ttt.nose_temperature/T_m:.3f}")
        
        pass_nose, msg_nose = model.verify_nose_position()
        if pass_nose:
            st.success(msg_nose)
        else:
            st.error(msg_nose)
        
        st.metric("臨界冷却速度 $R_c$", f"{ttt.critical_cooling_rate:.2e} K/s")
        
        st.markdown("""
        **計算式:**
        $$R_c = \\frac{T_m - T_n}{t_n}$$
        """)

with tab4:
    st.header("ステップ4: パラメータ感度解析")
    
    st.markdown("""
    ### 指示
    $\\sigma$ の値を $\\pm 10\\%$ 変化させたとき、TTT曲線のノーズの位置（時間 $t_n$）が何桁変化するか記録しなさい。
    
    ### 考察
    「$\\sigma$ がわずかに変わるだけで、非晶質の作りやすさが劇的に変わる」ことをデータで示しなさい。
    """)
    
    variation_percent = st.slider("変動幅 [%]", min_value=5, max_value=30, value=10)
    
    sigma_result = analysis.analyze_sigma_sensitivity(variation_percent=variation_percent, n_points=21)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig5, ax5 = viz.plot_sensitivity_analysis(sigma_result)
        st.pyplot(fig5)
        plt.close(fig5)
    
    with col2:
        st.subheader("感度係数")
        coeffs = analysis.calculate_sensitivity_coefficients()
        
        for param, coeff in coeffs.items():
            st.metric(f"$S_{{{param}}}$", f"{coeff:.2f}")
            st.caption(f"1%の{param}変化 → {abs(coeff):.1f}%のR_c変化")
    
    st.subheader("感度解析結果表")
    
    df_sensitivity = pd.DataFrame({
        'σ変動 [%]': sigma_result.variations,
        'σ [J/m²]': sigma_result.values,
        'R_c [K/s]': sigma_result.R_c_values,
        'log₁₀(R_c/R_c₀)': sigma_result.log_R_c_change,
        't_n [s]': sigma_result.t_nose_values,
        'T_n [K]': sigma_result.T_nose_values
    })
    
    st.dataframe(df_sensitivity.style.format({
        'σ変動 [%]': '{:+.1f}',
        'σ [J/m²]': '{:.4e}',
        'R_c [K/s]': '{:.2e}',
        'log₁₀(R_c/R_c₀)': '{:+.2f}',
        't_n [s]': '{:.2e}',
        'T_n [K]': '{:.1f}'
    }), use_container_width=True)
    
    st.markdown("""
    ### 重要な観察
    界面エネルギー $\\sigma$ は核生成障壁 $\\Delta G^*$ に3乗で効くため（$\\Delta G^* \\propto \\sigma^3$）、
    わずかな $\\sigma$ の変化が臨界冷却速度 $R_c$ に数桁の変化をもたらします。
    
    これは、非晶質形成能の予測において $\\sigma$ の正確な値を知ることが極めて重要であることを示しています。
    """)

with tab5:
    st.header("最終課題: 実測値との比較")
    
    st.markdown("""
    ### 指示
    特定の合金系について計算を行い、文献に記載されている臨界冷却速度 $R_c$ と比較しなさい。
    
    ### 評価基準
    - 計算値と実験値の桁が合っているか？（例：実験が $10^2$ K/s なら、計算が $10^1 \\sim 10^3$ の範囲にあれば概ね良好）
    - ズレが大きい場合、Doolittle式のパラメータ $B$ または $\\sigma$ のどちらに原因があるか推論せよ
    """)
    
    R_c_calc = model.critical_cooling_rate()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("比較結果")
        
        log_diff = np.log10(R_c_calc / R_c_exp)
        
        st.metric("計算値 $R_c$", f"{R_c_calc:.2e} K/s")
        st.metric("実験値 $R_c$", f"{R_c_exp:.2e} K/s")
        st.metric("比率 (計算/実験)", f"{R_c_calc/R_c_exp:.2f}")
        st.metric("桁の差", f"{log_diff:+.2f}")
        
        if abs(log_diff) <= 1:
            st.success("良好な一致（1桁以内）")
        elif abs(log_diff) <= 2:
            st.warning("許容範囲の一致（2桁以内）")
        else:
            st.error("不十分な一致（2桁以上のずれ）")
    
    with col2:
        st.subheader("材料データベース比較")
        
        comparison_data = []
        for mat_name in db.list_materials():
            mat = db.get_material(mat_name)
            if mat.R_c_exp is not None:
                try:
                    mat_model = DavisUhlmannModel(
                        T_m=mat.T_m, T_g=mat.T_g, delta_H_f=mat.delta_H_f,
                        sigma=mat.sigma, V_m=mat.V_m, eta_0=mat.eta_0,
                        D_star=mat.D_star, T_0=mat.T_0
                    )
                    R_c_mat = mat_model.critical_cooling_rate()
                    comparison_data.append({
                        '材料': mat.composition,
                        'R_c (計算)': R_c_mat,
                        'R_c (実験)': mat.R_c_exp,
                        'log差': np.log10(R_c_mat / mat.R_c_exp)
                    })
                except Exception:
                    pass
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison.style.format({
                'R_c (計算)': '{:.2e}',
                'R_c (実験)': '{:.2e}',
                'log差': '{:+.2f}'
            }), use_container_width=True)
    
    st.subheader("ズレの原因分析")
    
    if abs(log_diff) > 1:
        st.markdown("""
        計算値と実験値に大きなズレがある場合、以下の原因が考えられます：
        
        **計算値が過大評価の場合（$R_c^{calc} > R_c^{exp}$）:**
        - $\\sigma$ が小さすぎる可能性 → $\\sigma$ を増加させると $R_c$ は減少
        - $D^*$ が大きすぎる可能性 → $D^*$ を減少させると $R_c$ は減少
        
        **計算値が過小評価の場合（$R_c^{calc} < R_c^{exp}$）:**
        - $\\sigma$ が大きすぎる可能性 → $\\sigma$ を減少させると $R_c$ は増加
        - $D^*$ が小さすぎる可能性 → $D^*$ を増加させると $R_c$ は増加
        """)

with tab6:
    st.header("📚 理論背景")
    
    st.markdown("""
    ## 1. なぜ液体を急冷するとガラスになるのか？
    
    液体を冷却すると、通常は結晶化が起こります。しかし、十分に速く冷却すると、
    結晶核が成長する時間がないまま液体が凍結し、**ガラス（非晶質）**が形成されます。
    
    この「十分に速い」冷却速度を**臨界冷却速度 $R_c$** と呼びます。
    
    ## 2. CALPHAD熱力学
    
    結晶化の**駆動力**は、液体と固体のギブスエネルギー差です：
    
    $$\\Delta G_m = G^L - G^S$$
    
    **Thompson-Spaepen近似:**
    $$\\Delta G_m = \\Delta H_f \\cdot \\frac{T_m - T}{T_m} \\cdot \\frac{2T}{T_m + T}$$
    
    - $T = T_m$ で $\\Delta G_m = 0$（平衡状態）
    - $T < T_m$ で $\\Delta G_m > 0$（結晶化の駆動力）
    
    ## 3. Doolittle粘度モデル
    
    液体の粘度は温度とともに急激に変化します。**VFT方程式**：
    
    $$\\eta(T) = \\eta_0 \\cdot \\exp\\left(\\frac{D^* \\cdot T_0}{T - T_0}\\right)$$
    
    - $T_g$ で $\\eta \\approx 10^{12}$ Pa·s（ガラス転移の定義）
    - $T_m$ で $\\eta \\approx 10^{-3}$ Pa·s（液体状態）
    
    **脆弱性指数 $m$** は液体の「強さ」を表します：
    - 強い液体（SiO₂）: $m \\approx 20$
    - 弱い液体（金属ガラス）: $m \\approx 50-100$
    
    ## 4. Davis-Uhlmannモデル
    
    TTT曲線は**核生成速度 $I$** と**成長速度 $U$** から計算されます：
    
    **核生成速度:**
    $$I = I_0 \\cdot \\exp\\left(-\\frac{\\Delta G^*}{k_B T}\\right)$$
    
    **核生成障壁:**
    $$\\Delta G^* = \\frac{16\\pi\\sigma^3}{3(\\Delta G_v)^2}$$
    
    **成長速度（Wilson-Frenkel）:**
    $$U = \\frac{D}{a} \\cdot \\left[1 - \\exp\\left(-\\frac{\\Delta G_m}{RT}\\right)\\right]$$
    
    **JMAK動力学から結晶化時間:**
    $$t = \\left[\\frac{3 \\ln(1/(1-X))}{\\pi I U^3}\\right]^{1/4}$$
    
    ## 5. 臨界冷却速度
    
    TTT曲線の「ノーズ」（最短結晶化時間）から：
    
    $$R_c = \\frac{T_m - T_n}{t_n}$$
    
    ## 6. 単位に関する注意
    
    計算がうまくいかない場合、以下をチェック：
    - $R$（ガス定数）vs $k_B$（ボルツマン定数）
    - エネルギー: J/mol vs J/atom
    - モル体積 $V_m$ の単位
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 参考文献
- Turnbull, D. (1969). Contemp. Phys.
- Uhlmann, D.R. (1972). J. Non-Cryst. Solids
- Angell, C.A. (1995). Science
- Inoue, A. (2000). Acta Mater.
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**学生へのアドバイス**

計算がうまくいかない時の多くは**単位のミスマッチ**です。
- $R$ vs $k_B$？
- J/mol vs J/atom？
- $V_m$ をかけ忘れていないか？
""")
