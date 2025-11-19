import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.linear_model import Lasso
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class BlackBoxProblem:
    def __init__(self, name="Rosenbrock"):
        self.name = name
        if name == "Rosenbrock":
            self.n_dim = 10
            self.eff_dim = [0, 1]
            self.bounds = np.array([[-2.0, 2.0]] * self.n_dim)
        else:
            self.n_dim = 20
            self.eff_dim = [0, 1, 2, 3, 4, 5]
            self.bounds = np.array([[0.0, 1.0]] * self.n_dim)

    def evaluate(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_eff = X[:, self.eff_dim]

        if self.name == "Rosenbrock":
            x = X_eff[:, 0]
            y = X_eff[:, 1]
            val = (1 - x)**2 + 100 * (y - x**2)**2
            return -val
        else:
            alpha = np.array([[1.0, 1.2, 3.0, 3.2],
                            [1.0, 1.2, 3.0, 3.2],
                            [1.0, 1.2, 3.0, 3.2],
                            [1.0, 1.2, 3.0, 3.2]])
            A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                         [0.05, 10, 17, 0.1, 8, 14],
                         [3, 3.5, 1.7, 10, 17, 8],
                         [17, 8, 0.05, 10, 0.1, 14]])
            P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])

            result = np.zeros(X_eff.shape[0])
            for i in range(X_eff.shape[0]):
                outer = 0
                for j in range(4):
                    inner = 0
                    for k in range(6):
                        inner += A[j, k] * (X_eff[i, k] - P[j, k])**2
                    outer += alpha[j, 0] * np.exp(-inner)
                result[i] = -outer
            return result


def expected_improvement(X, gp, y_max, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)

    with np.errstate(divide='warn', invalid='warn'):
        imp = mu - y_max - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0

    return ei


def propose_location(acquisition, gp, y_max, bounds, n_restarts=25):
    dim = bounds.shape[0]
    min_val = float('inf')
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(1, -1), gp, y_max)

    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)
        res = minimize(min_obj, x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x


def plot_relevance_comparison(X_train, y_train, gp_model, problem):
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_train, y_train)
    linear_importance = np.abs(lasso.coef_)

    length_scales = gp_model.kernel_.k2.length_scale
    ard_importance = 1.0 / np.array(length_scales)
    ard_importance = ard_importance / np.max(ard_importance)
    linear_importance = linear_importance / (np.max(linear_importance) + 1e-10)

    dim_labels = [f"x{i}" for i in range(problem.n_dim)]
    colors_linear = ['red' if i in problem.eff_dim else 'lightgray' for i in range(problem.n_dim)]
    colors_ard = ['blue' if i in problem.eff_dim else 'lightgray' for i in range(problem.n_dim)]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Linear Perspective: Lasso Coefficients (|β|)',
                       'Non-linear Perspective: GP-ARD Relevance (1/ℓ)'),
        vertical_spacing=0.15
    )

    fig.add_trace(
        go.Bar(x=dim_labels, y=linear_importance,
               marker_color=colors_linear,
               name='Lasso',
               hovertemplate='Dimension: %{x}<br>Importance: %{y:.3f}<extra></extra>'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=dim_labels, y=ard_importance,
               marker_color=colors_ard,
               name='GP-ARD',
               hovertemplate='Dimension: %{x}<br>Relevance: %{y:.3f}<extra></extra>'),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Dimension", row=2, col=1)
    fig.update_yaxes(title_text="Normalized Importance", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Relevance", row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Variable Importance: Linear vs Non-linear Models"
    )

    return fig


def plot_optimization_trace(history, method_name):
    iterations = list(range(len(history)))
    best_so_far = np.maximum.accumulate(history)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iterations,
        y=best_so_far,
        mode='lines+markers',
        name=method_name,
        line=dict(width=2),
        hovertemplate='Iteration: %{x}<br>Best f(x): %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Optimization Progress: {method_name}",
        xaxis_title="Iteration",
        yaxis_title="Best f(x) Found",
        hovermode='closest',
        height=400
    )

    return fig


def plot_2d_exploration(X_train, y_train, problem, iteration):
    if problem.name != "Rosenbrock":
        return None

    x0_vals = X_train[:, 0]
    x1_vals = X_train[:, 1]

    x0_grid = np.linspace(problem.bounds[0, 0], problem.bounds[0, 1], 100)
    x1_grid = np.linspace(problem.bounds[1, 0], problem.bounds[1, 1], 100)
    X0, X1 = np.meshgrid(x0_grid, x1_grid)

    Z = np.zeros_like(X0)
    for i in range(X0.shape[0]):
        for j in range(X0.shape[1]):
            x_test = np.zeros(problem.n_dim)
            x_test[0] = X0[i, j]
            x_test[1] = X1[i, j]
            Z[i, j] = problem.evaluate(x_test.reshape(1, -1))[0]

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=x0_grid,
        y=x1_grid,
        z=Z,
        colorscale='Viridis',
        contours=dict(
            coloring='heatmap',
            showlabels=True
        ),
        name='Objective Function',
        hovertemplate='x0: %{x:.3f}<br>x1: %{y:.3f}<br>f(x): %{z:.3f}<extra></extra>'
    ))

    colors = np.linspace(0, 1, len(x0_vals))

    fig.add_trace(go.Scatter(
        x=x0_vals,
        y=x1_vals,
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Iteration"),
            line=dict(width=1, color='white')
        ),
        name='Explored Points',
        hovertemplate='x0: %{x:.3f}<br>x1: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"2D Exploration Space (Effective Dimensions x0, x1) - Iteration {iteration}",
        xaxis_title="x0",
        yaxis_title="x1",
        height=500,
        hovermode='closest'
    )

    return fig


st.set_page_config(page_title="Sparse Bayesian Optimization", layout="wide")

st.title("🎯 Non-linear Sparse Bayesian Optimization")
st.markdown("""

このアプリケーションでは、**非線形関数の最適化**において、ARD (Automatic Relevance Determination) が
どのように「不要な次元」を無視して効率的に探索するかを体験できます。
""")

with st.sidebar:
    st.header("⚙️ Settings")

    prob_name = st.selectbox(
        "Target Function",
        ["Rosenbrock", "Hartmann"],
        help="Rosenbrock: 10次元空間に埋め込まれた2次元関数\nHartmann: 20次元空間に埋め込まれた6次元関数"
    )

    method = st.selectbox(
        "Optimization Method",
        ["Random Search", "Standard BO", "Sparse BO (ARD)"],
        help="Random: ランダムサンプリング\nStandard BO: 通常のベイズ最適化\nSparse BO: ARDカーネルを使用"
    )

    n_initial = st.slider("Initial Samples", 5, 20, 10,
                         help="初期ランダムサンプル数")
    n_iterations = st.slider("Optimization Iterations", 10, 50, 20,
                            help="最適化ループの回数")

    run_button = st.button("🚀 Run Optimization", type="primary")

problem = BlackBoxProblem(prob_name)

st.info(f"""
**現在の設定:**
- 関数: {prob_name}
- 全次元数: {problem.n_dim}
- 有効次元数: {len(problem.eff_dim)} (Index: {problem.eff_dim})
- 手法: {method}

**目的:** Lasso(線形)とGP-ARD(非線形)が、この「有効次元」をどう見つけるか観察してください。
""")

if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.gp_model = None
    st.session_state.history = None

if run_button:
    st.session_state.optimization_done = False

    progress_bar = st.progress(0)
    status_text = st.empty()

    np.random.seed(42)
    X_train = np.random.uniform(
        problem.bounds[:, 0],
        problem.bounds[:, 1],
        size=(n_initial, problem.n_dim)
    )
    y_train = problem.evaluate(X_train)

    history = [np.max(y_train)]

    status_text.text(f"初期サンプリング完了: {n_initial}点")
    progress_bar.progress(0)

    if method == "Random Search":
        for i in range(n_iterations):
            x_new = np.random.uniform(
                problem.bounds[:, 0],
                problem.bounds[:, 1],
                size=problem.n_dim
            )
            y_new = problem.evaluate(x_new.reshape(1, -1))[0]

            X_train = np.vstack([X_train, x_new])
            y_train = np.append(y_train, y_new)
            history.append(np.max(y_train))

            progress_bar.progress((i + 1) / n_iterations)
            status_text.text(f"Iteration {i+1}/{n_iterations}: Best f(x) = {np.max(y_train):.4f}")

        gp_model = None

    else:
        if method == "Sparse BO (ARD)":
            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                     Matern(length_scale=[1.0] * problem.n_dim,
                           length_scale_bounds=(1e-2, 1e2),
                           nu=2.5)
        else:
            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                     Matern(length_scale=1.0,
                           length_scale_bounds=(1e-2, 1e2),
                           nu=2.5)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )

        for i in range(n_iterations):
            gp.fit(X_train, y_train)

            x_new = propose_location(
                expected_improvement,
                gp,
                np.max(y_train),
                problem.bounds,
                n_restarts=25
            )

            y_new = problem.evaluate(x_new.reshape(1, -1))[0]

            X_train = np.vstack([X_train, x_new])
            y_train = np.append(y_train, y_new)
            history.append(np.max(y_train))

            progress_bar.progress((i + 1) / n_iterations)
            status_text.text(f"Iteration {i+1}/{n_iterations}: Best f(x) = {np.max(y_train):.4f}")

        gp_model = gp

    st.session_state.optimization_done = True
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.gp_model = gp_model
    st.session_state.history = history
    st.session_state.method = method
    st.session_state.problem = problem

    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ 最適化完了！ 最良値: {np.max(y_train):.4f}")

if st.session_state.optimization_done:
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    gp_model = st.session_state.gp_model
    history = st.session_state.history
    method = st.session_state.method
    problem = st.session_state.problem

    st.markdown("---")
    st.header("📊 Analysis Results")

    tab1, tab2, tab3 = st.tabs([
        "1️⃣ Mental Model Comparison",
        "2️⃣ Optimization Trace",
        "3️⃣ Exploration Space"
    ])

    with tab1:
        st.markdown("""

        **上段 (Lasso):** 線形回帰の係数。線形で近似できる範囲での重要度を示します。

        **下段 (GP-ARD):** ガウス過程のARDカーネルによる関連度。非線形な相互作用を含めた真の重要度を示します。

        **重要:** 非線形関数（例: y=x²）の場合、Lassoの係数は0になりがちですが、ARDは反応します。
        """)

        if gp_model is not None:
            fig_relevance = plot_relevance_comparison(X_train, y_train, gp_model, problem)
            st.plotly_chart(fig_relevance, use_container_width=True)

            st.markdown("#### 🔍 詳細情報")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**有効次元 (赤/青):**")
                st.write(f"インデックス: {problem.eff_dim}")

            with col2:
                st.markdown("**無効次元 (グレー):**")
                ineffective = [i for i in range(problem.n_dim) if i not in problem.eff_dim]
                st.write(f"インデックス: {ineffective}")

            if method == "Sparse BO (ARD)":
                length_scales = gp_model.kernel_.k2.length_scale
                st.markdown("**ARD Length Scales:**")
                ls_df_data = {
                    "Dimension": [f"x{i}" for i in range(problem.n_dim)],
                    "Length Scale": [f"{ls:.4f}" for ls in length_scales],
                    "Relevance (1/ℓ)": [f"{1/ls:.4f}" for ls in length_scales],
                    "Type": ["Effective" if i in problem.eff_dim else "Ineffective"
                            for i in range(problem.n_dim)]
                }
                st.dataframe(ls_df_data, use_container_width=True)
        else:
            st.warning("Random Searchでは変数重要度の推定は行われません。")

    with tab2:
        st.markdown("""

        横軸にイテレーション、縦軸に「現在までの最良値」を表示します。
        ARDがいかに早く最適解に近づくかを確認してください。
        """)

        fig_trace = plot_optimization_trace(history, method)
        st.plotly_chart(fig_trace, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial Best", f"{history[0]:.4f}")
        with col2:
            st.metric("Final Best", f"{history[-1]:.4f}")
        with col3:
            improvement = history[-1] - history[0]
            st.metric("Improvement", f"{improvement:.4f}")

    with tab3:
        st.markdown("""

        有効な2変数の空間（x₀, x₁）における等高線図と探索点を表示します。
        色が濃い点ほど新しい探索点です。
        """)

        if problem.name == "Rosenbrock":
            fig_2d = plot_2d_exploration(X_train, y_train, problem, len(history))
            st.plotly_chart(fig_2d, use_container_width=True)

            st.markdown("#### 📈 探索の集中度")
            x0_std = np.std(X_train[:, 0])
            x1_std = np.std(X_train[:, 1])
            other_dims_std = np.mean([np.std(X_train[:, i])
                                     for i in range(2, problem.n_dim)])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("x₀ Std Dev", f"{x0_std:.4f}")
            with col2:
                st.metric("x₁ Std Dev", f"{x1_std:.4f}")
            with col3:
                st.metric("Other Dims Avg Std", f"{other_dims_std:.4f}")

            if method == "Sparse BO (ARD)":
                st.info("""
                **期待される結果:** ARDは有効次元（x₀, x₁）を集中的に探索するため、
                これらの次元の標準偏差が大きくなり、無効次元の標準偏差は小さくなります。
                """)
        else:
            st.info("Hartmann関数は6次元なため、2D可視化は利用できません。")

st.markdown("---")
st.markdown("""

1. **非線形性の理解:** Rosenbrock関数のような曲がった谷を持つ関数において、
   Lassoの係数が低くても、GPのARD重要度が高くなるケースを確認してください。

2. **ARDの収束:** データ点数が少ないうち（初期）はARDの重要度がランダムですが、
   点数が増えるにつれて「正解の有効次元」の重要度が突出してきます。

3. **Sparse BOの威力:** 無効な次元（ダミー変数）が多数存在しても、
   ARDが有効次元を見抜いて最適化が進むことを確認してください。

4. **なぜARDはSparsityを実現できるのか？**
   長さスケール ℓ → ∞ になると、カーネル関数の値が定数になり、
   その次元の変化が出力に寄与しなくなるためです。
""")
