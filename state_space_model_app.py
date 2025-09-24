"""
状態空間モデル・データ同化システム (State Space Model & Data Assimilation System)
カルマンフィルタ、パーティクルフィルタ、アンサンブルカルマンフィルタによるデータ同化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.linalg import solve_discrete_are
from scipy.stats import multivariate_normal, norm
from typing import Tuple, List, Dict, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['IPAGothic', 'IPAPGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class KalmanFilter:
    """カルマンフィルタによるデータ同化"""
    
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray):
        """
        Parameters:
        F: 状態遷移行列 (n_states x n_states)
        H: 観測行列 (n_obs x n_states)
        Q: システムノイズ共分散行列 (n_states x n_states)
        R: 観測ノイズ共分散行列 (n_obs x n_obs)
        x0: 初期状態 (n_states,)
        P0: 初期共分散行列 (n_states x n_states)
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.P = P0.copy()
        
        self.n_states = F.shape[0]
        self.n_obs = H.shape[0]
        
        self.states_history = []
        self.covariances_history = []
        self.predictions_history = []
        self.innovations_history = []
        self.log_likelihood = 0.0
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """予測ステップ（フォーキャスト）"""
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        return x_pred, P_pred
    
    def update(self, z: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray) -> None:
        """更新ステップ（アナリシス）"""
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ P_pred
        
        det_S = np.linalg.det(S)
        if det_S > 0:
            self.log_likelihood += -0.5 * (np.log(2 * np.pi * det_S) + y.T @ np.linalg.inv(S) @ y)
        
        self.innovations_history.append(y.copy())
    
    def assimilate(self, observations: np.ndarray) -> Dict:
        """データ同化実行"""
        self.states_history = []
        self.covariances_history = []
        self.predictions_history = []
        self.innovations_history = []
        self.log_likelihood = 0.0
        
        n_timesteps = observations.shape[0]
        
        for t in range(n_timesteps):
            x_pred, P_pred = self.predict()
            self.predictions_history.append(x_pred.copy())
            
            z = observations[t].reshape(-1, 1) if observations[t].ndim == 0 else observations[t].reshape(-1, 1)
            self.update(z, x_pred, P_pred)
            
            self.states_history.append(self.x.copy())
            self.covariances_history.append(self.P.copy())
        
        return {
            'states': np.array(self.states_history),
            'covariances': np.array(self.covariances_history),
            'predictions': np.array(self.predictions_history),
            'innovations': np.array(self.innovations_history),
            'log_likelihood': self.log_likelihood
        }

class ParticleFilter:
    """パーティクルフィルタによるデータ同化"""
    
    def __init__(self, n_particles: int, state_dim: int, obs_dim: int,
                 transition_func: Callable, observation_func: Callable,
                 process_noise_func: Callable, obs_noise_std: float):
        """
        Parameters:
        n_particles: パーティクル数
        state_dim: 状態次元
        obs_dim: 観測次元
        transition_func: 状態遷移関数
        observation_func: 観測関数
        process_noise_func: プロセスノイズ生成関数
        obs_noise_std: 観測ノイズ標準偏差
        """
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.transition_func = transition_func
        self.observation_func = observation_func
        self.process_noise_func = process_noise_func
        self.obs_noise_std = obs_noise_std
        
        self.particles = np.random.randn(n_particles, state_dim)
        self.weights = np.ones(n_particles) / n_particles
        
        self.states_history = []
        self.particles_history = []
        self.weights_history = []
    
    def predict(self) -> None:
        """予測ステップ"""
        for i in range(self.n_particles):
            noise = self.process_noise_func()
            self.particles[i] = self.transition_func(self.particles[i]) + noise
    
    def update(self, observation: np.ndarray) -> None:
        """更新ステップ（重み計算）"""
        for i in range(self.n_particles):
            predicted_obs = self.observation_func(self.particles[i])
            if self.obs_dim == 1:
                likelihood = norm.pdf(
                    observation.flatten()[0], 
                    loc=predicted_obs.flatten()[0], 
                    scale=self.obs_noise_std
                )
            else:
                cov_matrix = np.eye(self.obs_dim) * (self.obs_noise_std**2)
                mv_normal = multivariate_normal(mean=predicted_obs.flatten(), cov=cov_matrix)
                likelihood = mv_normal.pdf(observation.flatten())
            self.weights[i] *= likelihood
        
        self.weights /= np.sum(self.weights)
    
    def resample(self) -> None:
        """リサンプリング"""
        n_eff = 1.0 / np.sum(self.weights**2)
        
        if n_eff < self.n_particles / 2:
            indices = np.zeros(self.n_particles, dtype=int)
            cumsum = np.cumsum(self.weights)
            u = np.random.rand() / self.n_particles
            
            for i in range(self.n_particles):
                u_i = u + i / self.n_particles
                indices[i] = np.searchsorted(cumsum, u_i)
            
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def assimilate(self, observations: np.ndarray) -> Dict:
        """パーティクルフィルタによるデータ同化"""
        self.states_history = []
        self.particles_history = []
        self.weights_history = []
        
        n_timesteps = observations.shape[0]
        
        for t in range(n_timesteps):
            self.predict()
            
            obs = observations[t]
            self.update(obs)
            
            self.resample()
            
            state_estimate = np.average(self.particles, weights=self.weights, axis=0)
            
            self.states_history.append(state_estimate.copy())
            self.particles_history.append(self.particles.copy())
            self.weights_history.append(self.weights.copy())
        
        return {
            'states': np.array(self.states_history),
            'particles': self.particles_history,
            'weights': self.weights_history
        }

class EnsembleKalmanFilter:
    """アンサンブルカルマンフィルタによるデータ同化"""
    
    def __init__(self, n_ensemble: int, state_dim: int, obs_dim: int,
                 transition_func: Callable, observation_func: Callable,
                 process_noise_std: float, obs_noise_std: float):
        """
        Parameters:
        n_ensemble: アンサンブルサイズ
        state_dim: 状態次元
        obs_dim: 観測次元
        transition_func: 状態遷移関数
        observation_func: 観測関数
        process_noise_std: プロセスノイズ標準偏差
        obs_noise_std: 観測ノイズ標準偏差
        """
        self.n_ensemble = n_ensemble
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.transition_func = transition_func
        self.observation_func = observation_func
        self.process_noise_std = process_noise_std
        self.obs_noise_std = obs_noise_std
        
        self.ensemble = np.random.randn(n_ensemble, state_dim)
        
        self.states_history = []
        self.ensemble_history = []
        self.spread_history = []
    
    def predict(self) -> None:
        """予測ステップ"""
        for i in range(self.n_ensemble):
            noise = np.random.randn(self.state_dim) * self.process_noise_std
            self.ensemble[i] = self.transition_func(self.ensemble[i]) + noise
    
    def update(self, observation: np.ndarray) -> None:
        """更新ステップ（アンサンブル変換）"""
        x_mean = np.mean(self.ensemble, axis=0)
        
        X = self.ensemble - x_mean
        
        H_ensemble = np.array([self.observation_func(member) for member in self.ensemble])
        h_mean = np.mean(H_ensemble, axis=0)
        
        Y = H_ensemble - h_mean
        
        P_xy = X.T @ Y / (self.n_ensemble - 1)
        P_yy = Y.T @ Y / (self.n_ensemble - 1) + np.eye(self.obs_dim) * self.obs_noise_std**2
        
        K = P_xy @ np.linalg.inv(P_yy)
        
        for i in range(self.n_ensemble):
            obs_noise = np.random.randn(self.obs_dim) * self.obs_noise_std
            innovation = observation + obs_noise - H_ensemble[i]
            self.ensemble[i] = self.ensemble[i] + K @ innovation
    
    def assimilate(self, observations: np.ndarray) -> Dict:
        """EnKFによるデータ同化"""
        self.states_history = []
        self.ensemble_history = []
        self.spread_history = []
        
        n_timesteps = observations.shape[0]
        
        for t in range(n_timesteps):
            self.predict()
            
            obs = observations[t]
            self.update(obs)
            
            state_estimate = np.mean(self.ensemble, axis=0)
            ensemble_spread = np.std(self.ensemble, axis=0)
            
            self.states_history.append(state_estimate.copy())
            self.ensemble_history.append(self.ensemble.copy())
            self.spread_history.append(ensemble_spread.copy())
        
        return {
            'states': np.array(self.states_history),
            'ensemble': self.ensemble_history,
            'spread': np.array(self.spread_history)
        }

def generate_lorenz_data(n_points: int = 1000, dt: float = 0.01, 
                        sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0,
                        obs_noise_std: float = 0.5, obs_interval: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ローレンツ方程式による真値・観測データ生成"""
    
    def lorenz_rhs(state):
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
    
    true_states = np.zeros((n_points, 3))
    true_states[0] = [1.0, 1.0, 1.0]  # 初期条件
    
    for i in range(1, n_points):
        k1 = lorenz_rhs(true_states[i-1])
        k2 = lorenz_rhs(true_states[i-1] + dt/2 * k1)
        k3 = lorenz_rhs(true_states[i-1] + dt/2 * k2)
        k4 = lorenz_rhs(true_states[i-1] + dt * k3)
        
        true_states[i] = true_states[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    obs_times = np.arange(0, n_points, obs_interval)
    observations = true_states[obs_times] + np.random.randn(len(obs_times), 3) * obs_noise_std
    
    time_grid = np.arange(n_points) * dt
    
    return time_grid, true_states, observations

def generate_simple_model_data(model_type: str, n_points: int = 100, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """シンプルな状態空間モデルのデータ生成"""
    t = np.arange(n_points)
    
    if model_type == 'random_walk':
        process_noise = kwargs.get('process_noise', 0.1)
        obs_noise = kwargs.get('obs_noise', 0.2)
        
        true_state = np.cumsum(np.random.randn(n_points) * process_noise)
        observations = true_state + np.random.randn(n_points) * obs_noise
        
    elif model_type == 'ar1':
        phi = kwargs.get('phi', 0.8)
        process_noise = kwargs.get('process_noise', 0.3)
        obs_noise = kwargs.get('obs_noise', 0.2)
        
        true_state = np.zeros(n_points)
        true_state[0] = np.random.randn()
        
        for i in range(1, n_points):
            true_state[i] = phi * true_state[i-1] + np.random.randn() * process_noise
        
        observations = true_state + np.random.randn(n_points) * obs_noise
        
    elif model_type == 'local_level_trend':
        level_noise = kwargs.get('level_noise', 0.1)
        trend_noise = kwargs.get('trend_noise', 0.05)
        obs_noise = kwargs.get('obs_noise', 0.2)
        
        level = np.zeros(n_points)
        trend = np.zeros(n_points)
        level[0] = 0.0
        trend[0] = 0.1
        
        for i in range(1, n_points):
            level[i] = level[i-1] + trend[i-1] + np.random.randn() * level_noise
            trend[i] = trend[i-1] + np.random.randn() * trend_noise
        
        true_state = level
        observations = true_state + np.random.randn(n_points) * obs_noise
    
    return np.array(t), np.array(true_state), np.array(observations)

def create_data_assimilation_app():
    """データ同化システムStreamlitアプリケーション"""
    
    st.set_page_config(
        page_title="状態空間モデル・データ同化システム",
        page_icon="🌊",
        layout="wide"
    )
    
    st.title("🌊 状態空間モデル・データ同化システム")
    st.markdown("**カルマンフィルタ・パーティクルフィルタ・アンサンブルカルマンフィルタによるデータ同化**")
    
    st.sidebar.header("⚙️ 設定")
    
    data_type = st.sidebar.selectbox(
        "データタイプ",
        ["ローレンツ方程式", "ランダムウォーク", "AR(1)モデル", "ローカルレベル+トレンド", "CSVアップロード"]
    )
    
    assimilation_method = st.sidebar.selectbox(
        "データ同化手法",
        ["カルマンフィルタ", "パーティクルフィルタ", "アンサンブルカルマンフィルタ", "手法比較"]
    )
    
    if data_type == "ローレンツ方程式":
        st.sidebar.subheader("ローレンツ方程式パラメータ")
        n_points = st.sidebar.slider("データ点数", 500, 2000, 1000)
        sigma = st.sidebar.slider("σ (Prandtl数)", 5.0, 15.0, 10.0)
        rho = st.sidebar.slider("ρ (Rayleigh数)", 20.0, 35.0, 28.0)
        beta = st.sidebar.slider("β", 1.0, 4.0, 8.0/3.0)
        obs_noise_std = st.sidebar.slider("観測ノイズ", 0.1, 2.0, 0.5)
        obs_interval = st.sidebar.slider("観測間隔", 5, 20, 10)
        
        time_grid, true_states, observations = generate_lorenz_data(
            n_points, obs_noise_std=obs_noise_std, obs_interval=obs_interval,
            sigma=sigma, rho=rho, beta=beta
        )
        
    elif data_type == "CSVアップロード":
        uploaded_file = st.sidebar.file_uploader("CSVファイル", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.write("データプレビュー:")
            st.sidebar.write(df.head())
            
            time_col = st.sidebar.selectbox("時間列", df.columns)
            value_col = st.sidebar.selectbox("値列", df.columns)
            
            time_grid = df[time_col].values
            observations = np.array(df[value_col]).reshape(-1, 1)
            true_states = None
        else:
            st.warning("CSVファイルをアップロードしてください")
            return
            
    else:
        st.sidebar.subheader("モデルパラメータ")
        n_points = st.sidebar.slider("データ点数", 50, 500, 100)
        
        if data_type == "ランダムウォーク":
            process_noise = st.sidebar.slider("プロセスノイズ", 0.05, 0.5, 0.1)
            obs_noise = st.sidebar.slider("観測ノイズ", 0.05, 0.5, 0.2)
            
            time_grid, true_states, observations = generate_simple_model_data(
                'random_walk', n_points, process_noise=process_noise, obs_noise=obs_noise
            )
            
        elif data_type == "AR(1)モデル":
            phi = st.sidebar.slider("AR係数", -0.99, 0.99, 0.8)
            process_noise = st.sidebar.slider("プロセスノイズ", 0.1, 0.8, 0.3)
            obs_noise = st.sidebar.slider("観測ノイズ", 0.05, 0.5, 0.2)
            
            time_grid, true_states, observations = generate_simple_model_data(
                'ar1', n_points, phi=phi, process_noise=process_noise, obs_noise=obs_noise
            )
            
        elif data_type == "ローカルレベル+トレンド":
            level_noise = st.sidebar.slider("レベルノイズ", 0.05, 0.3, 0.1)
            trend_noise = st.sidebar.slider("トレンドノイズ", 0.01, 0.2, 0.05)
            obs_noise = st.sidebar.slider("観測ノイズ", 0.05, 0.5, 0.2)
            
            time_grid, true_states, observations = generate_simple_model_data(
                'local_level_trend', n_points, 
                level_noise=level_noise, trend_noise=trend_noise, obs_noise=obs_noise
            )
    
    if assimilation_method in ["パーティクルフィルタ", "手法比較"]:
        st.sidebar.subheader("パーティクルフィルタ設定")
        n_particles = st.sidebar.slider("パーティクル数", 100, 1000, 500)
    
    if assimilation_method in ["アンサンブルカルマンフィルタ", "手法比較"]:
        st.sidebar.subheader("アンサンブルカルマンフィルタ設定")
        n_ensemble = st.sidebar.slider("アンサンブルサイズ", 20, 100, 50)
    
    if st.sidebar.button("🚀 データ同化実行", type="primary"):
        
        if data_type == "ローレンツ方程式":
            st.subheader("📊 ローレンツ方程式データ")
            
            if true_states is not None:
                fig = go.Figure(data=go.Scatter3d(
                    x=true_states[:, 0],
                    y=true_states[:, 1],
                    z=true_states[:, 2],
                    mode='lines',
                    name='真値軌道',
                    line=dict(color='blue', width=3)
                ))
                
                obs_times = np.arange(0, len(true_states), obs_interval)
                fig.add_trace(go.Scatter3d(
                    x=observations[:, 0],
                    y=observations[:, 1],
                    z=observations[:, 2],
                    mode='markers',
                    name='観測値',
                    marker=dict(color='red', size=4)
                ))
                
                fig.update_layout(
                    title="ローレンツアトラクタ",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z"
                    ),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.subheader("📊 時系列データ")
            fig = go.Figure()
            
            if true_states is not None:
                fig.add_trace(go.Scatter(
                    x=time_grid, y=true_states,
                    mode='lines',
                    name='真値',
                    line=dict(color='blue', width=2)
                ))
            
            if observations.ndim == 1:
                obs_y = observations
                obs_x = time_grid
            else:
                obs_y = observations[:, 0]
                obs_x = time_grid[::obs_interval] if data_type == "ローレンツ方程式" else time_grid
            
            fig.add_trace(go.Scatter(
                x=obs_x, y=obs_y,
                mode='markers',
                name='観測値',
                marker=dict(color='red', size=6)
            ))
            
            fig.update_layout(
                title="時系列データ",
                xaxis_title="時間",
                yaxis_title="値",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with st.spinner("データ同化を実行中..."):
            
            if assimilation_method == "カルマンフィルタ" or assimilation_method == "手法比較":
                if data_type == "ローレンツ方程式":
                    F = np.eye(3)  # 簡単のため単位行列
                    H = np.eye(3)
                    Q = np.eye(3) * 0.1
                    R = np.eye(3) * obs_noise_std**2
                    x0 = observations[0]
                    P0 = np.eye(3) * 10.0
                    
                    kf = KalmanFilter(F, H, Q, R, x0, P0)
                    kf_results = kf.assimilate(observations)
                    
                else:
                    if data_type == "AR(1)モデル":
                        F = np.array([[phi]])
                        Q = np.array([[process_noise**2]])
                    else:
                        F = np.array([[1.0]])
                        Q = np.array([[0.1]])
                    
                    H = np.array([[1.0]])
                    R = np.array([[obs_noise**2]])
                    x0 = np.array([observations[0]])
                    P0 = np.array([[1.0]])
                    
                    kf = KalmanFilter(F, H, Q, R, x0, P0)
                    kf_results = kf.assimilate(observations.reshape(-1, 1))
        
        st.subheader("🎯 データ同化結果")
        
        if assimilation_method == "カルマンフィルタ":
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**カルマンフィルタ統計**")
                st.write(f"対数尤度: {kf_results['log_likelihood']:.4f}")
                
                if data_type != "ローレンツ方程式":
                    estimated_states = kf_results['states'][:, 0]
                    if true_states is not None:
                        rmse = np.sqrt(np.mean((estimated_states - true_states)**2))
                        st.write(f"RMSE: {rmse:.4f}")
            
            with col2:
                st.write("**フィルタ性能**")
                innovations = np.array(kf_results['innovations'])
                st.write(f"イノベーション平均: {np.mean(innovations):.4f}")
                st.write(f"イノベーション標準偏差: {np.std(innovations):.4f}")
            
            if data_type == "ローレンツ方程式":
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('X成分', 'Y成分', 'Z成分')
                )
                
                if true_states is not None:
                    obs_times = np.arange(0, len(true_states), obs_interval)
                else:
                    obs_times = np.arange(len(observations))
                
                for i, component in enumerate(['X', 'Y', 'Z']):
                    if true_states is not None:
                        fig.add_trace(go.Scatter(
                            x=time_grid, y=true_states[:, i],
                            mode='lines', name=f'真値 {component}',
                            line=dict(color='blue')
                        ), row=i+1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=time_grid[obs_times], y=kf_results['states'][:, i],
                        mode='lines', name=f'推定値 {component}',
                        line=dict(color='red')
                    ), row=i+1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=time_grid[obs_times], y=observations[:, i],
                        mode='markers', name=f'観測値 {component}',
                        marker=dict(color='green', size=4)
                    ), row=i+1, col=1)
                
                fig.update_layout(height=800, title="カルマンフィルタによるデータ同化結果")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                fig = go.Figure()
                
                if true_states is not None:
                    fig.add_trace(go.Scatter(
                        x=time_grid, y=true_states,
                        mode='lines', name='真値',
                        line=dict(color='blue', width=2)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=time_grid, y=kf_results['states'][:, 0],
                    mode='lines', name='カルマンフィルタ推定値',
                    line=dict(color='red', width=2)
                ))
                
                std_dev = np.sqrt(kf_results['covariances'][:, 0, 0])
                upper_bound = kf_results['states'][:, 0] + 1.96 * std_dev
                lower_bound = kf_results['states'][:, 0] - 1.96 * std_dev
                
                time_grid_array = np.array(time_grid)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([time_grid_array, time_grid_array[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95%信頼区間'
                ))
                
                if observations.ndim == 1:
                    obs_y = observations
                    obs_x = time_grid
                else:
                    obs_y = observations[:, 0]
                    obs_x = time_grid
                
                fig.add_trace(go.Scatter(
                    x=obs_x, y=obs_y,
                    mode='markers', name='観測値',
                    marker=dict(color='green', size=6)
                ))
                
                fig.update_layout(
                    title="カルマンフィルタによるデータ同化結果",
                    xaxis_title="時間",
                    yaxis_title="値",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📖 使用方法"):
        st.markdown("""
        
        - **ローレンツ方程式**: カオス的な非線形動力学系
        - **ランダムウォーク**: 確率的な時系列モデル
        - **AR(1)モデル**: 自己回帰モデル
        - **ローカルレベル+トレンド**: 構造時系列モデル
        
        - **カルマンフィルタ**: 線形ガウシアンシステムの最適推定
        - **パーティクルフィルタ**: 非線形・非ガウシアンシステム対応
        - **アンサンブルカルマンフィルタ**: 高次元システム向け
        
        1. **予測ステップ**: モデルによる状態予測
        2. **更新ステップ**: 観測データによる状態修正
        3. **品質管理**: イノベーション統計による診断
        
        - **RMSE**: 推定精度
        - **対数尤度**: モデル適合度
        - **イノベーション統計**: フィルタ性能診断
        """)

if __name__ == "__main__":
    create_data_assimilation_app()
