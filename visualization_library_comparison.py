"""
材料工学向け可視化ライブラリ包括的比較ツール
Comprehensive Visualization Library Comparison Tool for Materials Science

このアプリケーションは、Python の主要な可視化ライブラリ（matplotlib, plotly, seaborn）を
材料科学の実例データを用いて比較し、EDA（探索的データ解析）ツールとしても機能します。

主な機能:
- 複数の可視化ライブラリの機能比較
- 材料科学データセットの探索的データ解析
- パフォーマンス計測とベンチマーク
- 材料科学特有の可視化デモ（拡散プロファイル、相図など）
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from io import StringIO
import os
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ページ設定
st.set_page_config(
    page_title="可視化ライブラリ比較ツール",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ライブラリ機能情報
LIBRARY_CAPABILITIES = {
    "matplotlib": {
        "インタラクティブ": "❌ 静的",
        "3D対応": "✅ 対応",
        "アニメーション": "✅ 対応",
        "エクスポート": "PNG, PDF, SVG",
        "特徴": "出版品質、高度なカスタマイズ",
        "適用場面": "論文図表、静的レポート"
    },
    "plotly": {
        "インタラクティブ": "✅ 完全対応",
        "3D対応": "✅ 対応",
        "アニメーション": "✅ 対応",
        "エクスポート": "HTML, PNG, PDF",
        "特徴": "インタラクティブ、Web統合",
        "適用場面": "Webダッシュボード、探索的分析"
    },
    "seaborn": {
        "インタラクティブ": "❌ 静的",
        "3D対応": "❌ 非対応",
        "アニメーション": "❌ 非対応",
        "エクスポート": "PNG, PDF, SVG",
        "特徴": "統計的可視化、美しいデフォルト",
        "適用場面": "統計分析、分布比較"
    }
}

# =============================================================================
# データ読み込みモジュール
# =============================================================================

@st.cache_data
def load_lithium_battery_data():
    """リチウム電池材料データセットの読み込み"""
    try:
        data_path = os.path.join(SCRIPT_DIR, 'lithium_battery_materials.csv')
        df = pd.read_csv(data_path)
        metadata = {
            "name": "リチウム電池材料データ",
            "description": "材料組成と物性の関係を示すデータセット",
            "rows": len(df),
            "columns": len(df.columns),
            "recommended_x": "formation_energy_per_atom",
            "recommended_y": "band_gap",
            "recommended_color": "n_elements"
        }
        return df, metadata
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None

@st.cache_data
def load_heat_conduction_data():
    """熱伝導データセットの読み込み"""
    try:
        data_path = os.path.join(SCRIPT_DIR, 'heat_conduction_single.csv')
        df = pd.read_csv(data_path)
        metadata = {
            "name": "熱伝導シミュレーションデータ",
            "description": "時間・空間における温度分布の変化",
            "rows": len(df),
            "columns": len(df.columns),
            "recommended_x": "x",
            "recommended_y": "u",
            "recommended_color": "t"
        }
        return df, metadata
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None

@st.cache_data
def load_burgers_data():
    """Burgers方程式データセットの読み込み"""
    try:
        data_path = os.path.join(SCRIPT_DIR, 'burgers_single.csv')
        df = pd.read_csv(data_path)
        metadata = {
            "name": "Burgers方程式データ",
            "description": "非線形偏微分方程式の数値解",
            "rows": len(df),
            "columns": len(df.columns),
            "recommended_x": "x",
            "recommended_y": "u",
            "recommended_color": "t"
        }
        return df, metadata
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None

@st.cache_data
def generate_synthetic_diffusion_data():
    """合成拡散データの生成"""
    nx, nt = 100, 50
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    D = 0.1
    C = np.zeros_like(X)
    for i in range(nt):
        C[i, :] = 0.5 * (1 + np.tanh((x - 0.5 - D*t[i]) / 0.05))
    
    data = []
    for i in range(nt):
        for j in range(nx):
            data.append({
                't': T[i, j],
                'x': X[i, j],
                'C': C[i, j],
                'D': D
            })
    
    df = pd.DataFrame(data)
    metadata = {
        "name": "合成拡散プロファイル",
        "description": "1次元拡散方程式の数値解（合成データ）",
        "rows": len(df),
        "columns": len(df.columns),
        "recommended_x": "x",
        "recommended_y": "C",
        "recommended_color": "t"
    }
    return df, metadata

def load_uploaded_data(uploaded_file):
    """ユーザーアップロードデータの読み込み"""
    try:
        df = pd.read_csv(uploaded_file)
        metadata = {
            "name": f"アップロードデータ: {uploaded_file.name}",
            "description": "ユーザー提供データセット",
            "rows": len(df),
            "columns": len(df.columns),
            "recommended_x": df.columns[0] if len(df.columns) > 0 else None,
            "recommended_y": df.columns[1] if len(df.columns) > 1 else None,
            "recommended_color": df.columns[2] if len(df.columns) > 2 else None
        }
        return df, metadata
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None

# =============================================================================
# 可視化関数群 - Matplotlib
# =============================================================================

def plot_scatter_matplotlib(df, x, y, color=None, title="散布図"):
    """Matplotlib散布図"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if color and color in df.columns:
        scatter = ax.scatter(df[x], df[y], c=df[color], cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, ax=ax, label=color)
    else:
        ax.scatter(df[x], df[y], alpha=0.6, s=50, color='steelblue')
    
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_line_matplotlib(df, x, y, group=None, title="線グラフ"):
    """Matplotlib線グラフ"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if group and group in df.columns:
        for name, group_df in df.groupby(group):
            ax.plot(group_df[x], group_df[y], label=f'{group}={name:.3f}', alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.plot(df[x], df[y], linewidth=2, color='steelblue')
    
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_histogram_matplotlib(df, column, bins=30, title="ヒストグラム"):
    """Matplotlibヒストグラム"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df[column].dropna(), bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('頻度', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def plot_heatmap_matplotlib(df, x, y, z, title="ヒートマップ"):
    """Matplotlibヒートマップ"""
    try:
        pivot_data = df.pivot_table(values=z, index=y, columns=x, aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot_data, aspect='auto', cmap='viridis', origin='lower')
        
        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label=z)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"ヒートマップ生成エラー: {e}")
        return None

def plot_3d_scatter_matplotlib(df, x, y, z, color=None, title="3D散布図"):
    """Matplotlib 3D散布図"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if color and color in df.columns:
        scatter = ax.scatter(df[x], df[y], df[z], c=df[color], cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, ax=ax, label=color, shrink=0.5)
    else:
        ax.scatter(df[x], df[y], df[z], alpha=0.6, s=30, color='steelblue')
    
    ax.set_xlabel(x, fontsize=10)
    ax.set_ylabel(y, fontsize=10)
    ax.set_zlabel(z, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_box_matplotlib(df, column, group=None, title="箱ひげ図"):
    """Matplotlib箱ひげ図"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if group and group in df.columns:
        data_to_plot = [group_df[column].dropna() for name, group_df in df.groupby(group)]
        labels = [str(name) for name, _ in df.groupby(group)]
        ax.boxplot(data_to_plot, labels=labels)
        ax.set_xlabel(group, fontsize=12)
    else:
        ax.boxplot([df[column].dropna()])
        ax.set_xticklabels([column])
    
    ax.set_ylabel(column, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

# =============================================================================
# 可視化関数群 - Plotly
# =============================================================================

def plot_scatter_plotly(df, x, y, color=None, title="散布図"):
    """Plotly散布図"""
    if color and color in df.columns:
        fig = px.scatter(df, x=x, y=y, color=color, title=title,
                        color_continuous_scale='viridis', opacity=0.6)
    else:
        fig = px.scatter(df, x=x, y=y, title=title, opacity=0.6)
    
    fig.update_layout(
        height=600,
        hovermode='closest',
        template='plotly_white'
    )
    return fig

def plot_line_plotly(df, x, y, group=None, title="線グラフ"):
    """Plotly線グラフ"""
    if group and group in df.columns:
        df_sorted = df.sort_values([group, x])
        fig = px.line(df_sorted, x=x, y=y, color=group, title=title)
    else:
        df_sorted = df.sort_values(x)
        fig = px.line(df_sorted, x=x, y=y, title=title)
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

def plot_histogram_plotly(df, column, bins=30, title="ヒストグラム"):
    """Plotlyヒストグラム"""
    fig = px.histogram(df, x=column, nbins=bins, title=title)
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    return fig

def plot_heatmap_plotly(df, x, y, z, title="ヒートマップ"):
    """Plotlyヒートマップ"""
    try:
        pivot_data = df.pivot_table(values=z, index=y, columns=x, aggfunc='mean')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            hovertemplate=f'{x}: %{{x}}<br>{y}: %{{y}}<br>{z}: %{{z}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            xaxis_title=x,
            yaxis_title=y,
            template='plotly_white'
        )
        return fig
    except Exception as e:
        st.error(f"ヒートマップ生成エラー: {e}")
        return None

def plot_3d_scatter_plotly(df, x, y, z, color=None, title="3D散布図"):
    """Plotly 3D散布図"""
    if color and color in df.columns:
        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, title=title,
                           color_continuous_scale='viridis', opacity=0.6)
    else:
        fig = px.scatter_3d(df, x=x, y=y, z=z, title=title, opacity=0.6)
    
    fig.update_layout(
        height=700,
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z
        ),
        template='plotly_white'
    )
    return fig

def plot_box_plotly(df, column, group=None, title="箱ひげ図"):
    """Plotly箱ひげ図"""
    if group and group in df.columns:
        fig = px.box(df, x=group, y=column, title=title)
    else:
        fig = px.box(df, y=column, title=title)
    
    fig.update_layout(
        height=600,
        template='plotly_white'
    )
    return fig

def plot_violin_plotly(df, column, group=None, title="バイオリン図"):
    """Plotlyバイオリン図"""
    if group and group in df.columns:
        fig = px.violin(df, x=group, y=column, title=title, box=True)
    else:
        fig = px.violin(df, y=column, title=title, box=True)
    
    fig.update_layout(
        height=600,
        template='plotly_white'
    )
    return fig

def plot_contour_plotly(df, x, y, z, title="等高線図"):
    """Plotly等高線図"""
    try:
        pivot_data = df.pivot_table(values=z, index=y, columns=x, aggfunc='mean')
        
        fig = go.Figure(data=go.Contour(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color='white')
            )
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            xaxis_title=x,
            yaxis_title=y,
            template='plotly_white'
        )
        return fig
    except Exception as e:
        st.error(f"等高線図生成エラー: {e}")
        return None

# =============================================================================
# 可視化関数群 - Seaborn
# =============================================================================

def plot_scatter_seaborn(df, x, y, color=None, title="散布図"):
    """Seaborn散布図"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if color and color in df.columns:
        sns.scatterplot(data=df, x=x, y=y, hue=color, palette='viridis', alpha=0.6, s=50, ax=ax)
    else:
        sns.scatterplot(data=df, x=x, y=y, alpha=0.6, s=50, ax=ax, color='steelblue')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_line_seaborn(df, x, y, group=None, title="線グラフ"):
    """Seaborn線グラフ"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if group and group in df.columns:
        sns.lineplot(data=df, x=x, y=y, hue=group, palette='viridis', ax=ax)
    else:
        sns.lineplot(data=df, x=x, y=y, ax=ax, color='steelblue')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_histogram_seaborn(df, column, bins=30, title="ヒストグラム"):
    """Seabornヒストグラム"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(data=df, x=column, bins=bins, kde=True, ax=ax, color='steelblue')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def plot_box_seaborn(df, column, group=None, title="箱ひげ図"):
    """Seaborn箱ひげ図"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if group and group in df.columns:
        sns.boxplot(data=df, x=group, y=column, palette='viridis', ax=ax)
    else:
        sns.boxplot(data=df, y=column, ax=ax, color='steelblue')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def plot_violin_seaborn(df, column, group=None, title="バイオリン図"):
    """Seabornバイオリン図"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if group and group in df.columns:
        sns.violinplot(data=df, x=group, y=column, palette='viridis', ax=ax)
    else:
        sns.violinplot(data=df, y=column, ax=ax, color='steelblue')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

def plot_heatmap_seaborn(df, columns=None, title="相関ヒートマップ"):
    """Seaborn相関ヒートマップ"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if columns:
        corr_data = df[columns].corr()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_data = df[numeric_cols].corr()
    
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_pairplot_seaborn(df, columns=None, hue=None):
    """Seabornペアプロット"""
    if columns:
        plot_df = df[columns + ([hue] if hue else [])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        plot_df = df[list(numeric_cols) + ([hue] if hue and hue in df.columns else [])]
    
    if hue and hue in plot_df.columns:
        g = sns.pairplot(plot_df, hue=hue, palette='viridis', diag_kind='kde', plot_kws={'alpha': 0.6})
    else:
        g = sns.pairplot(plot_df, diag_kind='kde', plot_kws={'alpha': 0.6})
    
    return g.fig

# =============================================================================
# パフォーマンス計測
# =============================================================================

def benchmark_plotting_performance(n_points, plot_type='scatter'):
    """プロット性能のベンチマーク"""
    np.random.seed(42)
    x = np.random.randn(n_points)
    y = np.random.randn(n_points)
    z = np.random.randn(n_points)
    
    df_bench = pd.DataFrame({'x': x, 'y': y, 'z': z})
    
    results = []
    
    if plot_type == 'scatter':
        start = time.perf_counter()
        fig = plot_scatter_matplotlib(df_bench, 'x', 'y', title=f"Matplotlib ({n_points:,} points)")
        plt.close(fig)
        matplotlib_time = (time.perf_counter() - start) * 1000
        results.append({'ライブラリ': 'matplotlib', '時間(ms)': matplotlib_time, 'データ点数': n_points})
        
        start = time.perf_counter()
        fig = plot_scatter_plotly(df_bench, 'x', 'y', title=f"Plotly ({n_points:,} points)")
        plotly_time = (time.perf_counter() - start) * 1000
        results.append({'ライブラリ': 'plotly', '時間(ms)': plotly_time, 'データ点数': n_points})
        
        start = time.perf_counter()
        fig = plot_scatter_seaborn(df_bench, 'x', 'y', title=f"Seaborn ({n_points:,} points)")
        plt.close(fig)
        seaborn_time = (time.perf_counter() - start) * 1000
        results.append({'ライブラリ': 'seaborn', '時間(ms)': seaborn_time, 'データ点数': n_points})
    
    elif plot_type == 'line':
        df_bench = df_bench.sort_values('x')
        
        start = time.perf_counter()
        fig = plot_line_matplotlib(df_bench, 'x', 'y', title=f"Matplotlib ({n_points:,} points)")
        plt.close(fig)
        matplotlib_time = (time.perf_counter() - start) * 1000
        results.append({'ライブラリ': 'matplotlib', '時間(ms)': matplotlib_time, 'データ点数': n_points})
        
        start = time.perf_counter()
        fig = plot_line_plotly(df_bench, 'x', 'y', title=f"Plotly ({n_points:,} points)")
        plotly_time = (time.perf_counter() - start) * 1000
        results.append({'ライブラリ': 'plotly', '時間(ms)': plotly_time, 'データ点数': n_points})
        
        start = time.perf_counter()
        fig = plot_line_seaborn(df_bench, 'x', 'y', title=f"Seaborn ({n_points:,} points)")
        plt.close(fig)
        seaborn_time = (time.perf_counter() - start) * 1000
        results.append({'ライブラリ': 'seaborn', '時間(ms)': seaborn_time, 'データ点数': n_points})
    
    return pd.DataFrame(results)

# =============================================================================
# EDA機能
# =============================================================================

def perform_eda_analysis(df):
    """包括的なEDA分析"""
    st.subheader("📊 データセット概要")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("行数", f"{len(df):,}")
    with col2:
        st.metric("列数", len(df.columns))
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("数値列", len(numeric_cols))
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("欠損率", f"{missing_pct:.2f}%")
    
    st.subheader("📋 データプレビュー")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("📈 統計サマリー")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("🔍 データ型情報")
    dtype_df = pd.DataFrame({
        '列名': df.columns,
        'データ型': df.dtypes.values,
        '非NULL数': df.count().values,
        '欠損数': df.isnull().sum().values,
        '欠損率(%)': (df.isnull().sum() / len(df) * 100).values
    })
    st.dataframe(dtype_df, use_container_width=True)
    
    if len(numeric_cols) > 1:
        st.subheader("🔗 相関分析")
        
        corr_threshold = st.slider("相関係数の閾値", 0.0, 1.0, 0.7, 0.05)
        corr_matrix = df[numeric_cols].corr()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= corr_threshold:
                    high_corr_pairs.append({
                        '変数1': corr_matrix.columns[i],
                        '変数2': corr_matrix.columns[j],
                        '相関係数': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            st.write(f"**|相関係数| ≥ {corr_threshold} のペア:**")
            st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
        else:
            st.info(f"閾値 {corr_threshold} 以上の相関を持つペアはありません")
        
        fig = plot_heatmap_seaborn(df, columns=list(numeric_cols[:15]), title="相関行列ヒートマップ")
        st.pyplot(fig)
        plt.close()

def create_distribution_analysis(df, column):
    """分布分析"""
    st.subheader(f"📊 {column} の分布分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均", f"{df[column].mean():.4f}")
    with col2:
        st.metric("中央値", f"{df[column].median():.4f}")
    with col3:
        st.metric("標準偏差", f"{df[column].std():.4f}")
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("最小値", f"{df[column].min():.4f}")
    with col5:
        st.metric("最大値", f"{df[column].max():.4f}")
    with col6:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        st.metric("IQR", f"{iqr:.4f}")
    
    outliers_lower = df[column] < (q1 - 1.5 * iqr)
    outliers_upper = df[column] > (q3 + 1.5 * iqr)
    n_outliers = outliers_lower.sum() + outliers_upper.sum()
    
    if n_outliers > 0:
        st.warning(f"⚠️ 外れ値検出: {n_outliers} 個 ({n_outliers/len(df)*100:.2f}%)")

# =============================================================================
# メインアプリケーション
# =============================================================================

def main():
    st.title("📊 材料工学向け可視化ライブラリ包括的比較ツール")
    st.markdown("""
    このアプリケーションは、Python の主要な可視化ライブラリを材料科学データで比較し、
    探索的データ解析（EDA）ツールとしても機能します。
    
    **対応ライブラリ:** matplotlib, plotly, seaborn
    """)
    
    st.sidebar.title("⚙️ 設定")
    
    dataset_option = st.sidebar.selectbox(
        "データセット選択",
        ["リチウム電池材料", "熱伝導シミュレーション", "Burgers方程式", "合成拡散データ", "CSVアップロード"]
    )
    
    df, metadata = None, None
    
    if dataset_option == "リチウム電池材料":
        df, metadata = load_lithium_battery_data()
    elif dataset_option == "熱伝導シミュレーション":
        df, metadata = load_heat_conduction_data()
    elif dataset_option == "Burgers方程式":
        df, metadata = load_burgers_data()
    elif dataset_option == "合成拡散データ":
        df, metadata = generate_synthetic_diffusion_data()
    elif dataset_option == "CSVアップロード":
        uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=['csv'])
        if uploaded_file:
            df, metadata = load_uploaded_data(uploaded_file)
    
    if df is None or metadata is None:
        st.info("👈 サイドバーからデータセットを選択してください")
        return
    
    st.sidebar.success(f"✅ {metadata['name']} 読み込み完了")
    st.sidebar.write(f"**行数:** {metadata['rows']:,}")
    st.sidebar.write(f"**列数:** {metadata['columns']}")
    
    tabs = st.tabs(["📊 EDA概要", "🎨 ライブラリ比較", "⚡ パフォーマンス計測", "🔬 材料科学デモ", "📚 ライブラリ機能表"])
    
    # =============================================================================
    # タブ1: EDA概要
    # =============================================================================
    with tabs[0]:
        st.header("探索的データ解析（EDA）")
        
        perform_eda_analysis(df)
        
        st.subheader("📉 分布分析")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("分析する列を選択", numeric_cols)
            create_distribution_analysis(df, selected_col)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_histogram_matplotlib(df, selected_col, bins=50, title=f"{selected_col} のヒストグラム")
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig = plot_box_matplotlib(df, selected_col, title=f"{selected_col} の箱ひげ図")
                st.pyplot(fig)
                plt.close()
    
    # =============================================================================
    # タブ2: ライブラリ比較
    # =============================================================================
    with tabs[1]:
        st.header("可視化ライブラリ比較")
        
        st.markdown("""
        同じデータを異なるライブラリで可視化し、機能と表現力を比較します。
        各ライブラリの特徴を理解し、用途に応じた最適な選択ができます。
        """)
        
        plot_type = st.selectbox(
            "プロットタイプ選択",
            ["散布図", "線グラフ", "ヒストグラム", "箱ひげ図", "バイオリン図", "ヒートマップ", "3D散布図", "等高線図"]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if plot_type in ["散布図", "線グラフ", "3D散布図"]:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X軸", numeric_cols, index=0 if len(numeric_cols) > 0 else 0)
            with col2:
                y_col = st.selectbox("Y軸", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            color_col = st.selectbox("色分け（オプション）", ["なし"] + numeric_cols)
            color_col = None if color_col == "なし" else color_col
            
            if plot_type == "3D散布図":
                z_col = st.selectbox("Z軸", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
        
        elif plot_type in ["ヒストグラム", "箱ひげ図", "バイオリン図"]:
            x_col = st.selectbox("対象列", numeric_cols)
            group_col = st.selectbox("グループ化（オプション）", ["なし"] + numeric_cols)
            group_col = None if group_col == "なし" else group_col
        
        elif plot_type in ["ヒートマップ", "等高線図"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X軸", numeric_cols, index=0 if len(numeric_cols) > 0 else 0)
            with col2:
                y_col = st.selectbox("Y軸", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            with col3:
                z_col = st.selectbox("値（Z）", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
        
        libraries = st.multiselect(
            "比較するライブラリ",
            ["matplotlib", "plotly", "seaborn"],
            default=["matplotlib", "plotly"]
        )
        
        if st.button("🚀 可視化実行", type="primary"):
            if not libraries:
                st.warning("少なくとも1つのライブラリを選択してください")
                return
            
            st.subheader(f"📊 {plot_type}の比較")
            
            cols = st.columns(len(libraries))
            
            for idx, lib in enumerate(libraries):
                with cols[idx]:
                    st.markdown(f"### {lib}")
                    
                    try:
                        if plot_type == "散布図":
                            if lib == "matplotlib":
                                fig = plot_scatter_matplotlib(df, x_col, y_col, color_col, f"{lib} - 散布図")
                                st.pyplot(fig)
                                plt.close()
                            elif lib == "plotly":
                                fig = plot_scatter_plotly(df, x_col, y_col, color_col, f"{lib} - 散布図")
                                st.plotly_chart(fig, use_container_width=True)
                            elif lib == "seaborn":
                                fig = plot_scatter_seaborn(df, x_col, y_col, color_col, f"{lib} - 散布図")
                                st.pyplot(fig)
                                plt.close()
                        
                        elif plot_type == "線グラフ":
                            if lib == "matplotlib":
                                fig = plot_line_matplotlib(df, x_col, y_col, color_col, f"{lib} - 線グラフ")
                                st.pyplot(fig)
                                plt.close()
                            elif lib == "plotly":
                                fig = plot_line_plotly(df, x_col, y_col, color_col, f"{lib} - 線グラフ")
                                st.plotly_chart(fig, use_container_width=True)
                            elif lib == "seaborn":
                                fig = plot_line_seaborn(df, x_col, y_col, color_col, f"{lib} - 線グラフ")
                                st.pyplot(fig)
                                plt.close()
                        
                        elif plot_type == "ヒストグラム":
                            if lib == "matplotlib":
                                fig = plot_histogram_matplotlib(df, x_col, bins=30, title=f"{lib} - ヒストグラム")
                                st.pyplot(fig)
                                plt.close()
                            elif lib == "plotly":
                                fig = plot_histogram_plotly(df, x_col, bins=30, title=f"{lib} - ヒストグラム")
                                st.plotly_chart(fig, use_container_width=True)
                            elif lib == "seaborn":
                                fig = plot_histogram_seaborn(df, x_col, bins=30, title=f"{lib} - ヒストグラム")
                                st.pyplot(fig)
                                plt.close()
                        
                        elif plot_type == "箱ひげ図":
                            if lib == "matplotlib":
                                fig = plot_box_matplotlib(df, x_col, group_col, f"{lib} - 箱ひげ図")
                                st.pyplot(fig)
                                plt.close()
                            elif lib == "plotly":
                                fig = plot_box_plotly(df, x_col, group_col, f"{lib} - 箱ひげ図")
                                st.plotly_chart(fig, use_container_width=True)
                            elif lib == "seaborn":
                                fig = plot_box_seaborn(df, x_col, group_col, f"{lib} - 箱ひげ図")
                                st.pyplot(fig)
                                plt.close()
                        
                        elif plot_type == "バイオリン図":
                            if lib == "plotly":
                                fig = plot_violin_plotly(df, x_col, group_col, f"{lib} - バイオリン図")
                                st.plotly_chart(fig, use_container_width=True)
                            elif lib == "seaborn":
                                fig = plot_violin_seaborn(df, x_col, group_col, f"{lib} - バイオリン図")
                                st.pyplot(fig)
                                plt.close()
                            elif lib == "matplotlib":
                                st.info("matplotlibはバイオリン図を直接サポートしていません")
                        
                        elif plot_type == "ヒートマップ":
                            if lib == "matplotlib":
                                fig = plot_heatmap_matplotlib(df, x_col, y_col, z_col, f"{lib} - ヒートマップ")
                                if fig:
                                    st.pyplot(fig)
                                    plt.close()
                            elif lib == "plotly":
                                fig = plot_heatmap_plotly(df, x_col, y_col, z_col, f"{lib} - ヒートマップ")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            elif lib == "seaborn":
                                st.info("seabornは相関ヒートマップに特化しています")
                        
                        elif plot_type == "3D散布図":
                            if lib == "matplotlib":
                                fig = plot_3d_scatter_matplotlib(df, x_col, y_col, z_col, color_col, f"{lib} - 3D散布図")
                                st.pyplot(fig)
                                plt.close()
                            elif lib == "plotly":
                                fig = plot_3d_scatter_plotly(df, x_col, y_col, z_col, color_col, f"{lib} - 3D散布図")
                                st.plotly_chart(fig, use_container_width=True)
                            elif lib == "seaborn":
                                st.info("seabornは3D可視化をサポートしていません")
                        
                        elif plot_type == "等高線図":
                            if lib == "plotly":
                                fig = plot_contour_plotly(df, x_col, y_col, z_col, f"{lib} - 等高線図")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"{lib}の等高線図は未実装です")
                        
                        with st.expander("📝 コード例を見る"):
                            if plot_type == "散布図":
                                if lib == "matplotlib":
                                    st.code(f"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['{x_col}'], df['{y_col}'], alpha=0.6)
ax.set_xlabel('{x_col}')
ax.set_ylabel('{y_col}')
ax.set_title('散布図')
plt.show()
                                    """, language='python')
                                elif lib == "plotly":
                                    st.code(f"""
import plotly.express as px

fig = px.scatter(df, x='{x_col}', y='{y_col}', 
                title='散布図')
fig.show()
                                    """, language='python')
                                elif lib == "seaborn":
                                    st.code(f"""
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df, x='{x_col}', y='{y_col}')
plt.title('散布図')
plt.show()
                                    """, language='python')
                    
                    except Exception as e:
                        st.error(f"エラー: {str(e)}")
    
    # =============================================================================
    # タブ3: パフォーマンス計測
    # =============================================================================
    with tabs[2]:
        st.header("⚡ パフォーマンス計測")
        
        st.markdown("""
        異なるデータサイズでの各ライブラリのレンダリング速度を計測します。
        大規模データを扱う際のライブラリ選択の参考にしてください。
        """)
        
        bench_plot_type = st.selectbox(
            "ベンチマークするプロットタイプ",
            ["scatter", "line"],
            format_func=lambda x: "散布図" if x == "scatter" else "線グラフ"
        )
        
        data_sizes = st.multiselect(
            "データサイズ（点数）",
            [1000, 5000, 10000, 50000, 100000],
            default=[1000, 10000, 50000]
        )
        
        if st.button("🚀 ベンチマーク実行", type="primary"):
            if not data_sizes:
                st.warning("少なくとも1つのデータサイズを選択してください")
                return
            
            st.subheader("📊 ベンチマーク結果")
            
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, size in enumerate(sorted(data_sizes)):
                status_text.text(f"計測中: {size:,} 点...")
                results = benchmark_plotting_performance(size, bench_plot_type)
                all_results.append(results)
                progress_bar.progress((idx + 1) / len(data_sizes))
            
            status_text.text("計測完了!")
            
            combined_results = pd.concat(all_results, ignore_index=True)
            
            st.dataframe(combined_results, use_container_width=True)
            
            fig = px.line(combined_results, x='データ点数', y='時間(ms)', 
                         color='ライブラリ', markers=True,
                         title='レンダリング時間の比較',
                         log_x=True, log_y=True)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📝 パフォーマンス分析")
            
            for size in sorted(data_sizes):
                size_results = combined_results[combined_results['データ点数'] == size]
                fastest = size_results.loc[size_results['時間(ms)'].idxmin()]
                slowest = size_results.loc[size_results['時間(ms)'].idxmax()]
                
                st.write(f"**{size:,} 点の場合:**")
                st.write(f"- 最速: {fastest['ライブラリ']} ({fastest['時間(ms)']:.2f} ms)")
                st.write(f"- 最遅: {slowest['ライブラリ']} ({slowest['時間(ms)']:.2f} ms)")
                st.write(f"- 速度差: {slowest['時間(ms)'] / fastest['時間(ms)']:.2f}倍")
                st.write("")
    
    # =============================================================================
    # タブ4: 材料科学デモ
    # =============================================================================
    with tabs[3]:
        st.header("🔬 材料科学特有の可視化デモ")
        
        st.markdown("""
        材料科学で頻繁に使用される可視化パターンのデモンストレーションです。
        拡散プロファイル、相図、組成-物性関係などを実例で示します。
        """)
        
        demo_type = st.selectbox(
            "デモタイプ選択",
            ["拡散プロファイル時間発展", "濃度ヒートマップ", "組成-物性関係", "相関ペアプロット"]
        )
        
        if demo_type == "拡散プロファイル時間発展":
            st.subheader("📈 拡散プロファイルの時間発展")
            
            if 'x' in df.columns and 'C' in df.columns and 't' in df.columns:
                time_values = sorted(df['t'].unique())
                selected_times = st.multiselect(
                    "表示する時刻を選択",
                    time_values,
                    default=time_values[::max(1, len(time_values)//5)][:5]
                )
                
                if selected_times:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Matplotlib版（静的）**")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for t in selected_times:
                            df_t = df[df['t'] == t].sort_values('x')
                            ax.plot(df_t['x'], df_t['C'], label=f't={t:.3f}', linewidth=2)
                        ax.set_xlabel('位置 x', fontsize=12)
                        ax.set_ylabel('濃度 C', fontsize=12)
                        ax.set_title('拡散プロファイルの時間発展', fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.markdown("**Plotly版（インタラクティブ）**")
                        fig = go.Figure()
                        for t in selected_times:
                            df_t = df[df['t'] == t].sort_values('x')
                            fig.add_trace(go.Scatter(
                                x=df_t['x'], y=df_t['C'],
                                mode='lines',
                                name=f't={t:.3f}',
                                line=dict(width=2)
                            ))
                        fig.update_layout(
                            title='拡散プロファイルの時間発展',
                            xaxis_title='位置 x',
                            yaxis_title='濃度 C',
                            height=600,
                            hovermode='x unified',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("このデータセットには拡散プロファイルデータ（x, C, t列）が含まれていません")
        
        elif demo_type == "濃度ヒートマップ":
            st.subheader("🌡️ 濃度分布ヒートマップ")
            
            if 'x' in df.columns and 'C' in df.columns and 't' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Matplotlib版**")
                    fig = plot_heatmap_matplotlib(df, 'x', 't', 'C', 'Matplotlib - 濃度分布')
                    if fig:
                        st.pyplot(fig)
                        plt.close()
                
                with col2:
                    st.markdown("**Plotly版（インタラクティブ）**")
                    fig = plot_heatmap_plotly(df, 'x', 't', 'C', 'Plotly - 濃度分布')
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**等高線図（Plotly）**")
                fig = plot_contour_plotly(df, 'x', 't', 'C', '濃度等高線図')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("このデータセットには時空間データ（x, C, t列）が含まれていません")
        
        elif demo_type == "組成-物性関係":
            st.subheader("🔗 組成と物性の関係")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_prop = st.selectbox("X軸（組成/特性）", numeric_cols, index=0)
                with col2:
                    y_prop = st.selectbox("Y軸（物性）", numeric_cols, index=1)
                with col3:
                    color_prop = st.selectbox("色分け", ["なし"] + numeric_cols, index=0)
                
                color_prop = None if color_prop == "なし" else color_prop
                
                st.markdown("**Seabornによる統計的可視化**")
                fig = plot_scatter_seaborn(df, x_prop, y_prop, color_prop, 
                                          f'{x_prop} vs {y_prop}')
                st.pyplot(fig)
                plt.close()
                
                st.markdown("**Plotlyによるインタラクティブ探索**")
                fig = plot_scatter_plotly(df, x_prop, y_prop, color_prop,
                                         f'{x_prop} vs {y_prop}')
                st.plotly_chart(fig, use_container_width=True)
                
                if len(numeric_cols) >= 3:
                    z_prop = st.selectbox("Z軸（3D可視化用）", numeric_cols, index=2)
                    st.markdown("**3D散布図（Plotly）**")
                    fig = plot_3d_scatter_plotly(df, x_prop, y_prop, z_prop, color_prop,
                                                f'{x_prop} vs {y_prop} vs {z_prop}')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("このデータセットには十分な数値列がありません")
        
        elif demo_type == "相関ペアプロット":
            st.subheader("🔗 変数間の相関ペアプロット")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                n_vars = st.slider("表示する変数数", 2, min(10, len(numeric_cols)), 
                                  min(5, len(numeric_cols)))
                
                selected_vars = st.multiselect(
                    "変数を選択（空の場合は自動選択）",
                    numeric_cols,
                    default=[]
                )
                
                if not selected_vars:
                    selected_vars = numeric_cols[:n_vars]
                
                hue_var = st.selectbox("色分け変数（オプション）", ["なし"] + numeric_cols)
                hue_var = None if hue_var == "なし" else hue_var
                
                if st.button("ペアプロット生成"):
                    with st.spinner("ペアプロット生成中..."):
                        fig = plot_pairplot_seaborn(df, selected_vars, hue_var)
                        st.pyplot(fig)
                        plt.close()
            else:
                st.warning("このデータセットには十分な数値列がありません")
    
    # =============================================================================
    # タブ5: ライブラリ機能表
    # =============================================================================
    with tabs[4]:
        st.header("📚 ライブラリ機能比較表")
        
        st.markdown("""
        各可視化ライブラリの特徴と適用場面を一覧で比較します。
        プロジェクトの要件に応じて最適なライブラリを選択してください。
        """)
        
        capability_df = pd.DataFrame(LIBRARY_CAPABILITIES).T
        st.dataframe(capability_df, use_container_width=True)
        
        st.subheader("📊 プロットタイプ対応表")
        
        plot_support = {
            "プロットタイプ": [
                "散布図", "線グラフ", "ヒストグラム", "箱ひげ図", "バイオリン図",
                "ヒートマップ", "等高線図", "3D散布図", "3D曲面", "ペアプロット",
                "アニメーション", "インタラクティブ"
            ],
            "matplotlib": [
                "✅", "✅", "✅", "✅", "❌", "✅", "✅", "✅", "✅", "❌", "✅", "❌"
            ],
            "plotly": [
                "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "❌", "✅", "✅"
            ],
            "seaborn": [
                "✅", "✅", "✅", "✅", "✅", "✅", "❌", "❌", "❌", "✅", "❌", "❌"
            ]
        }
        
        plot_support_df = pd.DataFrame(plot_support)
        st.dataframe(plot_support_df, use_container_width=True)
        
        st.subheader("💡 選択ガイドライン")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Matplotlib")
            st.markdown("""
            **こんな時に使う:**
            - 論文・学会発表用の高品質図表
            - 細かいカスタマイズが必要
            - 静的な図で十分
            - PDF/SVG出力が必要
            
            **強み:**
            - 出版品質の図表
            - 完全なカスタマイズ性
            - 豊富なドキュメント
            - 安定した動作
            """)
        
        with col2:
            st.markdown("### Plotly")
            st.markdown("""
            **こんな時に使う:**
            - Webダッシュボード
            - インタラクティブな探索
            - データの詳細確認が必要
            - プレゼンテーション
            
            **強み:**
            - 完全インタラクティブ
            - 美しいデフォルト
            - ズーム・パン・ホバー
            - HTML出力
            """)
        
        with col3:
            st.markdown("### Seaborn")
            st.markdown("""
            **こんな時に使う:**
            - 統計的可視化
            - 分布の比較
            - 相関分析
            - 素早いEDA
            
            **強み:**
            - 統計に特化
            - 美しいデフォルト
            - 簡潔なAPI
            - Pandas統合
            """)
        
        st.subheader("🎯 用途別推奨ライブラリ")
        
        recommendations = {
            "用途": [
                "論文図表作成",
                "Webダッシュボード",
                "探索的データ解析",
                "プレゼンテーション",
                "大規模データ（10万点以上）",
                "3D可視化",
                "統計的可視化",
                "リアルタイム更新"
            ],
            "第1推奨": [
                "matplotlib",
                "plotly",
                "seaborn",
                "plotly",
                "matplotlib",
                "plotly",
                "seaborn",
                "plotly"
            ],
            "第2推奨": [
                "seaborn",
                "matplotlib",
                "plotly",
                "matplotlib",
                "plotly",
                "matplotlib",
                "matplotlib",
                "matplotlib"
            ],
            "理由": [
                "出版品質、PDF/SVG出力",
                "インタラクティブ、HTML出力",
                "統計機能、簡潔なAPI",
                "インタラクティブ、視覚的魅力",
                "静的レンダリングが高速",
                "完全な3D対応、回転可能",
                "統計特化、分布可視化",
                "動的更新、ストリーミング対応"
            ]
        }
        
        recommendations_df = pd.DataFrame(recommendations)
        st.dataframe(recommendations_df, use_container_width=True)

if __name__ == "__main__":
    main()
