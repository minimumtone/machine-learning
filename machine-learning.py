# -*- coding: utf-8 -*-
# ↑ BOM(Byte Order Mark)がないことを確認してください。

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV, cross_validate, train_test_split # train_test_split を追加
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer # MAE を追加
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import traceback
import inspect
import ast
import time
import joblib
import io
import datetime
from math import comb, prod # 組み合わせ計算用、総積計算用にprodを追加
# --- 機能追加：LHS用にscipy.stats.qmcをインポート ---
try:
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Regression Analysis App with PCA & Hyperparameter Tuning", layout="wide")
st.title("📊 Machine Learning Regression, PCA & Hyperparameter Tuning App")
st.markdown("""
This app performs regression analysis, PCA, and hyperparameter optimization (Grid Search) on uploaded CSV data.
Data is split into training and validation sets. Model evaluation (CV) uses the training set. Final model trained on the full training set is used for validation set predictions.
**Encoding errors?** Ensure 'Ignore' or 'Replace' is selected in the sidebar. If errors *still* persist, the CSV file is likely corrupted.
**Target Selection:** When a target variable is selected, all other columns are automatically selected as features. Feature selection updates instantly.
**PCA Target Handling:** Option to include the target variable in PCA calculation is available in the PCA section.
**Log Transformation:** Optionally apply log(1+x) transformation to features and/or target variable in the preprocessing step.
**Evaluation Metric:** Choose the primary metric (R², MAE, RMSE) for evaluation and Grid Search optimization. **Default is RMSE.**
**Hyperparameter Optimization:** Enable Grid Search in the setup section to find the best model parameters based on the chosen metric. Default parameter suggestions provided for Grid Search. **Polynomial degree check added.** A progress bar estimates GridSearch duration.
**PCA Biplot Coloring:** Select a feature below the PCA results to color the data points in the biplot. **Biplot is now square. Target variable selection automatically updates Biplot color.**
**LHS Simulation:** After running the regression, generate simulated input data using Latin Hypercube Sampling (LHS) based on feature ranges and predict outcomes with the final model. Requires SciPy. **Optimization method updated.** Optionally plots LHS results (top 10 highest/lowest predictions, **overlaid on original data**) on the original PCA biplot if feature sets match.
""")
if not SCIPY_AVAILABLE:
    st.warning("⚠️ **Optional Dependency Missing:** SciPy is not installed. LHS sampling feature will be disabled. Install it (~~~pip install scipy~~~) for full functionality.")


# --- Helper function for Biplot ---
def biplot(score, coeff, labels=None, pc_x=0, pc_y=1, explained_variance_ratio=None,
           color_data=None, color_label='Color Feature', is_categorical=False, point_alpha=0.5,
           fig_size=(7, 7), make_square=False,
           background_score=None, background_alpha=0.1, background_color='darkgrey'):
    """ Generates a PCA biplot with optional background data. """
    pca_x_label = f'PC{pc_x+1}'
    pca_y_label = f'PC{pc_y+1}'
    if explained_variance_ratio is not None and max(pc_x, pc_y) < len(explained_variance_ratio):
        pca_x_label += f' ({explained_variance_ratio[pc_x]*100:.2f}%)'
        pca_y_label += f' ({explained_variance_ratio[pc_y]*100:.2f}%)'

    xs = score[:, pc_x] if score is not None and score.shape[1] > pc_x else np.array([])
    ys = score[:, pc_y] if score is not None and score.shape[1] > pc_y else np.array([])
    bg_xs = background_score[:, pc_x] if background_score is not None and background_score.shape[1] > pc_x else np.array([])
    bg_ys = background_score[:, pc_y] if background_score is not None and background_score.shape[1] > pc_y else np.array([])
    n_loadings = coeff.shape[1] if coeff is not None else 0

    fig, ax = plt.subplots(figsize=fig_size)
    scatter_kwargs = {'alpha': point_alpha, 's': 40}
    cbar = None
    legend_handles = None
    is_cat_final = is_categorical

    # Plot background
    background_legend = None
    if bg_xs.size > 0 and bg_ys.size > 0: # Check size instead of len for numpy arrays
        ax.scatter(bg_xs, bg_ys, alpha=background_alpha, s=30, color=background_color, label='_nolegend_')
        background_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=background_color, markersize=8, label='Original Data (PCA)', alpha=0.8)

    # Process foreground
    foreground_legend_handles = []
    # Ensure xs is a numpy array and has elements before checking length
    if color_data is not None and isinstance(xs, np.ndarray) and xs.size > 0 and len(xs) == len(color_data):
        color_data_clean = color_data.copy()
        if not isinstance(color_data_clean, (pd.Series, np.ndarray)): color_data_clean = pd.Series(color_data_clean)
        # Check if color_data_clean is empty after potential conversion
        if color_data_clean.empty:
             is_numeric_input = False # Treat empty as non-numeric/categorical
             is_cat_final = True
        else:
            is_numeric_input = pd.api.types.is_numeric_dtype(color_data_clean.dtype) and not pd.api.types.is_bool_dtype(color_data_clean.dtype)
            is_cat_final = is_categorical or not is_numeric_input

        if is_numeric_input and not is_categorical:
            cmap = plt.get_cmap('viridis')
            scatter_kwargs['c'] = color_data_clean; scatter_kwargs['cmap'] = cmap
        else: # Categorical or empty
            is_cat_final = True
            if color_data_clean.empty:
                pass # Skip legend entries if empty
            else:
                color_data_clean = color_data_clean.astype(str).fillna('NaN')
                codes, unique_categories = pd.factorize(color_data_clean, sort=True); num_categories = len(unique_categories)
                cmap_name = 'tab10' if num_categories <= 10 else ('tab20' if num_categories <= 20 else 'viridis')
                cmap = plt.get_cmap(cmap_name, num_categories)
                scatter_kwargs['c'] = codes; scatter_kwargs['cmap'] = cmap; scatter_kwargs['vmin'] = -0.5; scatter_kwargs['vmax'] = num_categories - 0.5
                max_legend_entries = 15; categories_to_show = unique_categories; show_more = False
                if num_categories > max_legend_entries: categories_to_show = unique_categories[:max_legend_entries]; show_more = True
                for i, cat in enumerate(categories_to_show):
                    label = str(cat)
                    try: marker_color = cmap(i / max(1, num_categories - 1))
                    except Exception: marker_color = 'gray'
                    if label == 'NaN': marker_color = 'lightgrey'; label = 'NaN / Missing'
                    foreground_legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=marker_color, markersize=8, label=label))
                if show_more: foreground_legend_handles.append(plt.Line2D([0], [0], marker='', ls='', label=f'... ({num_categories - max_legend_entries} more)'))

    # Plot foreground
    scatter = None
    if xs.size > 0 and ys.size > 0: # Check size for numpy arrays
        try: # Add try-except for the scatter plot itself
            scatter = ax.scatter(xs, ys, **scatter_kwargs)
        except Exception as scatter_err:
            st.warning(f"Could not plot scatter points: {scatter_err}")


    # Combine legends
    legend_visible = False; final_legend_handles = []
    if background_legend: final_legend_handles.append(background_legend)
    if foreground_legend_handles: final_legend_handles.extend(foreground_legend_handles)
    if final_legend_handles:
        ax.legend(handles=final_legend_handles, title=color_label, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
        legend_visible = True
    elif not is_cat_final and color_data is not None and scatter is not None: # Ensure scatter exists
        cbar = fig.colorbar(scatter, ax=ax, label=color_label, pad=0.02, shrink=0.8); legend_visible = True

    # Plot loadings
    all_xs_list = []
    if xs.size > 0: all_xs_list.append(xs)
    if bg_xs.size > 0: all_xs_list.append(bg_xs)
    all_xs = np.concatenate(all_xs_list) if all_xs_list else np.array([])

    all_ys_list = []
    if ys.size > 0: all_ys_list.append(ys)
    if bg_ys.size > 0: all_ys_list.append(bg_ys)
    all_ys = np.concatenate(all_ys_list) if all_ys_list else np.array([])


    if coeff is not None and all_xs.size > 0 and all_ys.size > 0:
        x_range = all_xs.max() - all_xs.min() if all_xs.size > 1 else 1; y_range = all_ys.max() - all_ys.min() if all_ys.size > 1 else 1
        arrow_scale = max(x_range, y_range) * 0.4 if max(x_range, y_range) > 0 else 1; arrow_scale = max(arrow_scale, 0.1)
        for i in range(n_loadings):
            if pc_x < coeff.shape[0] and pc_y < coeff.shape[0] and i < coeff.shape[1]:
                ax_x, ax_y = coeff[pc_x, i] * arrow_scale, coeff[pc_y, i] * arrow_scale
                if abs(ax_x) > 1e-6 or abs(ax_y) > 1e-6:
                    ax.arrow(0, 0, ax_x, ax_y, color='r', alpha=0.8, head_width=max(0.001, 0.02 * arrow_scale), length_includes_head=True)
                    if labels is not None and i < len(labels): ax.text(ax_x * 1.15, ax_y * 1.15, labels[i], color='g', ha='center', va='center', fontsize=9)

    # Final adjustments
    ax.set_xlabel(pca_x_label); ax.set_ylabel(pca_y_label); ax.set_title('PCA Biplot'); ax.grid(True, linestyle='--', alpha=0.6); ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
    if all_xs.size > 0 and all_ys.size > 0:
        x_max_abs = np.abs(all_xs).max(); y_max_abs = np.abs(all_ys).max()
        if make_square: x_lim_val = y_lim_val = max(x_max_abs, y_max_abs) * 1.2
        else: x_lim_val = x_max_abs * 1.2; y_lim_val = y_max_abs * 1.2
    else: x_lim_val, y_lim_val = 1, 1
    x_lim_val = max(x_lim_val, 0.1); y_lim_val = max(y_lim_val, 0.1) # Ensure minimum limit
    ax.set_xlim(-x_lim_val, x_lim_val); ax.set_ylim(-y_lim_val, y_lim_val)
    if make_square: ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.80 if legend_visible else 1, 1])
    return fig


# --- Helper function to parse parameter ranges ---
def parse_param_string(param_str, param_type=float, default_value=None, is_list=True):
    # Kept for complex inputs like MLP layers
    if not isinstance(param_str, str) or not param_str.strip():
        if default_value is not None: return [default_value] if is_list and not isinstance(default_value, list) else default_value
        else: return [] if is_list else None
    items = [item.strip() for item in param_str.split(',')]; parsed_list = []; error_messages = []
    for item in items:
        if not item: continue
        try:
            if item.lower() == 'none' and param_type in [int, float, tuple, str]: parsed_list.append(None); continue
            if param_type == int: parsed_list.append(int(item))
            elif param_type == float: parsed_list.append(float(item))
            elif param_type == str: parsed_list.append(None if item.lower() == 'none' else item)
            elif param_type == tuple:
                try:
                    evaluated = ast.literal_eval(item)
                    if isinstance(evaluated, int): parsed_list.append((evaluated,))
                    elif isinstance(evaluated, tuple) and all(isinstance(i, int) for i in evaluated): parsed_list.append(evaluated)
                    else: raise ValueError(f"Tuple elements must be integers: {item}")
                except (ValueError, SyntaxError, TypeError) as e: raise ValueError(f"Cannot parse '{item}' as tuple. Format: '(100,)' or '(50, 25)'. Error: {e}")
            else: parsed_list.append(item)
        except ValueError as e: error_messages.append(f"Skipping '{item}' ({param_type.__name__}): {e}")
    if error_messages: st.warning("Param parsing issues:"); [st.warning(f"- {msg}") for msg in error_messages]
    if not parsed_list:
        if default_value is not None: st.warning(f"No valid items from '{param_str}'. Using default: {default_value}"); return [default_value] if is_list and not isinstance(default_value, list) else default_value
        else: return [] if is_list else None
    if not is_list:
        if len(parsed_list) == 1: return parsed_list[0]
        elif len(parsed_list) > 1: st.warning(f"Expected single value, got list. Using first: {parsed_list[0]}"); return parsed_list[0]
        else: return default_value
    return parsed_list

# --- Helper Function for Polynomial Feature Count ---
def calculate_poly_features(n_features, degree):
    try: return comb(n_features + degree, degree) - 1
    except Exception: st.warning("Could not calculate poly feature count."); return (n_features ** degree) if degree > 1 else n_features

# --- Callback function ---
def update_dependent_selections_on_target_change():
    """Updates PCA color and feature selection when the target variable changes."""
    target_val = st.session_state.get('target_selector')
    current_df = st.session_state.get('df')

    # Update PCA Biplot Color Feature
    if target_val:
        st.session_state['pca_biplot_color_col'] = target_val

    # Update Feature Selector to include all other columns
    if target_val and current_df is not None and not current_df.empty:
        all_columns = current_df.columns.tolist()
        default_features = [col for col in all_columns if col != target_val]
        st.session_state.feature_selector = default_features

# --- Define state keys ---
analysis_result_keys = {
    'final_estimator', 'X_reg_global', 'y_reg_global', 'X_train', 'X_test', 'y_train', 'y_test',
    'feature_cols_global', 'final_estimator_trained', 'target_col_global',
    'model_type_full_global', 'importance_df', 'pca_imputer', 'pca_scaler',
    'pca_model', 'numeric_cols_pca', 'pca_loadings_matrix',
    'explained_variance_ratio_pca', 'pca_scores_global', 'lhs_results_df',
    'run_analysis_clicked', 'grid_search_results_df', 'cv_results_processed',
    'test_results', 'y_pred_cv_train', 'y_pred_test', 'k_folds_run',
    'test_split_ratio_run', 'pca_df_shape'
}
# ★★★ Updated widget_state_defaults with multiple GS defaults ★★★
widget_state_defaults = {
    'df': None, 'uploaded_filename': None,
    'target_selector': None, 'feature_selector': [],
    'pca_include_target': False, 'pca_biplot_color_col': "None",
    'test_split_ratio': 20,
    'nan_strategy': "Drop Rows with NaNs",
    'log_target_checkbox': False, 'log_features_checkbox': False,
    'scaling_checkbox': True,
    'model_selector': "Linear Regression",
    'primary_metric_selector': "RMSE", 'k_folds_slider': 10,
    'optimize_check': False, 'opt_metric_selector': 'RMSE',
    # Fixed Params (Single Value Defaults)
    'r_a': 1.0, 'l_a': 1.0, 'p_d_num': 2, 'p_r_c': True, 'p_r_a': 1.0,
    'rf_est': 100, 'rf_max_d': 'None', 'rf_max_f': 'sqrt', 'rf_min_s': 2, 'rf_min_l': 1,
    'mlp_h': '100', 'mlp_a': 'relu', 'mlp_s': 'adam', 'mlp_alpha': 0.0001, 'mlp_lr': 0.001,
    'mlp_iter': 500, 'mlp_stop': True, 'mlp_pat': 10,
    # GridSearch Params (Multiple Value Defaults)
    'r_a_gs': [0.1, 1.0, 10.0],          # Ridge Alpha Options
    'l_a_gs': [0.01, 0.1, 1.0],          # Lasso Alpha Options
    'p_d_gs': [2, 3],                    # Poly Degree Options
    'p_r_a_gs': [0.1, 1.0, 10.0],        # Poly Ridge Alpha Options
    'rf_n_est_gs': [100, 200, 500],      # RF n_estimators Options
    'rf_max_depth_gs': ['10', '20', 'None'], # RF max_depth Options (str)
    'rf_max_feat_gs': ['sqrt', '0.8'],   # RF max_features Options (str/float)
    'rf_min_split_gs': [2, 5, 10],       # RF min_split Options
    'rf_min_leaf_gs': [1, 3, 5],         # RF min_leaf Options
    'mlp_h_g': '(100,), (50, 50)',       # MLP hidden_layer_sizes (string)
    'mlp_a_gs': ['relu', 'tanh'],        # MLP activation Options
    'mlp_s_gs': ['adam'],                # MLP solver Options
    'mlp_alpha_gs': [0.0001, 0.001, 0.01], # MLP alpha Options
    'mlp_lr_gs': [0.001, 0.01],           # MLP lr_init Options
    # LHS
    'lhs_n': 10000
}

# --- Initialize session state ---
for key, default_value in widget_state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
for key in analysis_result_keys:
    if key not in st.session_state:
        st.session_state[key] = None


# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("1. Upload CSV", type=["csv"])
encoding_error_handling = st.sidebar.selectbox("2. CSV Encoding Errors", ["Strict", "Ignore", "Replace"], index=1, help="'Ignore' or 'Replace' if errors occur.")
pandas_error_handling = 'strict' if encoding_error_handling == "Strict" else ('ignore' if encoding_error_handling == "Ignore" else 'replace')

# --- Sidebar - Data Split & Preprocessing ---
st.sidebar.subheader("🧪 3. Data Split")
test_size = st.sidebar.slider(
    "Validation Set Split Ratio (%)", 10, 50,
    value=st.session_state.test_split_ratio,
    step=5,
    key="test_split_ratio",
    help="Percentage of data for validation."
)
test_split_frac = test_size / 100.0

st.sidebar.subheader("🧹 4. Preprocessing")
nan_strategy_options = ["Drop Rows with NaNs", "Impute Mean", "Impute Median"]
current_nan_strategy = st.session_state.nan_strategy
if current_nan_strategy not in nan_strategy_options:
    current_nan_strategy = nan_strategy_options[0]
    st.session_state.nan_strategy = current_nan_strategy

missing_value_strategy = st.sidebar.selectbox(
    "NaN Handling", nan_strategy_options,
    index=nan_strategy_options.index(current_nan_strategy),
    key="nan_strategy"
)

log_transform_target = st.sidebar.checkbox(
    "Log Target(Y)",
    value=st.session_state.log_target_checkbox,
    key="log_target_checkbox"
)

log_transform_features = st.sidebar.checkbox(
    "Log Features(X)",
    value=st.session_state.log_features_checkbox,
    key="log_features_checkbox"
)

use_scaling = st.sidebar.checkbox(
    "Scale Feats",
    value=st.session_state.scaling_checkbox,
    key="scaling_checkbox"
)

# --- Main Logic ---
numeric_cols = []; target_col = None; feature_cols = []

if uploaded_file is not None:
    # File Reading Logic
    load_new_file = (st.session_state.df is None or (hasattr(uploaded_file, 'name') and uploaded_file.name != st.session_state.uploaded_filename))
    if load_new_file:
        st.warning("New file/first load, resetting analysis results (keeping selections).")
        for key in analysis_result_keys:
            if key in st.session_state:
                 if key == 'run_analysis_clicked': st.session_state[key] = False
                 elif key == 'final_estimator_trained': st.session_state[key] = False
                 else: st.session_state[key] = None

        if hasattr(uploaded_file, 'name'):
            st.session_state.uploaded_filename = uploaded_file.name

        df_load = None; error_messages = []; successful_encoding = None; encodings_to_try = ['utf-8', 'utf-8-sig', 'shift-jis', 'cp932', 'latin-1']
        st.subheader("Reading File..."); read_placeholder = st.empty()
        for enc in encodings_to_try:
            read_placeholder.write(f"🔄 Trying: `{enc}`...")
            try:
                uploaded_file.seek(0); df_load = pd.read_csv(uploaded_file, encoding=enc, encoding_errors=pandas_error_handling)
                st.sidebar.success(f"✅ Loaded ({enc})"); successful_encoding = enc; read_placeholder.success(f"Loaded: '{successful_encoding}'")
                st.session_state['df'] = df_load
                st.session_state['run_analysis_clicked'] = False

                persisted_target = st.session_state.get('target_selector')
                numeric_cols_new_df = df_load.select_dtypes(include=np.number).columns.tolist()
                all_columns_new_df = df_load.columns.tolist()

                if persisted_target and persisted_target in numeric_cols_new_df:
                     default_features_new_df = [col for col in all_columns_new_df if col != persisted_target]
                     st.session_state.feature_selector = default_features_new_df
                elif numeric_cols_new_df:
                     new_default_target = numeric_cols_new_df[-1]
                     st.session_state.target_selector = new_default_target
                     default_features_new_df = [col for col in all_columns_new_df if col != new_default_target]
                     st.session_state.feature_selector = default_features_new_df
                     st.session_state.pca_biplot_color_col = new_default_target
                else:
                    st.session_state.target_selector = None
                    st.session_state.feature_selector = []
                    st.session_state.pca_biplot_color_col = "None"

                break
            except Exception as e: error_messages.append(f"Err[{enc}]: {e}")
        if st.session_state['df'] is None: read_placeholder.empty(); st.error("💥 FATAL: Cannot read CSV."); [st.code(m, language='text') for m in error_messages]; st.stop()
        st.markdown("---")


    # Data Overview
    current_df = st.session_state.get('df', pd.DataFrame())
    if not current_df.empty:
        st.header("💾 Data Overview"); st.dataframe(current_df.head()); st.write(f"Shape: {current_df.shape}"); st.subheader("Statistics")
        try: st.dataframe(current_df.describe(include='all').round(3))
        except Exception: st.dataframe(current_df.describe().round(3))

        # --- Target & Feature Selection ---
        st.markdown("---"); st.header("🎯 Target & Features Selection")
        col_target, col_features = st.columns([1, 2])

        with col_target:
            st.subheader("Target (Y)")
            numeric_cols_reg = current_df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols_reg: st.error("No numeric columns available in the data."); st.stop()

            current_target_selection = st.session_state.get('target_selector')
            target_index = 0
            if current_target_selection in numeric_cols_reg:
                target_index = numeric_cols_reg.index(current_target_selection)
            elif len(numeric_cols_reg) > 0:
                target_index = len(numeric_cols_reg) - 1
                if st.session_state.target_selector not in numeric_cols_reg:
                    st.session_state.target_selector = numeric_cols_reg[target_index]

            target_col_selected = st.selectbox(
                "Select Target Variable: Press enter to reflect",
                numeric_cols_reg,
                index=target_index,
                key="target_selector",
                on_change=update_dependent_selections_on_target_change,
                help="Select the numeric variable to predict."
            )
            st.caption("ℹ️ Features below will update automatically when Target changes.")
            target_col = st.session_state.target_selector

        with col_features:
            st.subheader("Features (X)")
            available_features = current_df.columns.tolist()
            if target_col:
                options_features = [col for col in available_features if col != target_col]
                feature_selection_state = st.session_state.get('feature_selector', [])
                valid_selection = [f for f in feature_selection_state if f in options_features]

                feature_cols_selected = st.multiselect(
                    "Select Feature Variables:",
                    options_features,
                    default=st.session_state.feature_selector,
                    key="feature_selector",
                    help="Select the predictor variables. Automatically updated when Target changes."
                )
                feature_cols = st.session_state.feature_selector
            else:
                 feature_cols_selected = []; feature_cols = []
                 st.warning("Select Target first.")

        # --- PCA Analysis ---
        st.markdown("---"); st.header("🧩 PCA")
        st.checkbox(
            "Include Target in PCA Calculation",
            value=st.session_state.pca_include_target,
            key="pca_include_target",
            help="Check to include target in PCA."
        )
        pca_performed = False
        if not current_df.empty and target_col:
            numeric_data_pca_base = current_df.select_dtypes(include=np.number); pca_target_status = ""; cols_for_this_pca = []
            if not st.session_state.pca_include_target and target_col in numeric_data_pca_base.columns:
                numeric_data_pca = numeric_data_pca_base.drop(columns=[target_col]); pca_target_status = f"(Target '{target_col}' EXCLUDED)"
            else:
                numeric_data_pca = numeric_data_pca_base.copy()
                if target_col in numeric_data_pca.columns and st.session_state.pca_include_target: pca_target_status = f"(Target '{target_col}' INCLUDED)"
                else: pca_target_status = "(All numeric columns included)"
            cols_for_this_pca = numeric_data_pca.columns.tolist(); st.write(f"PCA on numeric columns {pca_target_status}"); original_idx_pca = numeric_data_pca.index
            if numeric_data_pca.empty or len(cols_for_this_pca) < 2: st.warning("PCA needs >= 2 numeric columns.")
            else:
                pca_needs_recalc = (st.session_state.get('pca_model') is None or
                                    st.session_state.get('pca_scores_global') is None or
                                    st.session_state.get('numeric_cols_pca') != cols_for_this_pca or
                                    st.session_state.get('pca_df_shape') != numeric_data_pca.shape)

                if pca_needs_recalc:
                    st.write("Calculating PCA...");
                    try:
                        if not isinstance(numeric_data_pca, pd.DataFrame): numeric_data_pca = pd.DataFrame(numeric_data_pca, columns=cols_for_this_pca, index=original_idx_pca)
                        if numeric_data_pca.isnull().values.any():
                            st.info("Imputing NaNs with mean for PCA calculation.")
                            pca_imputer_internal = SimpleImputer(strategy='mean')
                            numeric_data_pca_imputed = pca_imputer_internal.fit_transform(numeric_data_pca)
                        else:
                            numeric_data_pca_imputed = numeric_data_pca.values

                        pca_scaler_internal = StandardScaler(); scaled = pca_scaler_internal.fit_transform(numeric_data_pca_imputed)
                        pca_model_internal = PCA(); scores = pca_model_internal.fit_transform(scaled)

                        st.session_state['pca_model'] = pca_model_internal
                        st.session_state['pca_scaler'] = pca_scaler_internal
                        st.session_state['numeric_cols_pca'] = cols_for_this_pca
                        st.session_state['pca_loadings_matrix'] = pca_model_internal.components_.T
                        st.session_state['explained_variance_ratio_pca'] = pca_model_internal.explained_variance_ratio_
                        st.session_state['pca_scores_global'] = scores
                        st.session_state['pca_df_shape'] = numeric_data_pca.shape
                        pca_performed = True
                    except Exception as e: st.error(f"PCA error: {e}"); st.code(traceback.format_exc(),language='python'); pca_performed=False
                else: st.write("Using cached PCA."); pca_performed = True

                if pca_performed and st.session_state.get('pca_scores_global') is not None:
                    _loadings=st.session_state['pca_loadings_matrix']; _expl_var=st.session_state['explained_variance_ratio_pca']; pca_scores=st.session_state['pca_scores_global']; _numeric_cols_pca=st.session_state['numeric_cols_pca']
                    n_comps=len(_expl_var); comp_idx=np.arange(1,n_comps+1); cum_var=np.cumsum(_expl_var)
                    st.subheader("PCA Results"); pc1,pc2=st.columns(2)
                    with pc1: fig_s,ax_s=plt.subplots(figsize=(4.5,3)); ax_s.bar(comp_idx,_expl_var); ax_s.set_title("Scree Plot"); st.pyplot(fig_s)
                    with pc2: fig_c,ax_c=plt.subplots(figsize=(4.5,3)); ax_c.plot(comp_idx,cum_var,'o--'); ax_c.axhline(0.9,c='g',ls=':'); ax_c.axhline(0.95,c='r',ls=':'); ax_c.set_title("Cumulative Var"); ax_c.set_ylim(0,1.1); st.pyplot(fig_c)
                    st.dataframe(pd.DataFrame({'PC':comp_idx,'Expl Var':_expl_var,'Cum Var':cum_var}).round(4)); st.markdown("---")
                    if n_comps>=2:
                        st.subheader("PCA Biplot (PC1 vs PC2)"); color_opts=["None"]+current_df.columns.tolist(); sel_color=st.session_state.get('pca_biplot_color_col',"None"); idx=color_opts.index(sel_color) if sel_color in color_opts else 0
                        color_col_selected_pca = st.selectbox("🎨 Color Biplot:",color_opts,index=idx,key='pca_biplot_color_col')
                        color_data=None; is_cat=False; label=""
                        color_col = st.session_state.pca_biplot_color_col
                        if color_col!="None" and color_col in current_df.columns:
                             if len(original_idx_pca) == len(pca_scores):
                                 try:
                                     color_data=current_df.loc[original_idx_pca, color_col].reset_index(drop=True)
                                     label=color_col
                                     is_cat=not pd.api.types.is_numeric_dtype(color_data.dtype) or pd.api.types.is_bool_dtype(color_data.dtype) or color_data.nunique()<20;
                                 except Exception as e: st.error(f"PCA color prep err: {e}")
                             else:
                                 st.warning("Index mismatch between PCA scores and original data, cannot apply color reliably.")

                        try:
                             if pca_scores is not None and pca_scores.shape[0] > 0:
                                 fig=biplot(pca_scores,_loadings,labels=_numeric_cols_pca,explained_variance_ratio=_expl_var,color_data=color_data,color_label=label,is_categorical=is_cat,make_square=True);
                                 st.pyplot(fig)
                             else:
                                 st.warning("Cannot generate Biplot: PCA scores are missing or empty.")
                        except Exception as e: st.error(f"Biplot err: {e}"); st.code(traceback.format_exc(),language='python')
                    else: st.warning("Biplot needs >= 2 PCs.")
        elif not target_col:
             st.info("Select a Target variable to perform PCA.")


        # Regression Setup
        st.markdown("---"); st.header("📈 Regression Analysis Setup")
        # --- Regression Setup Check ---
        target_col_state = st.session_state.get('target_selector')
        feature_cols_state = st.session_state.get('feature_selector', [])
        df_columns_available = list(current_df.columns) if current_df is not None else []
        target_selected_and_valid = target_col_state is not None and target_col_state in df_columns_available
        features_selected = bool(feature_cols_state)
        valid_features_exist = True
        if features_selected and target_selected_and_valid:
            if target_col_state in feature_cols_state: valid_features_exist = False
            elif not all(feat in df_columns_available for feat in feature_cols_state): valid_features_exist = False
        elif not features_selected: valid_features_exist = False
        can_run_analysis = target_selected_and_valid and valid_features_exist
        if not can_run_analysis:
             st.warning("⚠️ Please select a valid Target & Features in the section above.")
        # --- END Check ---

        # Model Selection & Eval
        st.subheader("🧠 Model, Metric & CV"); mcol1,mcol2,mcol3=st.columns([1,1,1])
        model_options = ["Linear Regression","Ridge","Lasso","Polynomial Regression","Random Forest","Neural Network (MLP)"]
        current_model = st.session_state.model_selector
        if current_model not in model_options: current_model = model_options[0]
        with mcol1: model_type=st.selectbox("Model", model_options, key="model_selector", index=model_options.index(current_model)); mt_full=model_type;
        if model_type=="Ridge": mt_full="Ridge(L2)"
        elif model_type=="Lasso": mt_full="Lasso(L1)"
        elif model_type=="Neural Network (MLP)": mt_full="MLP"
        metric_map={"R²":"r2","MAE":"neg_mean_absolute_error","RMSE":"neg_root_mean_squared_error"}; metric_disp_key_map={"r2":"R²","neg_mean_absolute_error":"MAE","neg_root_mean_squared_error":"RMSE"}; metric_short_key_map={"r2":"r2","neg_mean_absolute_error":"mae","neg_root_mean_squared_error":"rmse"}
        with mcol2:
            metric_options = list(metric_map.keys());
            current_primary_metric = st.session_state.primary_metric_selector
            if current_primary_metric not in metric_options: current_primary_metric = "RMSE"
            primary_metric=st.selectbox("Primary Metric (Test Set)", metric_options, index=metric_options.index(current_primary_metric), key="primary_metric_selector", help="Primary metric for validation set.")

        opt_metric=primary_metric; primary_metric_scorer=metric_map[primary_metric]; primary_metric_short=metric_disp_key_map[primary_metric_scorer]
        with mcol3:
            k_folds=st.slider("k-Folds CV (for Training)",2,20,value=st.session_state.k_folds_slider, step=1,key="k_folds_slider", help="Folds for CV on TRAINING set.")

        # Hyperparameters
        st.subheader("⚙️ Hyperparameters");
        optimize_hp=st.checkbox("Enable Grid Search (on Training Data)",value=st.session_state.optimize_check, key="optimize_check");
        if optimize_hp and model_type=="Linear Regression": st.info("Linear Reg no HP."); optimize_hp=False

        log_transform_target_state = st.session_state.log_target_checkbox
        log_transform_features_state = st.session_state.log_features_checkbox
        use_scaling_state = st.session_state.scaling_checkbox
        missing_value_strategy_state = st.session_state.nan_strategy

        grid_model_prefix="regressor__model__" if log_transform_target_state else "model__"; grid_step_prefix="regressor__" if log_transform_target_state else ""
        opt_metric_display=primary_metric; opt_metric_scorer=primary_metric_scorer; opt_metric_short=primary_metric_short
        if optimize_hp:
            with st.expander("Grid Search Opt Metric (Train CV)",True):
                 current_opt_metric = st.session_state.opt_metric_selector
                 if current_opt_metric not in metric_options: current_opt_metric = primary_metric
                 opt_metric_display_gs=st.selectbox("Optimize Metric",metric_options,index=metric_options.index(current_opt_metric), key="opt_metric_selector")
                 opt_metric_display=opt_metric_display_gs; opt_metric_scorer=metric_map[opt_metric_display_gs]; opt_metric_short=metric_disp_key_map[opt_metric_scorer]

        exp_title=f"Configure: {mt_full}";
        if optimize_hp: exp_title+=f" (Optimizing: {opt_metric_short} on Train CV)"
        with st.expander(exp_title,True):
            param_grid={}; poly_check_needed=False; poly_max_degree=1; n_input_features_poly=len(feature_cols)
            # --- Updated Grid Search Parameter Input ---
            try:
                if model_type=="Linear Regression": st.write("No HP.")
                elif model_type=="Ridge":
                    if optimize_hp:
                        alpha_options_r = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                        # Use state for default selection
                        sel_alphas = st.multiselect("Alpha Options (Ridge):", alpha_options_r, default=st.session_state.r_a_gs, key="r_a_gs")
                        if sel_alphas: param_grid[grid_model_prefix+'alpha'] = sorted(sel_alphas)
                        else: st.warning("Select at least one Alpha for Ridge GridSearch.")
                    else: alpha_ridge=st.slider("Alpha",0.01,100.0,step=0.01,key="r_a")
                elif model_type=="Lasso":
                    if optimize_hp:
                        alpha_options_l = [0.001, 0.01, 0.1, 1.0, 10.0]
                        sel_alphas = st.multiselect("Alpha Options (Lasso):", alpha_options_l, default=st.session_state.l_a_gs, key="l_a_gs")
                        if sel_alphas: param_grid[grid_model_prefix+'alpha'] = sorted(sel_alphas)
                        else: st.warning("Select at least one Alpha for Lasso GridSearch.")
                    else: alpha_lasso=st.slider("Alpha",0.01,10.0,step=0.01,key="l_a")
                elif model_type=="Polynomial Regression":
                    poly_check_needed=True; poly_step_name='poly'; poly_degree_key=grid_step_prefix+poly_step_name+'__degree'; model_alpha_key=grid_model_prefix+'alpha'
                    if optimize_hp:
                        degree_opts = [2, 3, 4, 5]
                        sel_degs = st.multiselect("Polynomial Degree(s):", degree_opts, default=st.session_state.p_d_gs, key="p_d_gs")
                        if not sel_degs: st.warning("Select degree(s)"); param_grid[poly_degree_key]=[2]; poly_max_degree=2
                        else: param_grid[poly_degree_key]=sorted(sel_degs); poly_max_degree=max(sel_degs)

                        st.write("Ridge Alpha Options (after Poly):")
                        alpha_options_pr = [0.01, 0.1, 1.0, 10.0, 100.0]
                        sel_alphas_pr = st.multiselect("Alpha Options:", alpha_options_pr, default=st.session_state.p_r_a_gs, key="p_r_a_gs")
                        if sel_alphas_pr: param_grid[model_alpha_key] = sorted(sel_alphas_pr)
                        else: st.warning("Select at least one Alpha for Polynomial+Ridge GridSearch.")
                    else:
                        deg_fixed=st.number_input("Poly Degree",2,10,step=1,key="p_d_num"); poly_max_degree=deg_fixed
                        use_ridge_after_poly=st.checkbox("Use Ridge",key="p_r_c");
                        if use_ridge_after_poly: alpha_poly_ridge=st.slider("Ridge Alpha",0.01,100.0,step=0.01,key="p_r_a")
                elif model_type=="Random Forest":
                    rf_prefix=grid_model_prefix
                    if optimize_hp:
                        st.markdown("**Grid Search Parameters for Random Forest:**")
                        c1, c2 = st.columns(2)
                        with c1:
                            # n_estimators
                            n_est_options = [50, 100, 200, 500, 1000]
                            n_est_selected = st.multiselect("Num Trees (n_estimators):", n_est_options, default=st.session_state.rf_n_est_gs, key="rf_n_est_gs")
                            if n_est_selected: param_grid[rf_prefix+'n_estimators'] = sorted(n_est_selected)

                            # max_depth
                            max_depth_options_num = [5, 10, 20, 30, None]
                            # Ensure default values are strings before passing to multiselect
                            default_max_depth_gs_str = [str(d) for d in st.session_state.rf_max_depth_gs]
                            max_depth_selected_str = st.multiselect("Max Depth (max_depth):", [str(d) for d in max_depth_options_num], default=default_max_depth_gs_str, key="rf_max_depth_gs")
                            parsed_depths = []
                            for d_str in max_depth_selected_str:
                                if d_str == 'None': parsed_depths.append(None)
                                elif d_str.isdigit(): parsed_depths.append(int(d_str))
                            if parsed_depths: param_grid[rf_prefix+'max_depth'] = parsed_depths

                            # max_features
                            max_feat_options = ['sqrt', 'log2', 0.6, 0.8, 1.0]
                            default_max_feat_gs_str = [str(f) for f in st.session_state.rf_max_feat_gs]
                            max_feat_selected_str = st.multiselect("Max Feats (max_features):", [str(f) for f in max_feat_options], default=default_max_feat_gs_str, key="rf_max_feat_gs")
                            parsed_features = []
                            for f_str in max_feat_selected_str:
                                if f_str in ['sqrt', 'log2']: parsed_features.append(f_str)
                                else:
                                    try: f_val = float(f_str); parsed_features.append(f_val)
                                    except ValueError: st.warning(f"Ignoring invalid max_features value: {f_str}")
                            if parsed_features: param_grid[rf_prefix+'max_features'] = parsed_features
                        with c2:
                            # min_samples_split
                            min_split_options = [2, 5, 10, 20]
                            min_split_selected = st.multiselect("Min Samples Split:", min_split_options, default=st.session_state.rf_min_split_gs, key="rf_min_split_gs")
                            if min_split_selected: param_grid[rf_prefix+'min_samples_split'] = sorted(min_split_selected)

                            # min_samples_leaf
                            min_leaf_options = [1, 3, 5, 10]
                            min_leaf_selected = st.multiselect("Min Samples Leaf:", min_leaf_options, default=st.session_state.rf_min_leaf_gs, key="rf_min_leaf_gs")
                            if min_leaf_selected: param_grid[rf_prefix+'min_samples_leaf'] = sorted(min_leaf_selected)

                        st.caption("Select multiple values for each parameter to explore.")
                        if not param_grid: st.warning("Select at least one value for each parameter in GridSearch.")

                    else: # Fixed parameters
                        rf1,rf2=st.columns(2);
                        with rf1: rf_n_estimators=st.number_input("Trees",10,2000,step=10,key="rf_est"); rf_max_d_opt=st.selectbox("Depth",["None",5,10,15,20,30],key="rf_max_d"); rf_max_d_val=None if rf_max_d_opt=="None" else int(rf_max_d_opt); rf_max_f_opt=st.selectbox("Feats",["sqrt","log2","all"],key="rf_max_f"); rf_max_f_val=None if rf_max_f_opt=="all" else rf_max_f_opt
                        with rf2: rf_min_s_split=st.slider("Min Split",2,50,step=1,key="rf_min_s"); rf_min_s_leaf=st.slider("Min Leaf",1,50,step=1,key="rf_min_l")

                elif model_type=="Neural Network (MLP)":
                    mlp_prefix=grid_model_prefix; fixed_iter=500; fixed_early_stop=True; fixed_patience=10
                    if optimize_hp:
                        st.markdown("**Grid Search Parameters for MLP:**")
                        c1,c2=st.columns(2);
                        with c1:
                            # Hidden Layer Sizes (Keep as text input due to complexity)
                            hls_text = st.text_input('Hidden Layer Sizes (csv tuples):', key="mlp_h_g", help="E.g., `(100,)` or `(50,50), (25,25)`")
                            parsed_hls = parse_param_string(hls_text, tuple, [(100,)])
                            if parsed_hls: param_grid[mlp_prefix+'hidden_layer_sizes'] = parsed_hls
                            else: st.warning("Invalid Hidden Layer Sizes format.")

                            # Activation
                            act_options = ["relu", "tanh", "logistic", "identity"]
                            sel_act = st.multiselect("Activation:", act_options, default=st.session_state.mlp_a_gs, key="mlp_a_gs")
                            if sel_act: param_grid[mlp_prefix+'activation'] = sel_act
                            else: st.warning("Select at least one Activation for MLP GridSearch.")

                            # Solver
                            solver_options = ["adam", "sgd", "lbfgs"]
                            sel_solver = st.multiselect("Solver:", solver_options, default=st.session_state.mlp_s_gs, key="mlp_s_gs")
                            if sel_solver: param_grid[mlp_prefix+'solver'] = sel_solver
                            else: st.warning("Select at least one Solver for MLP GridSearch.")

                        with c2:
                            # Alpha (L2 Regularization)
                            alpha_options = [0.00001, 0.0001, 0.001, 0.01, 0.1]
                            sel_alpha = st.multiselect("Alpha (L2):", alpha_options, default=st.session_state.mlp_alpha_gs, key="mlp_alpha_gs")
                            if sel_alpha: param_grid[mlp_prefix+'alpha'] = sorted(sel_alpha)
                            else: st.warning("Select at least one Alpha for MLP GridSearch.")

                            # Learning Rate Init (only for 'sgd' or 'adam')
                            lr_options = [0.0001, 0.001, 0.01, 0.1]
                            sel_lr = st.multiselect("Learning Rate Init:", lr_options, default=st.session_state.mlp_lr_gs, key="mlp_lr_gs")
                            if sel_lr: param_grid[mlp_prefix+'learning_rate_init'] = sorted(sel_lr)
                            # No warning if empty, as it might not be used by 'lbfgs'

                            # Fixed params for Grid Search
                            st.number_input("Max Iterations (fixed for GS):", fixed_iter, disabled=True)
                            st.checkbox("Early Stopping (fixed for GS):", fixed_early_stop, disabled=True)
                            st.number_input("Patience (fixed for GS):", fixed_patience, disabled=True)
                        st.caption("Select multiple values for parameters to explore. Learning rate only used by 'adam' and 'sgd' solvers.")
                        if not param_grid: st.warning("Select parameters to explore in Grid Search.")

                    else: # Fixed parameters
                        c1,c2=st.columns(2);
                        with c1:
                            mlp_h_str=st.text_input("Layers",key="mlp_h"); mlp_h_val=(100,)
                            try: layers=[int(x.strip()) for x in mlp_h_str.split(',') if x.strip().isdigit()]; mlp_h_val=tuple(layers) if layers else (100,)
                            except ValueError: st.error("Bad Layers. Using (100,).")
                            mlp_a_fixed=st.selectbox("Activation",["relu","tanh","logistic","identity"],key="mlp_a")
                            mlp_s_fixed=st.selectbox("Solver",["adam","sgd","lbfgs"],key="mlp_s")
                        with c2:
                            mlp_alpha_fixed=st.number_input("Alpha L2",0.0,0.1,format="%.5f",key="mlp_alpha")
                            mlp_lr_fixed=st.number_input("LR Init",0.00001,0.1,format="%.5f",key="mlp_lr")
                            mlp_max_iter_fixed=st.number_input("Max Iters",50,5000,step=100,key="mlp_iter")
                            mlp_early_stop=st.checkbox("Early Stop",key="mlp_stop")
                            mlp_patience=fixed_patience
                            if mlp_early_stop: mlp_patience=st.number_input("Patience",3,100,step=1,key="mlp_pat")
            except Exception as e: st.error(f"Param setup error: {e}"); param_grid={}

            if poly_check_needed and n_input_features_poly > 0:
                try:
                    max_feats = calculate_poly_features(n_input_features_poly, poly_max_degree)
                    n_samples_approx = len(current_df)
                    if max_feats >= n_samples_approx:
                        st.warning(f"⚠️ Poly degree {poly_max_degree} -> {max_feats} features >= {n_samples_approx} samples?")
                except Exception as pc_e:
                    st.warning(f"Poly check failed: {pc_e}")

        # Run Analysis Button
        st.markdown("---"); st.header("🚀 Run Regression Analysis")
        run_label=f"Run {mt_full} with {k_folds}-Fold CV"; notes=[]
        if log_transform_target_state: notes.append("LogY")
        if log_transform_features_state: notes.append("LogX")
        if use_scaling_state: notes.append("ScaleX")
        if missing_value_strategy_state.startswith("Impute"): notes.append(missing_value_strategy_state)
        if notes: run_label+=f" ({','.join(notes)})"
        # Update can_run_gs check to ensure param_grid is not empty
        can_run_gs = optimize_hp and bool(param_grid)
        if optimize_hp: run_label+=f" [GS Opt:{opt_metric_short}]" if can_run_gs else " [GS Config Err]"
        if optimize_hp and not can_run_gs: st.warning("GridSearch enabled but parameter grid is empty or invalid.")

        if st.button(run_label, key="run_button", disabled=not can_run_analysis):
            st.session_state['run_analysis_clicked'] = True
            st.write("Resetting previous analysis results...")
            for key in analysis_result_keys:
                 if key in st.session_state:
                     if key == 'run_analysis_clicked': st.session_state[key] = True
                     elif key == 'final_estimator_trained': st.session_state[key] = False
                     else: st.session_state[key] = None

            _target=st.session_state.target_selector; _features=st.session_state.feature_selector; _df=st.session_state['df']
            if _target is None or not _features: st.error("Target or Features became unselected before running."); st.stop()
            if _target not in _df.columns or not all(f in _df.columns for f in _features): st.error("Target or Features columns not found in DataFrame."); st.stop()

            cols_reg=[_target]+_features; data_reg = _df[cols_reg].copy()

            # Data Prep
            num_feat_cols=[];
            for col in _features:
                if col in data_reg.columns:
                    if data_reg[col].dtype=='object' or pd.api.types.is_categorical_dtype(data_reg[col].dtype):
                         data_reg[col]=pd.to_numeric(data_reg[col],errors='coerce')
                    if pd.api.types.is_numeric_dtype(data_reg[col].dtype):
                         num_feat_cols.append(col)

            if _target in data_reg.columns:
                 if data_reg[_target].dtype=='object' or pd.api.types.is_categorical_dtype(data_reg[_target].dtype):
                      data_reg[_target]=pd.to_numeric(data_reg[_target],errors='coerce')
                 if not pd.api.types.is_numeric_dtype(data_reg[_target].dtype):
                      st.error(f"Target '{_target}' could not be converted to numeric."); st.stop()
            else:
                 st.error(f"Target column '{_target}' not found."); st.stop()

            rows_init=len(data_reg); rows_dropped=0
            if missing_value_strategy_state=="Drop Rows with NaNs":
                data_reg.dropna(subset=[_target] + _features, inplace=True)
                rows_dropped=rows_init-len(data_reg)
            elif missing_value_strategy_state in ["Impute Mean","Impute Median"]:
                rows_before=len(data_reg); data_reg.dropna(subset=[_target],inplace=True); rows_dropped=rows_before-len(data_reg)

            if rows_dropped>0: st.info(f"Dropped {rows_dropped} rows with NaNs (Target or All, based on selection).")
            if data_reg.empty: st.error("No data left after NaN handling."); st.stop()

            X_full=data_reg[_features].copy(); y_full=data_reg[_target].copy()
            if X_full.empty or y_full.empty: st.error("Data became empty after preparing features/target."); st.stop()

            # Train/Test Split
            test_split_frac = st.session_state.test_split_ratio / 100.0
            try:
                if len(X_full) < 5:
                     st.warning("Dataset too small for split. Using all data for training/testing.")
                     X_train, X_test, y_train, y_test = X_full.copy(), X_full.iloc[0:0], y_full.copy(), y_full.iloc[0:0]
                     test_split_ratio_display = 0.0
                else:
                     X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_split_frac, random_state=42)
                     test_split_ratio_display = test_split_frac
                st.info(f"Data split: {len(X_train)} training samples, {len(X_test)} validation samples (Ratio={test_split_ratio_display:.2f}).")
                st.session_state['X_train'] = X_train; st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train; st.session_state['y_test'] = y_test
                st.session_state['test_split_ratio_run'] = test_split_ratio_display * 100
            except Exception as split_e:
                 st.error(f"Error during data split: {split_e}. Check test split ratio and data size."); st.stop()

            st.session_state['X_reg_global']=X_full.copy(); st.session_state['y_reg_global']=y_full.copy(); st.session_state['feature_cols_global']=_features[:]; st.session_state['target_col_global']=_target; st.session_state['model_type_full_global']=mt_full

            neg = []
            if log_transform_target_state and y_train is not None and not y_train.empty and (y_train < 0).any(): st.warning(f"⚠️ Training Target '{_target}' contains negative values. Using log(1+x).")
            if log_transform_features_state:
                if X_train is not None and not X_train.empty:
                    numeric_cols_in_X_train = X_train.select_dtypes(include=np.number).columns.tolist()
                    neg = [c for c in numeric_cols_in_X_train if (X_train[c] < 0).any()]
                    if neg: st.warning(f"⚠️ Training Features {neg} contain negative values. Using log(1+x).")

            # CV setup
            k_cv = k_folds; n_samples_train = len(X_train) if X_train is not None else 0
            if n_samples_train < k_cv: k_cv=max(2,n_samples_train); st.warning(f"k adjusted to {k_cv} due to train samples ({n_samples_train}).")
            kf=KFold(n_splits=k_cv,shuffle=True,random_state=42)
            st.session_state['k_folds_run'] = k_cv

            # Build pipeline
            steps=[]; st.write("---"); st.write("**Pipeline Steps:**"); log1p_tf=FunctionTransformer(np.log1p,np.expm1,validate=False)
            if log_transform_features_state: steps.append(('logFeat',log1p_tf)); st.write("- Log Features")
            if X_train is not None and X_train.isnull().any().any():
                if missing_value_strategy_state=="Impute Mean": steps.append(('imputeM',SimpleImputer(strategy='mean'))); st.write("- Mean Impute")
                elif missing_value_strategy_state=="Impute Median": steps.append(('imputeMed',SimpleImputer(strategy='median'))); st.write("- Median Impute")
            if use_scaling_state: steps.append(('scale',StandardScaler())); st.write("- Scale Features")

            if model_type == "Polynomial Regression":
                 deg_fixed_state = st.session_state.p_d_num if not optimize_hp else 2
                 poly_deg_pipe = deg_fixed_state if not optimize_hp else (param_grid.get(poly_degree_key, [2])[0])
                 n_feat_in = X_train.shape[1] if X_train is not None else 0
                 max_deg_check = poly_max_degree if optimize_hp else deg_fixed_state
                 if n_feat_in > 0 and n_samples_train > 0:
                     try:
                         n_feat_out = calculate_poly_features(n_feat_in, max_deg_check)
                         st.write(f"(Max degree {max_deg_check} -> ~{n_feat_out} features from {n_feat_in}; {n_samples_train} train samples)")
                         if n_feat_out >= n_samples_train: st.error(f"❌ Ill-Determined: Poly degree {max_deg_check} -> {n_feat_out} features >= {n_samples_train} train samples. Reduce degree."); st.stop()
                     except Exception as poly_e: st.warning(f"Poly check error: {poly_e}")
                 elif n_feat_in == 0:
                     st.error("Cannot apply Polynomial Features: No input features selected or available."); st.stop()
                 steps.append(('poly', PolynomialFeatures(degree=poly_deg_pipe, include_bias=False)))
                 st.write(f"- Poly Features (degree {poly_deg_pipe})")

            # Model Instantiation
            model=None
            try:
                if model_type=="Linear Regression": model=LinearRegression()
                elif model_type=="Ridge":
                    alpha_r = st.session_state.r_a if not optimize_hp else 1.0
                    model=Ridge(alpha=alpha_r, random_state=42)
                elif model_type=="Lasso":
                    alpha_l = st.session_state.l_a if not optimize_hp else 1.0
                    model=Lasso(alpha=alpha_l, max_iter=10000,random_state=42)
                elif model_type=="Polynomial Regression":
                    if optimize_hp: model=Ridge(random_state=42); st.write(f"- Ridge (alpha=GS) after Poly")
                    else:
                        use_ridge_poly_state = st.session_state.p_r_c
                        alpha_poly_r_state = st.session_state.p_r_a
                        model=Ridge(alpha=alpha_poly_r_state, max_iter=10000,random_state=42) if use_ridge_poly_state else LinearRegression()
                        st.write(f"- {'Ridge(alpha=%.4f)'%alpha_poly_r_state if use_ridge_poly_state else 'Linear Reg'} after Poly")
                elif model_type=="Random Forest":
                    rf_pars={'random_state': 42, 'n_jobs': -1}
                    if not optimize_hp:
                        rf_n_estimators_state = st.session_state.rf_est
                        rf_max_d_opt_state = st.session_state.rf_max_d
                        rf_max_d_val_state = None if rf_max_d_opt_state=="None" else int(rf_max_d_opt_state)
                        rf_max_f_opt_state = st.session_state.rf_max_f
                        rf_max_f_val_state = None if rf_max_f_opt_state=="all" else rf_max_f_opt_state
                        rf_min_s_split_state = st.session_state.rf_min_s
                        rf_min_s_leaf_state = st.session_state.rf_min_l
                        rf_pars.update({'n_estimators': rf_n_estimators_state, 'max_depth': rf_max_d_val_state, 'min_samples_split': rf_min_s_split_state, 'min_samples_leaf': rf_min_s_leaf_state, 'max_features': rf_max_f_val_state})
                        st.write("- RF (fixed)")
                    else: st.write("- RF (GS)")
                    model=RandomForestRegressor(**rf_pars)
                elif model_type=="Neural Network (MLP)":
                    mlp_h_str_state = st.session_state.mlp_h
                    mlp_h_val_state = (100,)
                    try: layers_state=[int(x.strip()) for x in mlp_h_str_state.split(',') if x.strip().isdigit()]; mlp_h_val_state=tuple(layers_state) if layers_state else (100,)
                    except ValueError: mlp_h_val_state = (100,)

                    mlp_a_fixed_state = st.session_state.mlp_a
                    mlp_s_fixed_state = st.session_state.mlp_s
                    mlp_alpha_fixed_state = st.session_state.mlp_alpha
                    mlp_lr_fixed_state = st.session_state.mlp_lr
                    mlp_max_iter_fixed_state = st.session_state.mlp_iter
                    mlp_early_stop_state = st.session_state.mlp_stop
                    mlp_patience_state = st.session_state.mlp_pat if mlp_early_stop_state else fixed_patience

                    mlp_pars={'random_state': 42, 'max_iter': mlp_max_iter_fixed_state if not optimize_hp else fixed_iter, 'early_stopping': mlp_early_stop_state if not optimize_hp else fixed_early_stop, 'n_iter_no_change': mlp_patience_state if not optimize_hp else fixed_patience}
                    if not optimize_hp: mlp_pars.update({'hidden_layer_sizes': mlp_h_val_state, 'activation': mlp_a_fixed_state, 'solver': mlp_s_fixed_state, 'alpha': mlp_alpha_fixed_state, 'learning_rate_init': mlp_lr_fixed_state}); st.write("- MLP (fixed)")
                    else: mlp_pars.update({'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001, 'learning_rate_init': 0.001}); st.write("- MLP (GS)")
                    model=MLPRegressor(**mlp_pars)

            except NameError as ne: st.error(f"Config Error: Required variable not defined. Details: {ne}"); st.stop()
            except Exception as e: st.error(f"Model init error: {e}"); st.stop()
            if model is None: st.error("Model could not be instantiated."); st.stop()

            # Final Estimator
            pipe=Pipeline(steps+[('model',model)])
            estimator=TransformedTargetRegressor(regressor=pipe,func=np.log1p,inverse_func=np.expm1) if log_transform_target_state else pipe
            st.write("Final Estimator:"); st.code(str(estimator), language='python')

            # --- Execute CV/GS on TRAINING data ---
            score_map = metric_map
            refit_value_for_gs = opt_metric_display

            scores_cv_dict={}; results_gs=None; best_p=None; final_model_trained = None
            processed_cv_scores = {}; y_pred_cv_train = None; imp_df = None; test_scores = {}; y_pred_test = None

            gs_progress_bar = None
            total_fits = 0
            # Re-check can_run_gs after param_grid might have been populated
            can_run_gs = optimize_hp and bool(param_grid)
            if can_run_gs:
                try:
                    param_lengths = [len(v) for v in param_grid.values() if isinstance(v, list)]
                    n_candidates = prod(param_lengths) if param_lengths else 1
                    total_fits = n_candidates * k_cv
                    progress_text = f"Running GridSearch... (≈ {total_fits} model fits)"
                    st.info(progress_text)
                    gs_progress_bar = st.progress(0, text=progress_text)
                except Exception as e:
                    st.warning(f"Could not estimate GridSearch progress: {e}")
                    can_run_gs = False # Disable GS if estimation fails

            with st.spinner(f"⏳ Running {run_label} on Training Data..."):
                t0=time.time()
                try:
                    if X_train is None or y_train is None or X_train.empty or y_train.empty:
                        st.error("Cannot run analysis: Training data is empty."); st.stop()

                    if can_run_gs:
                        st.write(f"Executing GridSearch (Refit: {opt_metric_display})...")
                        gs = GridSearchCV(estimator, param_grid, scoring=score_map, refit=refit_value_for_gs, cv=kf, n_jobs=-1)
                        gs.fit(X_train, y_train)
                        if gs_progress_bar: gs_progress_bar.progress(1.0, text=f"GridSearch Complete (≈ {total_fits} fits).")

                        final_model_trained = gs.best_estimator_
                        st.session_state['final_estimator']=final_model_trained
                        best_p=gs.best_params_
                        results_gs=pd.DataFrame(gs.cv_results_)
                        st.session_state['grid_search_results_df'] = results_gs
                        best_idx=gs.best_index_
                        scores_cv_dict=results_gs.loc[best_idx].to_dict()
                        st.success(f"✅ GS done ({time.time()-t0:.2f}s). Best params:"); st.json(best_p)
                    else:
                        if gs_progress_bar: gs_progress_bar.empty()

                        st.write(f"Running {k_cv}-fold CV on Training Data...");
                        scores_cv_dict=cross_validate(estimator,X_train,y_train,cv=kf,scoring=score_map,n_jobs=-1, return_train_score=False)
                        st.write("Fitting final model on Training Data...")
                        estimator.fit(X_train, y_train)
                        final_model_trained = estimator
                        st.session_state['final_estimator']=final_model_trained
                        st.success(f"✅ CV and Final Fit done ({time.time()-t0:.2f}s).")
                    st.session_state['final_estimator_trained']=True

                    if scores_cv_dict:
                        metric_disp_key = metric_disp_key_map
                        for score_display_name, scorer_key in metric_map.items():
                            short_name = metric_disp_key[scorer_key]; test_score_key = f'test_{scorer_key}'; mean_score_key = f'mean_test_{scorer_key}'; std_score_key = f'std_test_{scorer_key}'
                            score_values_raw = None; is_mean_score = False
                            if test_score_key in scores_cv_dict: score_values_raw = scores_cv_dict[test_score_key]
                            elif mean_score_key in scores_cv_dict: score_values_raw = scores_cv_dict[mean_score_key]; is_mean_score = True
                            if score_values_raw is not None:
                                scores_array = np.array([score_values_raw]) if is_mean_score or isinstance(score_values_raw, (int, float)) else (np.array(score_values_raw) if isinstance(score_values_raw, (list, np.ndarray)) else np.array([]))
                                valid_scores = scores_array[np.isfinite(scores_array)]
                                if scorer_key.startswith('neg_'): valid_scores = -valid_scores
                                if len(valid_scores)>0: mean_val = np.mean(valid_scores); std_val = np.std(valid_scores) if not is_mean_score and len(valid_scores) > 1 else (scores_cv_dict.get(std_score_key, np.nan) if is_mean_score else np.nan); processed_cv_scores[short_name]={'mean':mean_val,'std':std_val,'values':valid_scores}
                                else: processed_cv_scores[short_name]={'mean':np.nan,'std':np.nan,'values':np.array([])}
                            else: processed_cv_scores[short_name]={'mean':np.nan,'std':np.nan,'values':np.array([])}
                        if 'RMSE' not in processed_cv_scores and 'MAE' in processed_cv_scores:
                            mse_key_neg='neg_mean_squared_error'; mse_test_key_cv=f'test_{mse_key_neg}'; mse_mean_key_gs=f'mean_test_{mse_key_neg}'; mse_std_key_gs = f'std_test_{mse_key_neg}'
                            mean_rmse, std_rmse, rmse_values_calc = np.nan, np.nan, np.array([])
                            if mse_test_key_cv in scores_cv_dict:
                                mse_scores_raw = scores_cv_dict[mse_test_key_cv]; valid_neg_mse = np.array(mse_scores_raw)[np.isfinite(mse_scores_raw)] if isinstance(mse_scores_raw, (list,np.ndarray)) else np.array([])
                                if len(valid_neg_mse)>0: valid_mse=-valid_neg_mse; rmse_values_calc=np.sqrt(np.maximum(0,valid_mse)); mean_rmse=np.mean(rmse_values_calc); std_rmse=np.std(rmse_values_calc) if len(rmse_values_calc)>1 else np.nan
                            elif mse_mean_key_gs in scores_cv_dict:
                                mean_neg_mse = scores_cv_dict[mse_mean_key_gs]
                                if not np.isnan(mean_neg_mse): mean_mse=-mean_neg_mse; mean_rmse=np.sqrt(np.maximum(0,mean_mse)); std_rmse=np.nan; rmse_values_calc=np.array([mean_rmse])
                            if not np.isnan(mean_rmse): processed_cv_scores['RMSE']={'mean':mean_rmse,'std':std_rmse,'values':rmse_values_calc}
                    st.session_state['cv_results_processed'] = processed_cv_scores

                    st.write("Evaluating final model on Validation Set...")
                    if final_model_trained and X_test is not None and y_test is not None and len(X_test) > 0:
                        try:
                            y_pred_test = final_model_trained.predict(X_test)
                            test_r2 = r2_score(y_test, y_pred_test); test_mae = mean_absolute_error(y_test, y_pred_test); test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                            test_scores = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
                            st.session_state['y_pred_test'] = y_pred_test
                        except Exception as test_e: st.error(f"Validation eval error: {test_e}"); test_scores = {}
                    elif len(X_test) == 0: st.info("Validation set empty, skipping validation evaluation."); test_scores = {}
                    else: st.warning("Could not evaluate on validation set (model or data missing)."); test_scores = {}
                    st.session_state['test_results'] = test_scores

                    st.write("Generating CV predictions on Training Data...");
                    if final_model_trained and X_train is not None and y_train is not None and not X_train.empty and not y_train.empty:
                        try: y_pred_cv_train = cross_val_predict(final_model_trained,X_train,y_train,cv=kf,n_jobs=-1)
                        except Exception as cvp_e: st.error(f"❌ CV prediction error on train data: {cvp_e}"); y_pred_cv_train = None
                    else: st.warning("Model or train data unavailable for CV preds."); y_pred_cv_train = None
                    st.session_state['y_pred_cv_train'] = y_pred_cv_train

                    if final_model_trained and X_train is not None and not X_train.empty:
                        _est_imp = final_model_trained; _X_imp_train = X_train; _mt_imp = st.session_state.get('model_type_full_global');
                        st.write(f"Calculating Feature Importance/Coefficients...")
                        try:
                            pipe_imp = _est_imp.regressor_ if isinstance(_est_imp, TransformedTargetRegressor) else _est_imp
                            final_model_imp_step = pipe_imp.named_steps['model']
                            feat_names_in = list(_X_imp_train.columns); feat_names_out = feat_names_in
                            current_transformer_names = feat_names_in
                            for step_name, step_transformer in pipe_imp.steps[:-1]:
                                try:
                                    if hasattr(step_transformer, 'get_feature_names_out'):
                                        current_transformer_names = list(step_transformer.get_feature_names_out(input_features=current_transformer_names))
                                    elif isinstance(step_transformer, PCA):
                                        if hasattr(step_transformer, 'n_components_') and step_transformer.n_components_ is not None:
                                            n_pca_comps = step_transformer.n_components_
                                        elif hasattr(step_transformer, 'n_features_in_'):
                                            n_pca_comps = min(step_transformer.n_features_in_, len(current_transformer_names))
                                            st.warning(f"PCA step '{step_name}': n_components_ not found, estimating based on input features ({n_pca_comps}).")
                                        else:
                                             n_pca_comps = 0
                                             st.warning(f"PCA step '{step_name}': Cannot determine number of components.")
                                        if n_pca_comps > 0:
                                            current_transformer_names = [f"PC{i+1}" for i in range(n_pca_comps)]
                                except Exception as name_err: st.warning(f"Feature name error in step '{step_name}': {name_err}. Using previous names.");
                            feat_names_out = current_transformer_names

                            imp_values = None; imp_type = None
                            if hasattr(final_model_imp_step, 'coef_') and model_type != "Neural Network (MLP)": imp_values = final_model_imp_step.coef_.flatten(); imp_type = 'Coefficient'
                            elif hasattr(final_model_imp_step, 'feature_importances_'): imp_values = final_model_imp_step.feature_importances_; imp_type = 'Importance'
                            if imp_values is not None:
                                if len(imp_values) == len(feat_names_out): imp_df=pd.DataFrame({'Feature':feat_names_out, imp_type:imp_values}).sort_values(imp_type,key=abs,ascending=False); st.session_state['importance_df'] = imp_df.copy()
                                else: st.warning(f"⚠️ Feature Importance/Coefficient length mismatch: Model expects {len(imp_values)} features, but pipeline output {len(feat_names_out)} feature names ('{feat_names_out[0]}'...). Importance results may be unreliable."); imp_df = None
                            else: st.info(f"Model '{_mt_imp}' does not provide standard coefficient or feature importance attributes."); imp_df = None
                        except Exception as e: st.error(f"Importance/Coef error: {e}"); st.code(traceback.format_exc(), language='python'); imp_df = None
                    else: st.info("Requires trained model and training data for importance.")

                except Exception as e:
                    st.error(f"Run error: {e}")
                    st.code(traceback.format_exc(), language='python')
                    st.session_state['final_estimator_trained'] = False
                    if gs_progress_bar: gs_progress_bar.empty()

            # --- Display results AFTER button press logic (if successful) ---
            if st.session_state.get('final_estimator_trained'):
                try:
                    if gs_progress_bar: gs_progress_bar.empty()

                    cv_scores_display = st.session_state.get('cv_results_processed', {})
                    test_scores_display = st.session_state.get('test_results', {})
                    k_folds_run = st.session_state.get('k_folds_run', k_folds)
                    test_split_ratio_run = st.session_state.get('test_split_ratio_run', 0.0)
                    y_pred_cv_train_plot = st.session_state.get('y_pred_cv_train')
                    y_train_plot = st.session_state.get('y_train')
                    y_pred_test_plot = st.session_state.get('y_pred_test')
                    y_test_plot = st.session_state.get('y_test')
                    imp_df_display = st.session_state.get('importance_df')
                    results_gs_display = st.session_state.get('grid_search_results_df')
                    _mt_imp = st.session_state.get('model_type_full_global', 'Model')
                    primary_metric_short_disp = primary_metric_short

                    # --- Performance Metrics Display ---
                    st.subheader(f"📊 Performance Results")
                    col_cv, col_test = st.columns(2)
                    with col_cv:
                        st.markdown(f"**Training Set CV Results ({k_folds_run}-Fold Avg ± Std)**")
                        if cv_scores_display:
                            for mkey in ['R²', 'MAE', 'RMSE']:
                                mdata = cv_scores_display.get(mkey)
                                if mdata and not np.isnan(mdata['mean']):
                                    std_str = f" ± {mdata['std']:.4f}" if not np.isnan(mdata['std']) else ""
                                    st.markdown(f"**{mkey}:** {mdata['mean']:.4f}{std_str}")
                                else:
                                    st.markdown(f"**{mkey}:** N/A")
                        else: st.write("No CV metrics calculated.")
                    with col_test:
                        st.markdown(f"**Validation Set Results ({test_split_ratio_run:.0f}% Hold-out)**")
                        if test_scores_display:
                             test_r2 = test_scores_display.get('R²', np.nan); test_mae = test_scores_display.get('MAE', np.nan); test_rmse = test_scores_display.get('RMSE', np.nan)
                             st.markdown(f"{'**' if primary_metric_short_disp == 'R²' else ''}Validation R²:{'**' if primary_metric_short_disp == 'R²' else ''} {test_r2:.4f}" if not np.isnan(test_r2) else "**Validation R²:** N/A")
                             st.markdown(f"{'**' if primary_metric_short_disp == 'MAE' else ''}Validation MAE:{'**' if primary_metric_short_disp == 'MAE' else ''} {test_mae:.4f}" if not np.isnan(test_mae) else "**Validation MAE:** N/A")
                             st.markdown(f"{'**' if primary_metric_short_disp == 'RMSE' else ''}Validation RMSE:{'**' if primary_metric_short_disp == 'RMSE' else ''} {test_rmse:.4f}" if not np.isnan(test_rmse) else "**Validation RMSE:** N/A")
                        elif len(st.session_state.get('X_test', pd.DataFrame())) == 0:
                             st.write("Validation set was empty.")
                        else:
                             st.write("No validation metrics calculated (e.g., prediction error).")

                    # --- Visualizations ---
                    st.subheader("📉 Visualizations: Actual vs. Predicted")
                    plot_data_list = []
                    if y_pred_cv_train_plot is not None and y_train_plot is not None and len(y_train_plot) > 0:
                        valid_train = np.isfinite(y_pred_cv_train_plot) & np.isfinite(y_train_plot)
                        if valid_train.sum() > 0: plot_data_list.append(pd.DataFrame({'Actual': y_train_plot[valid_train], 'Predicted': y_pred_cv_train_plot[valid_train], 'Data Source': 'Train (CV)'}))
                    if y_pred_test_plot is not None and y_test_plot is not None and len(y_test_plot)>0:
                        valid_test = np.isfinite(y_pred_test_plot) & np.isfinite(y_test_plot)
                        if valid_test.sum() > 0: plot_data_list.append(pd.DataFrame({'Actual': y_test_plot[valid_test], 'Predicted': y_pred_test_plot[valid_test], 'Data Source': 'Validation'}))

                    if plot_data_list:
                        plot_df_combined = pd.concat(plot_data_list, ignore_index=True)
                        if not plot_df_combined.empty:
                            fig_combined, ax_combined = plt.subplots(figsize=(6, 6))
                            sns.scatterplot(data=plot_df_combined, x='Actual', y='Predicted', hue='Data Source', style='Data Source', alpha=0.6, s=40, ax=ax_combined)
                            actual_min, actual_max = plot_df_combined['Actual'].min(), plot_df_combined['Actual'].max()
                            pred_min, pred_max = plot_df_combined['Predicted'].min(), plot_df_combined['Predicted'].max()
                            if pd.isna(actual_min) or pd.isna(pred_min) or pd.isna(actual_max) or pd.isna(pred_max):
                                st.warning("NaN values found in plot data, limits might be inaccurate.")
                                lim_min, lim_max = -1, 1
                            else:
                                lim_min = min(actual_min, pred_min); lim_max = max(actual_max, pred_max)
                            if lim_min == lim_max: padding = 1.0
                            else: padding = (lim_max - lim_min) * 0.05
                            plot_min = lim_min - padding; plot_max = lim_max + padding

                            ax_combined.plot([plot_min, plot_max], [plot_min, plot_max], '--r', lw=2, label='Ideal (y=x)')
                            ax_combined.set_xlabel("Actual Value"); ax_combined.set_ylabel("Predicted Value"); ax_combined.set_title("Actual vs. Predicted"); ax_combined.legend(); ax_combined.grid(True); ax_combined.set_xlim(plot_min, plot_max); ax_combined.set_ylim(plot_min, plot_max); ax_combined.set_aspect('equal', adjustable='box'); plt.tight_layout(); st.pyplot(fig_combined)
                        else: st.warning("No valid data points found for combined plot after filtering.")
                    else: st.warning("Prediction data unavailable or invalid for plotting.")

                    # --- Feature Importance ---
                    st.markdown("---"); st.subheader("✨ Feature Importance / Coefficients")
                    if imp_df_display is not None and not imp_df_display.empty:
                        imp_type = 'Coefficient' if 'Coefficient' in imp_df_display.columns else 'Importance'
                        st.write(f"Displaying {imp_type}s (based on model trained with Training Data).")
                        n_plot=min(20,len(imp_df_display));
                        if n_plot>0: fig,ax=plt.subplots(figsize=(7,max(4,n_plot*0.3))); sns.barplot(x=imp_type,y='Feature',data=imp_df_display.head(n_plot).iloc[::-1],ax=ax,palette='viridis'); ax.set_title(f'Top {n_plot} Feature {imp_type}s ({_mt_imp})'); plt.tight_layout(); st.pyplot(fig)
                        st.dataframe(imp_df_display.round(4))
                        try: csv_i=imp_df_display.to_csv(index=False).encode('utf-8-sig'); st.download_button("⬇️ DL Import/Coef",csv_i,f'{_mt_imp}_imp.csv','text/csv',key='dl_i')
                        except Exception as e: st.error(f"DL prep error (Imp): {e}")
                    else: st.info("No importance/coefficient data available or calculated.")

                    # Display Grid Search Results
                    if results_gs_display is not None:
                        with st.expander("🔍 Detailed Grid Search Results (from Training CV)", False):
                             try:
                                 opt_metric_disp_gs = st.session_state.get('opt_metric_display', primary_metric)
                                 opt_metric_scorer_gs = st.session_state.get('opt_metric_scorer', primary_metric_scorer)
                                 st.write(f"GS explored {len(results_gs_display)} combinations. Ranked by {opt_metric_disp_gs} (`{opt_metric_scorer_gs}`).")
                                 cols_s=['params']; rank_c=f'rank_test_{opt_metric_scorer_gs}'
                                 if rank_c not in results_gs_display.columns and f'mean_test_{opt_metric_scorer_gs}' in results_gs_display.columns:
                                     rank_c = f'mean_test_{opt_metric_scorer_gs}'
                                     st.warning(f"Rank column 'rank_test_{opt_metric_scorer_gs}' missing, ranking by '{rank_c}'.")
                                 elif rank_c not in results_gs_display.columns:
                                     st.error(f"Cannot rank GS results: Neither 'rank_test_{opt_metric_scorer_gs}' nor 'mean_test_{opt_metric_scorer_gs}' found.")
                                     st.dataframe(results_gs_display)
                                     raise ValueError("Missing ranking column in GS results")

                                 for d_name, s_key in metric_map.items():
                                     m_c,s_c=f'mean_test_{s_key}',f'std_test_{s_key}';
                                     if m_c in results_gs_display: cols_s.append(m_c)
                                     if s_c in results_gs_display: cols_s.append(s_c)
                                 if rank_c in results_gs_display and rank_c not in cols_s: cols_s.append(rank_c)
                                 if 'mean_fit_time' in results_gs_display: cols_s.append('mean_fit_time')
                                 if 'mean_score_time' in results_gs_display: cols_s.append('mean_score_time')

                                 res_df_disp=results_gs_display[cols_s].copy(); res_df_disp['params_str']=res_df_disp['params'].astype(str)
                                 rename_d={}; final_cols=['params_str']
                                 for col in res_df_disp.columns:
                                     if col in ['params','params_str']: continue
                                     new_n,neg=col,False;
                                     if col.startswith(('mean_test_neg_','std_test_neg_','rank_test_neg_')):
                                         if not col.startswith('rank_'):
                                             res_df_disp[col]=-res_df_disp[col];
                                         neg=True
                                     s_key_part = col.split('_test_')[-1]
                                     prefix = col.split('_test_')[0]+'_' if '_test_' in col else col.split('_')[0]+'_'
                                     orig_s_key = f"neg_{s_key_part}" if neg else s_key_part
                                     short_m = metric_disp_key_map.get(orig_s_key)
                                     if short_m:
                                         new_n = prefix.replace('_test_','_') + short_m
                                         rename_d[col]=new_n
                                     final_cols.append(rename_d.get(col,col))

                                 res_df_disp.rename(columns=rename_d,inplace=True);
                                 rank_c_renamed = rename_d.get(rank_c, rank_c);

                                 st.write(f"(Ranked by: {rank_c_renamed})")
                                 sort_asc = True
                                 if 'R²' in rank_c_renamed: sort_asc = False

                                 if rank_c_renamed not in res_df_disp.columns: st.error(f"Sort column '{rank_c_renamed}' missing after renaming.")
                                 else: res_df_disp=res_df_disp.sort_values(by=rank_c_renamed,ascending=sort_asc)

                                 final_cols_exist=[c for c in final_cols if c in res_df_disp.columns]
                                 [final_cols_exist.append(c) for c in res_df_disp.columns if c not in final_cols_exist]
                                 st.dataframe(res_df_disp[final_cols_exist].round(5)); st.caption("Lower MAE/RMSE better. Higher R² better. Times in sec.")
                                 mt_file_part = st.session_state.get('model_type_full_global', 'model')
                                 csv_gs=res_df_disp[final_cols_exist].to_csv(index=False).encode('utf-8-sig'); st.download_button("⬇️ DL GridSearch Results",csv_gs,f'{mt_file_part}_gridsearch.csv','text/csv',key='dl_gs')
                             except Exception as e: st.error(f"GS display error: {e}"); st.dataframe(results_gs_display)

                except Exception as e:
                    st.error(f"Error displaying results: {e}")
                    st.code(traceback.format_exc(), language='python')


    # --- Sections displayed only AFTER successful analysis run ---
    if st.session_state.get('final_estimator_trained'):
        # --- Model Download Section ---
        st.markdown("---")
        st.subheader("💾 Download Trained Model")
        model_dl = st.session_state.get('final_estimator')
        mt_dl = st.session_state.get('model_type_full_global', 'model')
        X_train_dl = st.session_state.get('X_train')
        if model_dl:
            train_samples_count = len(X_train_dl) if X_train_dl is not None else 'N/A'
            st.write(f"Download the final trained **{mt_dl}** model pipeline (trained on {train_samples_count} samples).")
            try:
                b = io.BytesIO(); joblib.dump(model_dl, b); b.seek(0)
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M'); safe_m = "".join(c if c.isalnum() else "_" for c in mt_dl); fn=f"trained_{safe_m}_{ts}.joblib"
                st.download_button(f"⬇️ Download Model ({fn})", b, fn, "application/octet-stream", key='dl_mod')
                st.caption("Load using ~~~joblib.load()~~~.")
            except Exception as e: st.error(f"Model DL error: {e}")
        else: st.warning("Trained model object not found.")

        # --- LHS Simulation Section ---
        st.markdown("---"); st.header("🔬 LHS Simulation")
        if not SCIPY_AVAILABLE: st.info("LHS requires SciPy (`pip install scipy`).")
        else:
            est_lhs=st.session_state.get('final_estimator'); X_full_lhs=st.session_state.get('X_reg_global'); tgt_lhs=st.session_state.get('target_col_global'); mt_lhs=st.session_state.get('model_type_full_global'); df_lhs=st.session_state.get('df')
            pca_scaler_lhs=st.session_state.get('pca_scaler'); pca_mod_lhs=st.session_state.get('pca_model'); pca_cols_lhs=st.session_state.get('numeric_cols_pca'); pca_load_lhs=st.session_state.get('pca_loadings_matrix'); pca_var_lhs=st.session_state.get('explained_variance_ratio_pca'); pca_scores_orig_lhs=st.session_state.get('pca_scores_global')
            pca_components_exist_lhs = all(comp is not None for comp in [pca_scaler_lhs, pca_mod_lhs, pca_cols_lhs, pca_load_lhs, pca_var_lhs, pca_scores_orig_lhs])

            if not all([est_lhs, X_full_lhs is not None, tgt_lhs, mt_lhs, df_lhs is not None]): st.error("LHS needs previous run info (model, data, target).")
            else:
                st.write("Generate simulated data & predict.");
                model_features_lhs = st.session_state.get('feature_cols_global', [])
                numeric_model_features = [f for f in model_features_lhs if f in df_lhs.select_dtypes(include=np.number).columns]
                non_num_model_features = [f for f in model_features_lhs if f not in numeric_model_features]

                if not numeric_model_features: st.warning("No numeric features used in the final model for LHS.")
                else:
                    st.write(f"**LHS Features (Numeric only):** `{', '.join(numeric_model_features)}`");
                    if non_num_model_features: st.caption(f"Non-numeric features used by model (will use NaN for simulation): `{', '.join(non_num_model_features)}`")

                    lhs_n=st.number_input("LHS Samples",100,100000, step=100, key="lhs_n")
                    if st.button("🚀 Run LHS & Predict", key="run_lhs"):
                        st.session_state['lhs_results_df']=None
                        with st.spinner(f"⏳ Running LHS ({lhs_n})..."):
                             try:
                                 bounds=df_lhs[numeric_model_features].agg(['min','max']); lb,ub=bounds.loc['min'].values,bounds.loc['max'].values; dim=len(numeric_model_features)
                                 mask=lb>=ub;
                                 if np.any(mask): st.warning("Adjusting bounds where min >= max."); adj=np.maximum(1e-6,0.01*np.abs(lb[mask])); lb[mask]-=adj; ub[mask]+=adj;
                                 if np.any(lb>=ub): st.error("Invalid bounds after adjustment."); st.stop()

                                 try: sampler=qmc.LatinHypercube(d=dim,optimization="random-cd",seed=42);
                                 except TypeError: st.warning("Using LHS default optimization (older SciPy?)."); sampler=qmc.LatinHypercube(d=dim,seed=42)

                                 s01=sampler.random(n=lhs_n); scaled=qmc.scale(s01,lb,ub); lhs_in_numeric=pd.DataFrame(scaled,columns=numeric_model_features)
                                 lhs_in_full = lhs_in_numeric.copy()
                                 for col in non_num_model_features: lhs_in_full[col]=np.nan

                                 model_expected_features_order = st.session_state.get('feature_cols_global', [])
                                 try: lhs_in_ordered=lhs_in_full[model_expected_features_order]
                                 except KeyError as e: st.error(f"LHS column mismatch error during reordering: {e}. Model expected: {model_expected_features_order}"); st.stop()

                                 preds=est_lhs.predict(lhs_in_ordered);
                                 res=lhs_in_numeric.copy(); pred_col=f'{tgt_lhs}_predicted'; res[pred_col]=preds;
                                 st.session_state['lhs_results_df']=res.copy(); st.success("LHS Done.")
                             except Exception as e: st.error(f"LHS error: {e}"); st.code(traceback.format_exc(),language='python')

                if st.session_state.get('lhs_results_df') is not None:
                    res_lhs=st.session_state['lhs_results_df']; pred_col=f'{tgt_lhs}_predicted'
                    st.subheader("LHS Simulation Results"); st.write(f"Top/Bottom 10 Predictions:"); st.markdown("##### Top 10"); st.dataframe(res_lhs.nlargest(10,pred_col).round(4)); st.markdown("##### Bottom 10"); st.dataframe(res_lhs.nsmallest(10,pred_col).round(4))
                    try: csv_l=res_lhs.to_csv(index=False).encode('utf-8-sig'); st.download_button("⬇️ DL LHS Results",csv_l,f'lhs_{mt_lhs}_{tgt_lhs}_{lhs_n}.csv','text/csv',key='dl_lhs')
                    except Exception as e: st.error(f"DL prep error (LHS): {e}")

                    st.subheader("🔬 Project LHS Samples onto Original PCA Biplot")
                    if not pca_performed or not pca_components_exist_lhs: st.warning("Original PCA components/scores missing or PCA was not performed. Cannot project LHS samples.")
                    else:
                        pca_cols_set_lhs = set(st.session_state.get('numeric_cols_pca', []))
                        lhs_numeric_cols_set_lhs = set(numeric_model_features)
                        features_match_pca_lhs = pca_cols_set_lhs.issubset(lhs_numeric_cols_set_lhs)

                        if not features_match_pca_lhs: st.warning(f"PCA features ({list(pca_cols_set_lhs)}) not a subset of LHS numeric features ({list(lhs_numeric_cols_set_lhs)}). Cannot project accurately.");
                        else:
                            st.write("Projecting top/bottom 10 predicted LHS samples over original data (grey)...")
                            try:
                                n_ext=10; plot_all=False; res_lhs_plot=st.session_state['lhs_results_df']; pred_col_plot=f'{tgt_lhs}_predicted'
                                if len(res_lhs_plot)<n_ext*2: st.warning(f"< {n_ext*2} LHS samples, plotting all."); ext_idx=res_lhs_plot.index; plot_all=True
                                else: top_idx=res_lhs_plot.nlargest(n_ext,pred_col_plot).index; bot_idx=res_lhs_plot.nsmallest(n_ext,pred_col_plot).index; ext_idx=top_idx.union(bot_idx)

                                lhs_res_filt=res_lhs_plot.loc[ext_idx];
                                pca_cols_list_lhs = st.session_state.numeric_cols_pca
                                lhs_pca_in_filt=lhs_res_filt[pca_cols_list_lhs]

                                pca_scaler_proj = st.session_state.pca_scaler
                                pca_model_proj = st.session_state.pca_model

                                lhs_sc_filt=pca_scaler_proj.transform(lhs_pca_in_filt)
                                lhs_scores_filt=pca_model_proj.transform(lhs_sc_filt)

                                preds_filt=lhs_res_filt[pred_col_plot]; color_cat_filt=pd.Series('Other',index=preds_filt.index)
                                if not plot_all: color_cat_filt[preds_filt.index.isin(top_idx)]='Highest 10'; color_cat_filt[preds_filt.index.isin(bot_idx)]='Lowest 10'; color_lbl=f"Pred '{tgt_lhs}' Extremes"
                                else: color_cat_filt='All LHS'; color_lbl='All LHS Samples'

                                fig_lhs_b=biplot(score=lhs_scores_filt, coeff=st.session_state.pca_loadings_matrix, labels=pca_cols_list_lhs,
                                                 explained_variance_ratio=st.session_state.explained_variance_ratio_pca,
                                                 color_data=color_cat_filt, color_label=color_lbl, is_categorical=True, point_alpha=0.7,
                                                 make_square=True, background_score=st.session_state.pca_scores_global)
                                st.pyplot(fig_lhs_b); caption="Top 10 highest/lowest predicted LHS samples" if not plot_all else "All LHS samples"; st.caption(f"{caption} (colored) over original data (grey). Arrows: PCA loadings.")
                            except KeyError as ke: st.error(f"LHS Biplot Error: Column mismatch. Needed PCA columns '{pca_cols_list_lhs}', but LHS results might be missing some. Error: {ke}")
                            except ValueError as ve: st.error(f"LHS Biplot Error (Value): {ve}"); st.code(traceback.format_exc(), language='python')
                            except Exception as e: st.error(f"LHS Biplot Error: {e}"); st.code(traceback.format_exc(), language='python')

# --- Footer / Initial State ---
if st.session_state.get('df') is None:
     st.info("☝️ **Welcome! Upload a CSV file using the sidebar to begin.**")

# --- Final Sidebar Section (Updated) ---
st.sidebar.markdown("---"); st.sidebar.subheader("📖 How to Use");
st.sidebar.info("""
1.  **Upload CSV:** Select your data file (Sidebar).
2.  **Encoding:** Choose error handling if needed (Sidebar).
3.  **Target & Features:** Select Y variable in the main area. All other columns will be automatically selected as X (Features). You can modify the feature selection manually if needed. Feature selection updates instantly when Target changes.
4.  **Data Split:** Define validation set size (Sidebar).
5.  **PCA Settings:** Configure PCA options (main area, under PCA header). PCA runs automatically after target selection.
6.  **Preprocessing:** Choose NaN handling, Log transforms, Scaling (Sidebar).
7.  **Model & Setup:** Select model, metric, CV folds (default 10), and configure hyperparameters (or enable Grid Search with improved parameter selection) in the main area.
8.  **Run Analysis:** Click the 'Run Analysis' button (enabled when Target & Features are valid). This performs train/test split, CV/GridSearch (train set), fits the final model (train set), and evaluates (test set). A progress bar provides an estimate for GridSearch.
9.  **Results:** View Train CV & Validation metrics, plots, importance in the main area. Download model & results.
10. **LHS Simulation:** (Optional, requires SciPy) Generate simulated data and predict outcomes using the final model (main area).
""")