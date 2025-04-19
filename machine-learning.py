# -*- coding: utf-8 -*-
# ↑ BOM(Byte Order Mark)がないことを確認してください。

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV, cross_validate # cross_validate を使用
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
from math import comb # 組み合わせ計算用に追加
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
Reading the CSV directly using pandas with selected encoding and error handling.
**Encoding errors?** Ensure 'Ignore' or 'Replace' is selected in the sidebar. If errors *still* persist, the CSV file is likely corrupted.
**PCA Target Handling:** Choose whether to exclude the selected target variable from the PCA calculation (default: exclude).
**Log Transformation:** Optionally apply log(1+x) transformation to features and/or target variable in the preprocessing step.
**Evaluation Metric:** Choose the primary metric (R², MAE, RMSE) for evaluation and Grid Search optimization.
**Hyperparameter Optimization:** Enable Grid Search in the setup section to find the best model parameters based on the chosen metric. **Polynomial degree check added to prevent ill-determined systems.**
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
    if len(bg_xs) > 0 and len(bg_ys) > 0:
        ax.scatter(bg_xs, bg_ys, alpha=background_alpha, s=30, color=background_color, label='_nolegend_')
        background_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=background_color, markersize=8, label='Original Data (PCA)', alpha=0.8)

    # Process foreground
    foreground_legend_handles = []
    if color_data is not None and len(xs) == len(color_data):
        color_data_clean = color_data.copy()
        if not isinstance(color_data_clean, (pd.Series, np.ndarray)): color_data_clean = pd.Series(color_data_clean)
        is_numeric_input = pd.api.types.is_numeric_dtype(color_data_clean.dtype) and not pd.api.types.is_bool_dtype(color_data_clean.dtype)
        is_cat_final = is_categorical or not is_numeric_input

        if is_numeric_input and not is_categorical:
            cmap = plt.get_cmap('viridis')
            scatter_kwargs['c'] = color_data_clean; scatter_kwargs['cmap'] = cmap
        else: # Categorical
            is_cat_final = True
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
    if len(xs) > 0 and len(ys) > 0: scatter = ax.scatter(xs, ys, **scatter_kwargs)

    # Combine legends
    legend_visible = False; final_legend_handles = []
    if background_legend: final_legend_handles.append(background_legend)
    if foreground_legend_handles: final_legend_handles.extend(foreground_legend_handles)
    if final_legend_handles:
        ax.legend(handles=final_legend_handles, title=color_label, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
        legend_visible = True
    elif not is_cat_final and color_data is not None and scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, label=color_label, pad=0.02, shrink=0.8); legend_visible = True

    # Plot loadings
    all_xs = np.concatenate((xs, bg_xs)) if len(bg_xs)>0 else xs; all_ys = np.concatenate((ys, bg_ys)) if len(bg_ys)>0 else ys
    if coeff is not None and len(all_xs) > 0 and len(all_ys) > 0:
        x_range = all_xs.max() - all_xs.min() if len(all_xs) > 1 else 1; y_range = all_ys.max() - all_ys.min() if len(all_ys) > 1 else 1
        arrow_scale = max(x_range, y_range) * 0.4 if max(x_range, y_range) > 0 else 1; arrow_scale = max(arrow_scale, 0.1)
        for i in range(n_loadings):
            if pc_x < coeff.shape[0] and pc_y < coeff.shape[0] and i < coeff.shape[1]:
                ax_x, ax_y = coeff[pc_x, i] * arrow_scale, coeff[pc_y, i] * arrow_scale
                if abs(ax_x) > 1e-6 or abs(ax_y) > 1e-6:
                    ax.arrow(0, 0, ax_x, ax_y, color='r', alpha=0.8, head_width=max(0.001, 0.02 * arrow_scale), length_includes_head=True)
                    if labels is not None and i < len(labels): ax.text(ax_x * 1.15, ax_y * 1.15, labels[i], color='g', ha='center', va='center', fontsize=9)

    # Final adjustments
    ax.set_xlabel(pca_x_label); ax.set_ylabel(pca_y_label); ax.set_title('PCA Biplot'); ax.grid(True, linestyle='--', alpha=0.6); ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
    if len(all_xs) > 0 and len(all_ys) > 0:
        if make_square: x_lim_val = y_lim_val = max(np.abs(all_xs).max(), np.abs(all_ys).max()) * 1.2
        else: x_lim_val = np.abs(all_xs).max() * 1.2; y_lim_val = np.abs(all_ys).max() * 1.2
    else: x_lim_val, y_lim_val = 1, 1
    x_lim_val = max(x_lim_val, 0.1); y_lim_val = max(y_lim_val, 0.1)
    ax.set_xlim(-x_lim_val, x_lim_val); ax.set_ylim(-y_lim_val, y_lim_val)
    if make_square: ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.80 if legend_visible else 1, 1])
    return fig

# --- Helper function to parse parameter ranges ---
def parse_param_string(param_str, param_type=float, default_value=None, is_list=True):
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

# --- Callback function to sync Target and PCA Color ---
def sync_pca_color_to_target():
    """Sets the PCA biplot color selection to the currently selected target variable."""
    target_val = st.session_state.get('target_selector')
    if target_val:
        st.session_state['pca_biplot_color_col'] = target_val

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("1. Upload CSV", type=["csv"])
encoding_error_handling = st.sidebar.selectbox("2. CSV Encoding Errors", ["Strict", "Ignore", "Replace"], index=1, help="'Ignore' or 'Replace' if errors occur.")
pandas_error_handling = 'strict' if encoding_error_handling == "Strict" else ('ignore' if encoding_error_handling == "Ignore" else 'replace')

# --- Globals & Session State Initialization ---
required_state_keys = {
    'df': None, 'uploaded_filename': None, 'target_selector': None, 'feature_selector': [], 'pca_exclude_target': True,
    'final_estimator': None, 'X_reg_global': None, 'feature_cols_global': None,
    'final_estimator_trained': False, 'target_col_global': None, 'model_type_full_global': None,
    'importance_df': None, 'pca_imputer': None, 'pca_scaler': None, 'pca_model': None,
    'numeric_cols_pca': None, 'pca_loadings_matrix': None, 'explained_variance_ratio_pca': None,
    'pca_scores_global': None,
    'lhs_results_df': None, 'run_analysis_clicked': False,
    'pca_biplot_color_col': "None"
}
for key, default_value in required_state_keys.items():
    if key not in st.session_state: st.session_state[key] = default_value

numeric_cols = []; target_col = None; feature_cols = []

# --- Main Logic ---
if uploaded_file is not None:
    # File Reading Logic
    load_new_file = (st.session_state['df'] is None or (hasattr(uploaded_file, 'name') and uploaded_file.name != st.session_state.get('uploaded_filename')))
    if load_new_file:
        st.warning("New file/first load, resetting state.")
        current_target = st.session_state.get('target_selector')
        current_features = st.session_state.get('feature_selector', [])
        current_pca_color = st.session_state.get('pca_biplot_color_col', "None")
        for key in required_state_keys: st.session_state[key] = required_state_keys[key]
        if current_target: st.session_state['target_selector'] = current_target
        if current_features: st.session_state['feature_selector'] = current_features
        st.session_state['pca_biplot_color_col'] = current_pca_color # Keep color selection attempt
        if hasattr(uploaded_file, 'name'): st.session_state.uploaded_filename = uploaded_file.name
        df_load = None; error_messages = []; successful_encoding = None; encodings_to_try = ['utf-8', 'utf-8-sig', 'shift-jis', 'cp932', 'latin-1']
        st.subheader("Reading File..."); read_placeholder = st.empty()
        for enc in encodings_to_try:
            read_placeholder.write(f"🔄 Trying: `{enc}`...")
            try:
                uploaded_file.seek(0); df_load = pd.read_csv(uploaded_file, encoding=enc, encoding_errors=pandas_error_handling)
                st.sidebar.success(f"✅ Loaded ({enc})"); successful_encoding = enc; read_placeholder.success(f"Loaded: '{successful_encoding}'"); st.session_state['df'] = df_load; st.session_state['run_analysis_clicked'] = False
                for key in ['final_estimator','X_reg_global','feature_cols_global','final_estimator_trained','target_col_global','model_type_full_global','importance_df','lhs_results_df','pca_imputer','pca_scaler','pca_model','numeric_cols_pca','pca_loadings_matrix','explained_variance_ratio_pca','pca_scores_global']:
                     if key in st.session_state: st.session_state[key] = None
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

        # Target & Feature Selection
        st.sidebar.subheader("🎯 3. Target & Features")
        numeric_cols_reg = current_df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols_reg: st.error("No numeric columns."); st.stop()
        target_index = 0; current_target_selection = st.session_state.get('target_selector')
        if current_target_selection in numeric_cols_reg: target_index = numeric_cols_reg.index(current_target_selection)
        elif len(numeric_cols_reg) > 0: target_index = len(numeric_cols_reg) - 1
        # --- ★★★ 修正: on_change コールバックを追加 ★★★ ---
        target_col_selected = st.sidebar.selectbox(
            "Target Variable (Y)", numeric_cols_reg, index=target_index, key="target_selector",
            on_change=sync_pca_color_to_target # コールバック関数を指定
            )
        # --- ★★★ 修正箇所 END ★★★ ---
        target_col = st.session_state.target_selector

        available_features = current_df.columns.tolist()
        if target_col:
            default_numeric_features = [col for col in numeric_cols_reg if col != target_col]
            current_feature_selection = st.session_state.get('feature_selector', [])
            valid_current_features = [f for f in current_feature_selection if f in available_features and f != target_col]
            if not valid_current_features and default_numeric_features: valid_current_features = default_numeric_features; st.session_state.feature_selector = valid_current_features
            elif not valid_current_features: st.warning("No default features available.")
            feature_cols_selected = st.sidebar.multiselect("Features (X)", available_features, default=valid_current_features, key="feature_selector"); feature_cols = st.session_state.feature_selector
        else: feature_cols_selected = []; feature_cols = []; st.sidebar.warning("Select Target.")

        # PCA Settings
        st.sidebar.subheader("🧩 4. PCA Settings"); st.session_state.pca_exclude_target = st.sidebar.checkbox("Exclude Target from PCA", value=st.session_state.pca_exclude_target, key="pca_exclude_target_checkbox")

        # PCA Analysis
        st.header("🧩 PCA"); pca_performed = False
        if not current_df.empty:
            numeric_data_pca_base = current_df.select_dtypes(include=np.number); pca_target_status = ""; cols_for_this_pca = []
            if st.session_state.pca_exclude_target and target_col and target_col in numeric_data_pca_base.columns: numeric_data_pca = numeric_data_pca_base.drop(columns=[target_col]); pca_target_status = f"(Target '{target_col}' excluded)"
            else: numeric_data_pca = numeric_data_pca_base; pca_target_status = "(All numeric included)"
            cols_for_this_pca = numeric_data_pca.columns.tolist(); st.write(f"PCA on numeric columns {pca_target_status}"); original_idx_pca = numeric_data_pca.index
            if numeric_data_pca.empty or len(cols_for_this_pca) < 2: st.warning("PCA needs >= 2 numeric columns.")
            else:
                pca_needs_recalc = (st.session_state.get('pca_model') is None or st.session_state.get('pca_scores_global') is None or st.session_state.get('numeric_cols_pca') != cols_for_this_pca)
                if pca_needs_recalc:
                    st.write("Calculating PCA...");
                    try:
                        pca_imputer = SimpleImputer(strategy='mean'); imputed = pca_imputer.fit_transform(numeric_data_pca); st.session_state['pca_imputer'] = pca_imputer
                        pca_scaler = StandardScaler(); scaled = pca_scaler.fit_transform(imputed); st.session_state['pca_scaler'] = pca_scaler
                        pca_model = PCA(); scores = pca_model.fit_transform(scaled); st.session_state['pca_model'] = pca_model; st.session_state['numeric_cols_pca'] = cols_for_this_pca; st.session_state['pca_loadings_matrix'] = pca_model.components_.T; st.session_state['explained_variance_ratio_pca'] = pca_model.explained_variance_ratio_; st.session_state['pca_scores_global'] = scores; pca_performed = True
                    except Exception as e: st.error(f"PCA error: {e}"); st.code(traceback.format_exc(),language='python'); pca_performed=False
                else: st.write("Using cached PCA."); pca_performed = True
                if pca_performed:
                    _loadings=st.session_state['pca_loadings_matrix']; _expl_var=st.session_state['explained_variance_ratio_pca']; pca_scores=st.session_state['pca_scores_global']; _numeric_cols_pca=st.session_state['numeric_cols_pca']
                    n_comps=len(_expl_var); comp_idx=np.arange(1,n_comps+1); cum_var=np.cumsum(_expl_var)
                    st.subheader("PCA Results"); pc1,pc2=st.columns(2)
                    with pc1: fig_s,ax_s=plt.subplots(figsize=(4.5,3)); ax_s.bar(comp_idx,_expl_var); ax_s.set_title("Scree Plot"); st.pyplot(fig_s)
                    with pc2: fig_c,ax_c=plt.subplots(figsize=(4.5,3)); ax_c.plot(comp_idx,cum_var,'o--'); ax_c.axhline(0.9,c='g',ls=':'); ax_c.axhline(0.95,c='r',ls=':'); ax_c.set_title("Cumulative Var"); ax_c.set_ylim(0,1.1); st.pyplot(fig_c)
                    st.dataframe(pd.DataFrame({'PC':comp_idx,'Expl Var':_expl_var,'Cum Var':cum_var}).round(4)); st.markdown("---")
                    if n_comps>=2:
                        st.subheader("PCA Biplot (PC1 vs PC2)")
                        color_opts=["None"]+current_df.columns.tolist()
                        # --- ★★★ 修正: key を使って session_state の値を直接参照 ★★★ ---
                        # selectbox の index を計算するために session_state の値を使う
                        sel_color=st.session_state.get('pca_biplot_color_col',"None")
                        idx=color_opts.index(sel_color) if sel_color in color_opts else 0
                        color_col_selected_pca = st.selectbox(
                            "🎨 Color Biplot:", color_opts, index=idx, key='pca_biplot_color_col'
                            )
                        # --- ★★★ 修正箇所 END ★★★ ---
                        color_data=None; is_cat=False; label=""
                        # Use the value from session state which reflects the latest selection
                        color_col = st.session_state.pca_biplot_color_col
                        if color_col!="None" and color_col in current_df.columns:
                            try: color_data=current_df.loc[original_idx_pca,color_col]; label=color_col; is_cat=not pd.api.types.is_numeric_dtype(color_data.dtype) or pd.api.types.is_bool_dtype(color_data.dtype) or color_data.nunique()<20;
                            except Exception as e: st.error(f"PCA color prep err: {e}")
                        try: fig=biplot(pca_scores,_loadings,labels=_numeric_cols_pca,explained_variance_ratio=_expl_var,color_data=color_data,color_label=label,is_categorical=is_cat,make_square=True); st.pyplot(fig)
                        except Exception as e: st.error(f"Biplot err: {e}"); st.code(traceback.format_exc(),language='python')
                    else: st.warning("Biplot needs >= 2 PCs.")

        # Regression Setup
        st.markdown("---"); st.header("📈 Regression Analysis Setup")
        target_col = st.session_state.get('target_selector'); feature_cols = st.session_state.get('feature_selector', [])
        if not target_col or not feature_cols: st.warning("⚠️ Select Target & Features."); st.stop()
        if target_col in feature_cols: st.error(f"❌ Target '{target_col}' in Features."); st.stop()

        # Preprocessing Sidebar
        st.sidebar.subheader("🧹 5. Preprocessing"); missing_value_strategy=st.sidebar.selectbox("NaN Handling",["Drop Rows with NaNs","Impute Mean","Impute Median"],0,key="nan_strategy"); log_transform_target=st.sidebar.checkbox("Log Target(Y)",False,key="log_target_checkbox"); log_transform_features=st.sidebar.checkbox("Log Features(X)",False,key="log_features_checkbox"); use_scaling=st.sidebar.checkbox("Scale Feats",True,key="scaling_checkbox")

        # Model Selection & Eval
        st.subheader("🧠 Model, Metric & CV"); mcol1,mcol2,mcol3=st.columns([1,1,1])
        with mcol1: model_type=st.selectbox("Model",["Linear Regression","Ridge","Lasso","Polynomial Regression","Random Forest","Neural Network (MLP)"],0,key="model_selector"); mt_full=model_type;
        if model_type=="Ridge": mt_full="Ridge(L2)"
        elif model_type=="Lasso": mt_full="Lasso(L1)"
        elif model_type=="Neural Network (MLP)": mt_full="MLP"
        metric_map={"R²":"r2","MAE":"neg_mean_absolute_error","RMSE":"neg_root_mean_squared_error"}; metric_disp_key_map={"r2":"R²","neg_mean_absolute_error":"MAE","neg_root_mean_squared_error":"RMSE"}; metric_short_key_map={"r2":"r2","neg_mean_absolute_error":"mae","neg_root_mean_squared_error":"rmse"}
        with mcol2: primary_metric=st.selectbox("Primary Metric",list(metric_map.keys()),0,key="primary_metric_selector")
        opt_metric=primary_metric; primary_metric_scorer=metric_map[primary_metric]; primary_metric_short=metric_disp_key_map[primary_metric_scorer]
        with mcol3: k_folds=st.slider("k-Folds CV",2,20,5,1,key="k_folds_slider")

        # Hyperparameters
        st.subheader("⚙️ Hyperparameters"); optimize_hp=st.checkbox("Enable Grid Search",False,key="optimize_check"); # Removed caption
        if optimize_hp and model_type=="Linear Regression": st.info("Linear Reg no HP."); optimize_hp=False
        grid_model_prefix="regressor__model__" if log_transform_target else "model__"; grid_step_prefix="regressor__" if log_transform_target else ""
        opt_metric_display=primary_metric; opt_metric_scorer=primary_metric_scorer; opt_metric_short=primary_metric_short
        if optimize_hp:
            with st.expander("Grid Search Opt Metric",True): opt_metric_display_gs=st.selectbox("Optimize Metric",list(metric_map.keys()),list(metric_map.keys()).index(primary_metric),key="opt_metric_selector"); opt_metric_display=opt_metric_display_gs; opt_metric_scorer=metric_map[opt_metric_display_gs]; opt_metric_short=metric_disp_key_map[opt_metric_scorer]
        exp_title=f"Configure: {mt_full}";
        if optimize_hp: exp_title+=f" (Optimizing: {opt_metric_short})"
        with st.expander(exp_title,True):
            param_grid={}; poly_check_needed=False; poly_max_degree=1; n_input_features_poly=len(feature_cols)
            try:
                if model_type=="Linear Regression": st.write("No HP.")
                elif model_type=="Ridge":
                    if optimize_hp: param_grid[grid_model_prefix+'alpha']=parse_param_string(st.text_input("Alpha(csv)","0.1,1,10",key="r_a_g"),float,[1.0])
                    else: alpha_ridge=st.slider("Alpha",0.01,100.0,1.0,0.01,key="r_a")
                elif model_type=="Lasso":
                    if optimize_hp: param_grid[grid_model_prefix+'alpha']=parse_param_string(st.text_input("Alpha(csv)","0.01,0.1,1",key="l_a_g"),float,[1.0])
                    else: alpha_lasso=st.slider("Alpha",0.01,10.0,1.0,0.01,key="l_a")
                elif model_type=="Polynomial Regression":
                    poly_check_needed=True; poly_step_name='poly'; poly_degree_key=grid_step_prefix+poly_step_name+'__degree'; model_alpha_key=grid_model_prefix+'alpha'
                    if optimize_hp:
                        degree_opts=[2,3,4,5]; sel_degs=st.multiselect("Poly Degree(s)",degree_opts,[2],key="p_d_g_multi")
                        if not sel_degs: st.warning("Select degree(s)"); param_grid[poly_degree_key]=[2]; poly_max_degree=2
                        else: param_grid[poly_degree_key]=sorted(sel_degs); poly_max_degree=max(sel_degs)
                        st.write("Using Ridge after poly."); param_grid[model_alpha_key]=parse_param_string(st.text_input("Ridge Alpha(csv)","0.1,1,10",key="p_r_a_g"),float,[1.0])
                    else:
                        deg_fixed=st.number_input("Poly Degree",2,10,2,1,key="p_d_num"); poly_max_degree=deg_fixed
                        use_ridge_after_poly=st.checkbox("Use Ridge",True,key="p_r_c");
                        if use_ridge_after_poly: alpha_poly_ridge=st.slider("Ridge Alpha",0.01,100.0,1.0,0.01,key="p_r_a")
                elif model_type=="Random Forest":
                    rf_prefix=grid_model_prefix
                    if optimize_hp:
                        c1,c2=st.columns(2);
                        with c1:
                            param_grid[rf_prefix+'n_estimators']=parse_param_string(st.text_input("Trees","100,200",key="rf_e_g"),int,[100])
                            d_str=st.text_input("Depth","None,10,20",key="rf_d_g")
                            p_d=parse_param_string(d_str,str,['None'],True)
                            f_d=[]
                            for item in p_d: # Corrected parsing loop
                                if item is None: f_d.append(None); continue
                                if isinstance(item, str):
                                    if item.isdigit(): f_d.append(int(item))
                                    elif item.lower() == 'none': f_d.append(None)
                                    else: st.warning(f"Ignoring invalid depth: {item}")
                                else: st.warning(f"Ignoring unexpected type depth: {item} ({type(item)})")
                            param_grid[rf_prefix + 'max_depth'] = f_d if f_d else [None]

                            f_str=st.text_input("Feats","sqrt,0.8",key="rf_f_g")
                            p_f=parse_param_string(f_str,str,['sqrt'],True)
                            f_f=[]
                            for f in p_f: # Corrected parsing loop
                                if f is None: st.warning("Ignoring 'None' for Max Feats."); continue
                                if isinstance(f, str) and f.lower() in ['sqrt', 'log2']: f_f.append(f.lower())
                                else:
                                    try: f_float=float(f);
                                    except (ValueError,TypeError): f_float=None
                                    if f_float is not None and 0<f_float<=1.0: f_f.append(f_float); continue
                                    try: f_int=int(f);
                                    except (ValueError,TypeError): f_int=None
                                    if f_int is not None and f_int>=1: f_f.append(f_int); continue
                                    st.warning(f"Ignoring invalid max_features: {f}. Using default 'sqrt'.")
                            param_grid[rf_prefix+'max_features']=f_f if f_f else ['sqrt']
                        with c2:
                            param_grid[rf_prefix+'min_samples_split']=parse_param_string(st.text_input("Min Split","2,5",key="rf_s_g"),int,[2])
                            param_grid[rf_prefix+'min_samples_leaf']=parse_param_string(st.text_input("Min Leaf","1,3",key="rf_l_g"),int,[1])
                    else:
                        rf1,rf2=st.columns(2);
                        with rf1: rf_n_estimators=st.number_input("Trees",10,2000,100,10,key="rf_est"); rf_max_d_opt=st.selectbox("Depth",["None",5,10,15,20,30],0,key="rf_max_d"); rf_max_d_val=None if rf_max_d_opt=="None" else int(rf_max_d_opt); rf_max_f_opt=st.selectbox("Feats",["sqrt","log2","all"],0,key="rf_max_f"); rf_max_f_val=None if rf_max_f_opt=="all" else rf_max_f_opt
                        with rf2: rf_min_s_split=st.slider("Min Split",2,50,2,1,key="rf_min_s"); rf_min_s_leaf=st.slider("Min Leaf",1,50,1,1,key="rf_min_l")
                elif model_type=="Neural Network (MLP)":
                    mlp_prefix=grid_model_prefix
                    # Removed st.caption("MLP GS slow.")
                    iter_fixed=500; early_stop=True; patience=10
                    if optimize_hp:
                        c1,c2=st.columns(2);
                        with c1: param_grid[mlp_prefix+'hidden_layer_sizes']=parse_param_string(st.text_input('Layers','(100,), (50,50)',key="mlp_h_g"),tuple,[(100,)]); param_grid[mlp_prefix+'activation']=parse_param_string(st.text_input("Activation","relu,tanh",key="mlp_a_g"),str,['relu']); param_grid[mlp_prefix+'solver']=parse_param_string(st.text_input("Solver","adam",key="mlp_s_g"),str,['adam'])
                        with c2: param_grid[mlp_prefix+'alpha']=parse_param_string(st.text_input("Alpha L2","0.0001,0.001",key="mlp_alpha_g"),float,[0.0001]); param_grid[mlp_prefix+'learning_rate_init']=parse_param_string(st.text_input("LR Init","0.001",key="mlp_lr_g"),float,[0.001]); st.number_input("MaxIter(GS)",iter_fixed,disabled=True); st.checkbox("Stop(GS)",early_stop,disabled=True); st.number_input("Patience(GS)",patience,disabled=True)
                    else:
                        c1,c2=st.columns(2);
                        with c1:
                            mlp_h_str=st.text_input("Layers","100",key="mlp_h")
                            try: layers=[int(x.strip()) for x in mlp_h_str.split(',') if x.strip().isdigit()]; mlp_h_val=tuple(layers) if layers else (100,)
                            except ValueError: st.error("Bad Layers. Using (100,)."); mlp_h_val=(100,)
                            mlp_a_fixed=st.selectbox("Activation",["relu","tanh","logistic","identity"],0,key="mlp_a")
                            mlp_s_fixed=st.selectbox("Solver",["adam","sgd","lbfgs"],0,key="mlp_s")
                        with c2:
                            mlp_alpha_fixed=st.number_input("Alpha L2",0.0,0.1,0.0001,format="%.5f",key="mlp_alpha")
                            mlp_lr_fixed=st.number_input("LR Init",0.00001,0.1,0.001,format="%.5f",key="mlp_lr")
                            mlp_max_iter_fixed=st.number_input("Max Iters",50,5000,iter_fixed,100,key="mlp_iter")
                            mlp_early_stop=st.checkbox("Early Stop",early_stop,key="mlp_stop")
                            mlp_patience=patience
                            if mlp_early_stop: mlp_patience=st.number_input("Patience",3,100,patience,1,key="mlp_pat")
            except Exception as e: st.error(f"Param setup error: {e}"); param_grid={}
            # Poly preliminary check
            if poly_check_needed and n_input_features_poly > 0:
                try:
                    max_feats = calculate_poly_features(n_input_features_poly, poly_max_degree)
                    n_samples_approx = len(current_df)
                    if max_feats >= n_samples_approx: # Separate line
                        st.warning(f"⚠️ Poly degree {poly_max_degree} -> {max_feats} features >= {n_samples_approx} samples?")
                except Exception as pc_e:
                    st.warning(f"Poly check failed: {pc_e}")

        # Run Analysis Button
        st.markdown("---"); st.header("🚀 Run Regression Analysis")
        run_label=f"Run {mt_full} with {k_folds}-Fold CV"; notes=[]
        if log_transform_target: notes.append("LogY")
        if log_transform_features: notes.append("LogX")
        if use_scaling: notes.append("ScaleX")
        if missing_value_strategy.startswith("Impute"): notes.append(missing_value_strategy)
        if notes: run_label+=f" ({','.join(notes)})"
        can_run_gs=optimize_hp and bool(param_grid)
        if optimize_hp: run_label+=f" [GS Opt:{opt_metric_short}]" if can_run_gs else " [GS Config Err]"
        if optimize_hp and not can_run_gs: st.warning("GS enabled but invalid params.")

        if st.button(run_label, key="run_button"):
            st.session_state['run_analysis_clicked'] = True
            for key in ['final_estimator', 'X_reg_global', 'feature_cols_global', 'final_estimator_trained', 'target_col_global', 'model_type_full_global', 'importance_df', 'lhs_results_df']:
                if key in st.session_state: st.session_state[key] = None
            _target=st.session_state.target_selector; _features=st.session_state.feature_selector; _df=st.session_state['df']
            if _target is None or not _features: st.error("Target/features missing."); st.stop()
            cols_reg=[_target]+_features; data_reg = _df[cols_reg].copy()

            # Data Prep
            num_feat_cols=[];
            for col in _features:
                if col in data_reg.columns:
                    if data_reg[col].dtype=='object' or pd.api.types.is_categorical_dtype(data_reg[col].dtype): data_reg[col]=pd.to_numeric(data_reg[col],errors='coerce')
                    if pd.api.types.is_numeric_dtype(data_reg[col].dtype): num_feat_cols.append(col)
            if data_reg[_target].dtype=='object' or pd.api.types.is_categorical_dtype(data_reg[_target].dtype): data_reg[_target]=pd.to_numeric(data_reg[_target],errors='coerce')
            if not pd.api.types.is_numeric_dtype(data_reg[_target].dtype): st.error(f"Target '{_target}' not numeric."); st.stop()
            rows_init=len(data_reg); rows_dropped=0
            if missing_value_strategy=="Drop Rows with NaNs": data_reg.dropna(subset=cols_reg,inplace=True); rows_dropped=rows_init-len(data_reg)
            elif missing_value_strategy in ["Impute Mean","Impute Median"]: rows_before=len(data_reg); data_reg.dropna(subset=[_target],inplace=True); rows_dropped=rows_before-len(data_reg)
            if rows_dropped>0: st.info(f"Dropped {rows_dropped} rows with NaNs.")
            if data_reg.empty: st.error("No data left after NaN handling."); st.stop()
            X=data_reg[_features].copy(); y=data_reg[_target].copy(); n_samples_final=len(X)
            st.session_state['X_reg_global']=X.copy(); st.session_state['feature_cols_global']=_features[:]; st.session_state['target_col_global']=_target; st.session_state['model_type_full_global']=mt_full

            neg = [] # Initialize neg
            if log_transform_target and (y<0).any(): st.warning(f"⚠️ Target '{_target}' < 0. Using log(1+x).")
            if log_transform_features:
                neg = [c for c in num_feat_cols if c in X.columns and (X[c]<0).any()]
                if neg: st.warning(f"⚠️ Features {neg} < 0. Using log(1+x).")

            # CV setup
            k_cv = k_folds;
            if n_samples_final < k_cv: k_cv=max(2,n_samples_final); st.warning(f"k adjusted to {k_cv} due to samples ({n_samples_final}).")
            kf=KFold(n_splits=k_cv,shuffle=True,random_state=42)

            # Build pipeline
            steps=[]; st.write("---"); st.write("**Pipeline Steps:**"); log1p_tf=FunctionTransformer(np.log1p,np.expm1,validate=False)
            if log_transform_features: steps.append(('logFeat',log1p_tf)); st.write("- Log Features")
            if missing_value_strategy=="Impute Mean" and X.isnull().any().any(): steps.append(('imputeM',SimpleImputer(strategy='mean'))); st.write("- Mean Impute")
            elif missing_value_strategy=="Impute Median" and X.isnull().any().any(): steps.append(('imputeMed',SimpleImputer(strategy='median'))); st.write("- Median Impute")
            if use_scaling: steps.append(('scale',StandardScaler())); st.write("- Scale Features")

            # Polynomial Feature Check (Runtime)
            if model_type == "Polynomial Regression":
                 poly_deg_pipe = deg_fixed if not optimize_hp else (param_grid[poly_degree_key][0] if poly_degree_key in param_grid and param_grid[poly_degree_key] else 2)
                 n_feat_in = X.shape[1]; max_deg_run = poly_max_degree
                 try:
                     n_feat_out = calculate_poly_features(n_feat_in, max_deg_run)
                     st.write(f"(Max degree {max_deg_run} -> ~{n_feat_out} features from {n_feat_in}; {n_samples_final} samples)")
                     if n_feat_out >= n_samples_final: st.error(f"❌ Ill-Determined: Poly degree {max_deg_run} -> {n_feat_out} features >= {n_samples_final} samples. Reduce degree."); st.stop()
                 except Exception as poly_e: st.warning(f"Poly check error: {poly_e}")
                 steps.append(('poly', PolynomialFeatures(degree=poly_deg_pipe, include_bias=False)))
                 st.write(f"- Poly Features (degree specified or from GS)")

            # Model Instantiation
            model=None
            try: # Model creation logic
                if model_type=="Linear Regression": model=LinearRegression()
                elif model_type=="Ridge": model=Ridge(alpha=alpha_ridge if not optimize_hp else 1.0,random_state=42)
                elif model_type=="Lasso": model=Lasso(alpha=alpha_lasso if not optimize_hp else 1.0,max_iter=10000,random_state=42)
                elif model_type=="Polynomial Regression":
                    if optimize_hp: model=Ridge(random_state=42); st.write(f"- Ridge (alpha=GS) after Poly")
                    else: model=Ridge(alpha=alpha_poly_ridge,max_iter=10000,random_state=42) if use_ridge_after_poly else LinearRegression(); st.write(f"- {'Ridge(alpha=%.4f)'%alpha_poly_ridge if use_ridge_after_poly else 'Linear Reg'} after Poly")
                elif model_type=="Random Forest":
                    rf_pars={'random_state': 42, 'n_jobs': -1}
                    if not optimize_hp: rf_pars.update({'n_estimators': rf_n_estimators, 'max_depth': rf_max_d_val, 'min_samples_split': rf_min_s_split, 'min_samples_leaf': rf_min_s_leaf, 'max_features': rf_max_f_val}); st.write("- RF (fixed)")
                    else: st.write("- RF (GS)")
                    model=RandomForestRegressor(**rf_pars)
                elif model_type=="Neural Network (MLP)":
                    mlp_pars={'random_state': 42, 'max_iter': mlp_max_iter_fixed if not optimize_hp else fixed_iter, 'early_stopping': mlp_early_stop if not optimize_hp else fixed_early_stop, 'n_iter_no_change': mlp_patience if not optimize_hp else fixed_patience}
                    if not optimize_hp: mlp_pars.update({'hidden_layer_sizes': mlp_h_val, 'activation': mlp_a_fixed, 'solver': mlp_s_fixed, 'alpha': mlp_alpha_fixed, 'learning_rate_init': mlp_lr_fixed}); st.write("- MLP (fixed)")
                    else: mlp_pars.update({'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001, 'learning_rate_init': 0.001}); st.write("- MLP (GS)")
                    model=MLPRegressor(**mlp_pars)
            except NameError as ne: st.error(f"Config Error: {ne}"); st.stop()
            except Exception as e: st.error(f"Model init error: {e}"); st.stop()
            if model is None: st.error("Model failed."); st.stop()

            # Final Estimator
            pipe=Pipeline(steps+[('model',model)])
            estimator=TransformedTargetRegressor(regressor=pipe,func=np.log1p,inverse_func=np.expm1) if log_transform_target else pipe
            st.write("Final Estimator:"); st.code(str(estimator), language='python')

            # Execute CV/GS
            scores_cv_dict={}; results_gs=None; best_p=None; score_map=metric_map;
            refit_value_for_gs = opt_metric_display # Use display name for refit

            with st.spinner(f"⏳ Running {run_label}..."):
                t0=time.time()
                try:
                    if can_run_gs:
                        st.write(f"Running GridSearch (Refit: {opt_metric_display})...")
                        gs = GridSearchCV(estimator, param_grid, scoring=score_map, refit=refit_value_for_gs, cv=kf, n_jobs=-1)
                        gs.fit(X,y)
                        st.session_state['final_estimator']=gs.best_estimator_; best_p=gs.best_params_; results_gs=pd.DataFrame(gs.cv_results_); best_idx=gs.best_index_; scores_cv_dict=results_gs.loc[best_idx].to_dict()
                        st.success(f"✅ GS done ({time.time()-t0:.2f}s). Best params:"); st.json(best_p)
                    else:
                        st.write(f"Running {k_cv}-fold CV..."); scores_cv_dict=cross_validate(estimator,X,y,cv=kf,scoring=score_map,n_jobs=-1, return_train_score=False); estimator.fit(X,y); st.session_state['final_estimator']=estimator
                        st.success(f"✅ CV done ({time.time()-t0:.2f}s).")
                    st.session_state['final_estimator_trained']=True

                    # Process Scores
                    processed_scores = {}; metric_disp_key = metric_disp_key_map
                    for score_display_name, scorer_key in metric_map.items():
                         short_name = metric_disp_key[scorer_key]; test_score_key = f'test_{scorer_key}'; mean_score_key = f'mean_test_{scorer_key}'; std_score_key = f'std_test_{scorer_key}'
                         if test_score_key in scores_cv_dict:
                             scores_array = scores_cv_dict[test_score_key]; valid_scores = np.array(scores_array)[np.isfinite(scores_array)] if isinstance(scores_array,(list, np.ndarray)) else np.array([])
                             if scorer_key.startswith('neg_'): valid_scores = -valid_scores
                             if len(valid_scores)>0: processed_scores[short_name]={'mean':np.mean(valid_scores),'std':np.std(valid_scores),'values':valid_scores}
                             else: processed_scores[short_name]={'mean':np.nan,'std':np.nan,'values':np.array([])}
                         elif mean_score_key in scores_cv_dict:
                             mean_val = scores_cv_dict[mean_score_key]; std_val = scores_cv_dict.get(std_score_key, np.nan); valid_mean = mean_val if np.isfinite(mean_val) else np.nan
                             if scorer_key.startswith('neg_'): valid_mean = -valid_mean
                             processed_scores[short_name]={'mean':valid_mean,'std':std_val,'values':np.array([valid_mean]) if not np.isnan(valid_mean) else np.array([])}
                    if 'RMSE' not in processed_scores and 'MAE' in processed_scores:
                        mse_key_neg='neg_mean_squared_error'; mse_test_key_cv=f'test_{mse_key_neg}'; mse_mean_key_gs=f'mean_test_{mse_key_neg}'
                        if mse_test_key_cv in scores_cv_dict:
                            valid_mse=-np.array(scores_cv_dict[mse_test_key_cv])[np.isfinite(scores_cv_dict[mse_test_key_cv])]
                            if len(valid_mse)>0: rmse_vals=np.sqrt(np.maximum(0,valid_mse)); processed_scores['RMSE']={'mean':np.mean(rmse_vals),'std':np.std(rmse_vals),'values':rmse_vals}
                        elif mse_mean_key_gs in scores_cv_dict:
                            mean_mse=-scores_cv_dict[mse_mean_key_gs]; std_mse=scores_cv_dict.get(f'std_test_{mse_key_neg}',np.nan)
                            if not np.isnan(mean_mse): processed_scores['RMSE']={'mean':np.sqrt(np.maximum(0,mean_mse)),'std':np.nan,'values':np.array([np.sqrt(np.maximum(0,mean_mse))])}

                    # Results Display
                    st.subheader(f"📊 Performance Results ({k_cv}-Fold CV Avg)"); st.write(f"Primary: **{primary_metric}**");
                    if can_run_gs: st.write(f"Optimized for: **{opt_metric_display}**")
                    if processed_scores:
                        n_metrics=len(processed_scores); res_cols=st.columns(n_metrics); metric_order=['R²','MAE','RMSE']; col_idx=0; displayed=set()
                        def disp_metric(mkey,primary,col):
                            if mkey in processed_scores:
                                summ=processed_scores[mkey]; mean,std=summ['mean'],summ['std']; lbl=f"**{mkey}**" if primary else mkey; delta=f"± {std:.4f}" if not np.isnan(std) else None; help_txt="Higher better" if mkey=='R²' else "Lower better";
                                with col: st.metric(lbl,f"{mean:.4f}" if not np.isnan(mean) else "N/A", delta=delta if primary else None,delta_color="off",help=help_txt); return True
                            return False
                        if primary_metric_short in processed_scores and col_idx<len(res_cols):
                            if disp_metric(primary_metric_short,True,res_cols[col_idx]): displayed.add(primary_metric_short); col_idx+=1
                        for mkey in metric_order:
                            if mkey!=primary_metric_short and mkey in processed_scores and col_idx<len(res_cols):
                                if disp_metric(mkey,False,res_cols[col_idx]): displayed.add(mkey); col_idx+=1
                        for mkey in processed_scores:
                            if mkey not in displayed and col_idx<len(res_cols):
                                if disp_metric(mkey,False,res_cols[col_idx]): displayed.add(mkey); col_idx+=1
                    else: st.warning("No valid metrics calculated.")

                    # CV Preds & Plots
                    st.write("---"); st.write("⏳ Generating CV predictions..."); y_pred_cv = None
                    # --- ★★★ 修正: CV予測実行前にモデル学習成功を確認 ★★★ ---
                    if st.session_state.get('final_estimator_trained') and st.session_state.get('final_estimator'):
                        try:
                            final_model_for_cv = st.session_state['final_estimator']
                            y_pred_cv = cross_val_predict(final_model_for_cv,X,y,cv=kf,n_jobs=-1)
                        except Exception as cvp_e:
                            st.error(f"CV prediction error: {cvp_e}")
                            y_pred_cv = None # Ensure it's None on error
                    else:
                         st.warning("Final model not trained successfully, skipping CV predictions.")
                         y_pred_cv = None
                    # --- ★★★ 修正箇所 END ★★★ ---

                    st.subheader("📉 Visualizations (CV Preds)")
                    if y_pred_cv is not None:
                        try:
                            y_np=y.to_numpy(); valid=np.isfinite(y_pred_cv)&np.isfinite(y_np); yv,ypv=y_np[valid],y_pred_cv[valid]
                            if len(yv)>0:
                                plt.style.use('seaborn-v0_8-whitegrid'); v1,v2=st.columns(2);
                                with v1: figA,ax=plt.subplots(figsize=(5.25,4.5));sns.scatterplot(x=yv,y=ypv,alpha=0.6,s=40,ax=ax,label="P");lim_min,lim_max=min(yv.min(),ypv.min()),max(yv.max(),ypv.max());pad=(lim_max-lim_min)*0.05;lmin,lmax=lim_min-pad,lim_max+pad;ax.plot([lmin,lmax],[lmin,lmax],'--r',lw=2,label='Ideal');ax.set_xlabel("Actual");ax.set_ylabel("Pred(CV)");ax.set_title("Actual vs Pred");ax.legend();ax.grid(True);ax.set_xlim(lmin,lmax);ax.set_ylim(lmin,lmax);plt.tight_layout();st.pyplot(figA)
                                with v2: res=yv-ypv;figR,ax=plt.subplots(figsize=(5.25,4.5));sns.scatterplot(x=ypv,y=res,alpha=0.6,s=40,ax=ax);ax.axhline(0,color='r',ls='--',lw=2,label='Zero');ax.set_xlabel("Pred(CV)");ax.set_ylabel("Residual");ax.set_title("Residual vs Pred");ax.legend();ax.grid(True);max_a=np.abs(res).max()*1.1 if len(res)>0 else 1;ax.set_ylim(-max_a,max_a);plt.tight_layout();st.pyplot(figR)
                            else: st.warning("No valid plot points.")
                        except Exception as e: st.error(f"Plot error: {e}")
                    else: st.warning("CV preds unavailable for plots.")

                    # Feature Importance
                    st.markdown("---"); st.subheader("✨ Feature Importance / Coefficients")
                    _est_imp=st.session_state.get('final_estimator'); _X_imp=st.session_state.get('X_reg_global'); _mt_imp=st.session_state.get('model_type_full_global'); imp_df=None
                    if _est_imp and _X_imp is not None and _mt_imp:
                        st.write(f"Extracting from final model ({_mt_imp})...")
                        try:
                            pipe_imp = _est_imp.regressor_ if isinstance(_est_imp, TransformedTargetRegressor) else _est_imp
                            final_model = pipe_imp.named_steps['model']
                            feat_names_in = list(_X_imp.columns); feat_names_out = feat_names_in
                            current_transformer_names = feat_names_in
                            for step_name, step_transformer in pipe_imp.steps[:-1]:
                                try:
                                    if hasattr(step_transformer, 'get_feature_names_out'):
                                        current_transformer_names = list(step_transformer.get_feature_names_out(input_features=current_transformer_names))
                                    elif isinstance(step_transformer, PCA) and hasattr(step_transformer, 'n_components_'):
                                        n_pca_comps = step_transformer.n_components_ if step_transformer.n_components_ else 0
                                        if n_pca_comps == 0:
                                             if hasattr(final_model, 'coef_'): n_pca_comps = final_model.coef_.shape[-1]
                                             elif hasattr(final_model, 'feature_importances_'): n_pca_comps = final_model.feature_importances_.shape[-1]
                                        current_transformer_names = [f"PC{i+1}" for i in range(n_pca_comps)] if n_pca_comps > 0 else current_transformer_names
                                except Exception as name_err: st.warning(f"Feat name err step '{step_name}': {name_err}");
                            feat_names_out = current_transformer_names

                            imp_values = None; imp_type = None
                            if hasattr(final_model, 'coef_') and model_type != "Neural Network (MLP)": imp_values = final_model.coef_.flatten(); imp_type = 'Coefficient'
                            elif hasattr(final_model, 'feature_importances_'): imp_values = final_model.feature_importances_; imp_type = 'Importance'
                            if imp_values is not None:
                                if len(imp_values) == len(feat_names_out):
                                    imp_df=pd.DataFrame({'Feature':feat_names_out, imp_type:imp_values}).sort_values(imp_type,key=abs,ascending=False)
                                    n_plot=min(20,len(imp_df));
                                    if n_plot>0: fig,ax=plt.subplots(figsize=(7,max(4,n_plot*0.3))); sns.barplot(x=imp_type,y='Feature',data=imp_df.head(n_plot).iloc[::-1],ax=ax,palette='viridis'); ax.set_title(f'Top {n_plot} Feature {imp_type}s ({_mt_imp})'); plt.tight_layout(); st.pyplot(fig)
                                    st.dataframe(imp_df.round(4))
                                else:
                                    st.warning(f"⚠️ Length mismatch: {len(feat_names_out)} names vs {len(imp_values)} {imp_type}s."); st.write("Names:", feat_names_out); st.write(f"{imp_type}s:", imp_values)
                                    try: # Fallback display
                                         n_outputs = getattr(final_model, 'n_features_out_', len(imp_values))
                                         if len(imp_values)==n_outputs: # Separate line
                                             st.info("Trying generic names.")
                                             generic_names=[f"Feature_{i}" for i in range(len(imp_values))]
                                             imp_df=pd.DataFrame({'Feature':generic_names, imp_type:imp_values}).sort_values(imp_type,key=abs,ascending=False)
                                             st.dataframe(imp_df.round(4))
                                    except Exception as fallback_e:
                                        st.warning(f"Fallback display failed: {fallback_e}")
                            else: st.info(f"Model '{_mt_imp}' no coef/imp.")
                            if imp_df is not None: st.session_state['importance_df']=imp_df.copy()
                        except Exception as e: st.error(f"Importance/Coef error: {e}"); st.code(traceback.format_exc(), language='python')
                    else: st.info("Requires trained model.")

                    # Download Importance Data
                    if st.session_state.get('importance_df') is not None:
                         _idf=st.session_state['importance_df']; _mt=st.session_state.get('model_type_full_global','m')
                         try: csv_i=_idf.to_csv(index=False).encode('utf-8-sig'); st.download_button("⬇️ DL Import/Coef",csv_i,f'{_mt}_imp.csv','text/csv',key='dl_i')
                         except Exception as e: st.error(f"DL prep error (Imp): {e}")
                    elif st.session_state.get('final_estimator_trained'): st.info("No importance data generated.")

                    # Display Grid Search Results
                    if results_gs is not None:
                        with st.expander("🔍 Detailed Grid Search Results", False):
                             try:
                                 st.write(f"GS explored {len(results_gs)} combinations. Ranked by {opt_metric_display} (`{opt_metric_scorer}`).") # Use scorer name
                                 cols_s=['params']; rank_c=f'rank_test_{opt_metric_scorer}' # Use scorer name
                                 if rank_c not in results_gs.columns: # Separate line
                                     rank_c=f'mean_test_{opt_metric_scorer}'
                                     st.warning(f"Rank column missing, sorting by {rank_c}")
                                 for d_name, s_key in metric_map.items():
                                     mean_c,std_c=f'mean_test_{s_key}',f'std_test_{s_key}'
                                     if mean_c in results_gs: cols_s.append(mean_c)
                                     if std_c in results_gs: cols_s.append(std_c)
                                 if rank_c in results_gs and rank_c not in cols_s: cols_s.append(rank_c) # Separate line
                                 if 'mean_fit_time' in results_gs: cols_s.append('mean_fit_time')
                                 if 'mean_score_time' in results_gs: cols_s.append('mean_score_time')

                                 res_df_disp=results_gs[cols_s].copy(); res_df_disp['params_str']=res_df_disp['params'].astype(str)
                                 rename_d={}; final_cols=['params_str']
                                 for col in res_df_disp.columns:
                                     if col in ['params','params_str']: continue
                                     new_n,neg=col,False;
                                     if col.startswith(('mean_test_neg_','std_test_neg_')): res_df_disp[col]=-res_df_disp[col]; neg=True
                                     s_key_part=col.split('_test_')[-1]; prefix=col.split('_test_')[0]+'_'
                                     orig_s_key=f"neg_{s_key_part}" if neg and f"neg_{s_key_part}" in metric_disp_key_map else (s_key_part if s_key_part in metric_disp_key_map else None)
                                     if orig_s_key: short_m=metric_disp_key_map[orig_s_key]; new_n=prefix.replace('_test_','_')+short_m; rename_d[col]=new_n
                                     final_cols.append(rename_d.get(col,col))
                                 res_df_disp.rename(columns=rename_d,inplace=True); rank_c_renamed=rename_d.get(rank_c,rank_c);
                                 if rank_c!=rank_c_renamed: st.write(f"(Sort by: {rank_c_renamed})")
                                 rank_c=rank_c_renamed; sort_asc=True;
                                 if rank_c.startswith('mean_'): sort_asc=False if 'R²' in rank_c else True
                                 if rank_c not in res_df_disp.columns: st.error(f"Sort column {rank_c} missing.")
                                 else: res_df_disp=res_df_disp.sort_values(by=rank_c,ascending=sort_asc)
                                 final_cols_exist=[c for c in final_cols if c in res_df_disp.columns]; [final_cols_exist.append(c) for c in res_df_disp.columns if c not in final_cols_exist]
                                 st.dataframe(res_df_disp[final_cols_exist].round(5)); st.caption("Lower MAE/RMSE better. Higher R² better. Times in sec.")
                                 csv_gs=res_df_disp[final_cols_exist].to_csv(index=False).encode('utf-8-sig'); st.download_button("⬇️ DL GridSearch Results",csv_gs,f'{mt_full}_gridsearch.csv','text/csv',key='dl_gs')
                             except Exception as e: st.error(f"GS display error: {e}"); st.dataframe(results_gs)

                except Exception as e: st.error(f"Run error: {e}"); st.code(traceback.format_exc(), language='python'); st.session_state['final_estimator_trained']=False

    # --- Sections displayed only if analysis was run successfully ---
    if st.session_state.get('final_estimator_trained'):
        # --- Model Download Section ---
        st.markdown("---")
        st.subheader("💾 Download Trained Model")
        model_dl = st.session_state.get('final_estimator')
        mt_dl = st.session_state.get('model_type_full_global', 'model')
        if model_dl:
            st.write(f"Download the final trained **{mt_dl}** model pipeline.")
            try:
                b = io.BytesIO(); joblib.dump(model_dl, b); b.seek(0)
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M'); safe_m = "".join(c if c.isalnum() else "_" for c in mt_dl); fn=f"trained_{safe_m}_{ts}.joblib"
                st.download_button(f"⬇️ Download Model ({fn})", b, fn, "application/octet-stream", key='dl_mod')
                st.caption("Load using ~~~joblib.load()~~~.")
            except Exception as e:
                st.error(f"Model DL error: {e}")
        else:
            st.warning("Trained model object not found.")

        # --- LHS Simulation Section ---
        st.markdown("---")
        st.header("🔬 Latin Hypercube Sampling (LHS) Simulation")
        if not SCIPY_AVAILABLE:
             st.info("LHS requires SciPy (`pip install scipy`).")
        else:
            est_lhs=st.session_state.get('final_estimator'); X_lhs=st.session_state.get('X_reg_global'); tgt_lhs=st.session_state.get('target_col_global'); mt_lhs=st.session_state.get('model_type_full_global'); df_lhs=st.session_state.get('df')
            pca_imp=st.session_state.get('pca_imputer'); pca_sc=st.session_state.get('pca_scaler'); pca_mod=st.session_state.get('pca_model'); pca_cols=st.session_state.get('numeric_cols_pca'); pca_load=st.session_state.get('pca_loadings_matrix'); pca_var=st.session_state.get('explained_variance_ratio_pca'); pca_scores_orig=st.session_state.get('pca_scores_global')
            if not all([est_lhs, X_lhs is not None, tgt_lhs, mt_lhs, df_lhs is not None]):
                 st.error("LHS needs previous run info.")
            else:
                st.write("Generate simulated data & predict."); num_feats_lhs = X_lhs.select_dtypes(include=np.number).columns.tolist(); all_feats=list(X_lhs.columns); non_num_lhs=[f for f in all_feats if f not in num_feats_lhs]
                if not num_feats_lhs:
                     st.warning("No numeric features for LHS.")
                else:
                    st.write(f"**LHS Features:** `{', '.join(num_feats_lhs)}`");
                    if non_num_lhs:
                         st.caption(f"Excluded: `{', '.join(non_num_lhs)}`")
                    lhs_n=st.number_input("LHS Samples",100,100000,10000,100,key="lhs_n")
                    if st.button("🚀 Run LHS & Predict", key="run_lhs"):
                        st.session_state['lhs_results_df']=None
                        with st.spinner(f"⏳ Running LHS ({lhs_n})..."):
                             try:
                                 bounds=df_lhs[num_feats_lhs].agg(['min','max']); lb,ub=bounds.loc['min'].values,bounds.loc['max'].values; dim=len(num_feats_lhs)
                                 mask=lb>=ub;
                                 if np.any(mask): st.warning("Adjusting bounds."); adj=np.maximum(1e-6,0.01*np.abs(lb[mask])); lb[mask]-=adj; ub[mask]+=adj;
                                 if np.any(lb>=ub): st.error("Invalid bounds."); st.stop()
                                 try: sampler=qmc.LatinHypercube(d=dim,optimization="random-cd",seed=42); st.info("Using LHS 'random-cd'.")
                                 except TypeError: st.warning("Using LHS default opt."); sampler=qmc.LatinHypercube(d=dim,seed=42)
                                 s01=sampler.random(n=lhs_n); scaled=qmc.scale(s01,lb,ub); lhs_in=pd.DataFrame(scaled,columns=num_feats_lhs)
                                 for col in non_num_lhs: lhs_in[col]=np.nan
                                 try: lhs_in_ord=lhs_in[all_feats]
                                 except KeyError as e: st.error(f"LHS col err: {e}"); st.stop()
                                 preds=est_lhs.predict(lhs_in_ord); res=lhs_in[num_feats_lhs].copy(); pred_col=f'{tgt_lhs}_predicted'; res[pred_col]=preds; st.session_state['lhs_results_df']=res.copy(); st.success("LHS Done.")
                             except Exception as e: st.error(f"LHS error: {e}"); st.code(traceback.format_exc(),language='python')
                if st.session_state.get('lhs_results_df') is not None:
                    res_lhs=st.session_state['lhs_results_df']; pred_col=f'{tgt_lhs}_predicted'
                    st.subheader("LHS Results"); st.write(f"Top/Bottom 10 Predictions:"); st.markdown("##### Top 10"); st.dataframe(res_lhs.nlargest(10,pred_col).round(4)); st.markdown("##### Bottom 10"); st.dataframe(res_lhs.nsmallest(10,pred_col).round(4))
                    try: csv_l=res_lhs.to_csv(index=False).encode('utf-8-sig'); st.download_button("⬇️ DL LHS Results",csv_l,f'lhs_{mt_lhs}_{tgt_lhs}_{lhs_n}.csv','text/csv',key='dl_lhs')
                    except Exception as e: st.error(f"DL prep error (LHS): {e}")

                    # Plot LHS on Biplot
                    st.subheader("🔬 Project LHS Samples onto Original PCA Biplot")
                    pca_ok=all(comp is not None for comp in [pca_imp,pca_sc,pca_mod,pca_cols,pca_load,pca_var,pca_scores_orig])
                    features_match_pca = pca_ok and pca_cols is not None and list(num_feats_lhs) == list(pca_cols)

                    if not pca_ok: st.warning("Original PCA components/scores missing.")
                    elif not features_match_pca: st.warning(f"LHS model feats != PCA feats. Cannot project."); st.caption(f"Model:{num_feats_lhs}"); st.caption(f"PCA:{pca_cols}")
                    else:
                        st.write("Projecting top/bottom 10 predicted LHS samples over original data (grey)...")
                        try:
                            n_ext=10; plot_all=False; res_lhs=st.session_state['lhs_results_df']; pred_col=f'{tgt_lhs}_predicted'
                            if len(res_lhs)<n_ext*2: st.warning(f"< {n_ext*2} LHS samples, plotting all."); ext_idx=res_lhs.index; plot_all=True
                            else: top_idx=res_lhs.nlargest(n_ext,pred_col).index; bot_idx=res_lhs.nsmallest(n_ext,pred_col).index; ext_idx=top_idx.union(bot_idx)
                            lhs_res_filt=res_lhs.loc[ext_idx]; lhs_pca_in_filt=lhs_res_filt[pca_cols]

                            expected_cols = pca_imp.feature_names_in_
                            try: lhs_pca_in_reordered = lhs_pca_in_filt[expected_cols]
                            except KeyError as ke: st.error(f"Col mismatch PCA transform: {ke}. Exp:{expected_cols}, Got:{list(lhs_pca_in_filt.columns)}"); st.stop()

                            lhs_imp_filt=pca_imp.transform(lhs_pca_in_reordered)
                            lhs_sc_filt=pca_sc.transform(lhs_imp_filt)
                            lhs_scores_filt=pca_mod.transform(lhs_sc_filt)

                            preds_filt=lhs_res_filt[pred_col]; color_cat_filt=pd.Series('Other',index=preds_filt.index)
                            if not plot_all: color_cat_filt[preds_filt.index.isin(top_idx)]='Highest 10'; color_cat_filt[preds_filt.index.isin(bot_idx)]='Lowest 10'; color_lbl=f"Pred '{tgt_lhs}' Extremes"
                            else: color_cat_filt='All LHS'; color_lbl='All LHS Samples'

                            fig_lhs_b=biplot(score=lhs_scores_filt, coeff=pca_load, labels=pca_cols, explained_variance_ratio=pca_var, color_data=color_cat_filt, color_label=color_lbl, is_categorical=True, point_alpha=0.7, make_square=True, background_score=pca_scores_orig)
                            st.pyplot(fig_lhs_b); caption="Top 10 highest/lowest predicted LHS samples" if not plot_all else "All LHS samples"; st.caption(f"{caption} (colored) over original data (grey). Arrows: PCA loadings.")
                        except ValueError as ve: st.error(f"LHS Biplot Error (Value): {ve}"); st.code(traceback.format_exc(), language='python')
                        except Exception as e: st.error(f"LHS Biplot Error: {e}"); st.code(traceback.format_exc(), language='python')

# --- Footer / Initial State ---
if st.session_state.get('df') is None:
     st.info("☝️ **Welcome! Upload a CSV file using the sidebar to begin.**")

# --- Final Sidebar Section ---
st.sidebar.markdown("---"); st.sidebar.subheader("📖 How to Use");
st.sidebar.info("""1. Upload CSV... 10. LHS Sim...""") # Abbreviated