import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm, spearmanr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import subprocess
import os
import json
import re
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from jinja2 import Template


class BlackBoxProblem:
    def __init__(self, name="Rosenbrock"):
        self.name = name
        if name == "Rosenbrock":
            self.n_dim = 10
            self.eff_dim = [0, 1]
            self.bounds = np.array([[-2.0, 2.0]] * self.n_dim)
            self.dim_names = [f"x{i}" for i in range(self.n_dim)]
        elif name == "Hartmann":
            self.n_dim = 20
            self.eff_dim = [0, 1, 2, 3, 4, 5]
            self.bounds = np.array([[0.0, 1.0]] * self.n_dim)
            self.dim_names = [f"x{i}" for i in range(self.n_dim)]
        elif name == "Steel Alloy":
            self.n_dim = 15
            self.eff_dim = [0, 3, 4, 5]  # C, Cr, Ni, Mo が主要な強化元素
            self.dim_names = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Ti", 
                            "Al", "Cu", "Nb", "B", "N", "P", "S"]
            self.bounds = np.array([
                [0.1, 0.5],    # C: 炭素
                [0.3, 2.0],    # Mn: マンガン
                [0.1, 1.0],    # Si: シリコン
                [0.0, 18.0],   # Cr: クロム（主要強化元素）
                [0.0, 12.0],   # Ni: ニッケル（主要強化元素）
                [0.0, 3.0],    # Mo: モリブデン（主要強化元素）
                [0.0, 0.3],    # V: バナジウム
                [0.0, 0.1],    # Ti: チタン
                [0.0, 0.1],    # Al: アルミニウム
                [0.0, 0.5],    # Cu: 銅
                [0.0, 0.1],    # Nb: ニオブ
                [0.0, 0.01],   # B: ホウ素
                [0.0, 0.02],   # N: 窒素
                [0.0, 0.05],   # P: リン（不純物）
                [0.0, 0.05]    # S: 硫黄（不純物）
            ])
        elif name == "Battery Cathode":
            self.n_dim = 12
            self.eff_dim = [0, 1, 2, 3]  # Li, Ni, Co, Mn がNCM組成の主要元素
            self.dim_names = ["Li", "Ni", "Co", "Mn", "Al", "Fe", "Ti", 
                            "Mg", "Zr", "Ca", "Na", "K"]
            self.bounds = np.array([
                [0.9, 1.1],    # Li: リチウム
                [0.3, 0.8],    # Ni: ニッケル（主要元素）
                [0.1, 0.3],    # Co: コバルト（主要元素）
                [0.1, 0.4],    # Mn: マンガン（主要元素）
                [0.0, 0.1],    # Al: アルミニウム（ドーパント）
                [0.0, 0.1],    # Fe: 鉄（ドーパント）
                [0.0, 0.05],   # Ti: チタン（ドーパント）
                [0.0, 0.05],   # Mg: マグネシウム（ドーパント）
                [0.0, 0.05],   # Zr: ジルコニウム（ドーパント）
                [0.0, 0.02],   # Ca: カルシウム（不純物）
                [0.0, 0.02],   # Na: ナトリウム（不純物）
                [0.0, 0.02]    # K: カリウム（不純物）
            ])
        else:
            self.n_dim = 20
            self.eff_dim = [0, 1, 2, 3, 4, 5]
            self.bounds = np.array([[0.0, 1.0]] * self.n_dim)
            self.dim_names = [f"x{i}" for i in range(self.n_dim)]

    def evaluate(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_eff = X[:, self.eff_dim]

        if self.name == "Rosenbrock":
            x = X_eff[:, 0]
            y = X_eff[:, 1]
            val = (1 - x)**2 + 100 * (y - x**2)**2
            return -val
        
        elif self.name == "Steel Alloy":
            C = X_eff[:, 0]   # 炭素
            Cr = X_eff[:, 1]  # クロム
            Ni = X_eff[:, 2]  # ニッケル
            Mo = X_eff[:, 3]  # モリブデン
            
            base_strength = 200.0
            
            C_contrib = 800.0 * C * np.exp(-10.0 * (C - 0.3)**2)
            
            Cr_contrib = 30.0 * Cr
            
            Ni_contrib = 25.0 * Ni
            
            Mo_contrib = 100.0 * Mo * np.exp(-2.0 * (Mo - 1.5)**2)
            
            interaction = 5.0 * Cr * Ni / (1.0 + Cr + Ni)
            
            noise = np.random.normal(0, 10, len(C))
            
            strength = base_strength + C_contrib + Cr_contrib + Ni_contrib + Mo_contrib + interaction + noise
            
            return strength
        
        elif self.name == "Battery Cathode":
            Li = X_eff[:, 0]  # リチウム
            Ni = X_eff[:, 1]  # ニッケル
            Co = X_eff[:, 2]  # コバルト
            Mn = X_eff[:, 3]  # マンガン
            
            base_capacity = 150.0
            
            Li_contrib = 50.0 * np.exp(-10.0 * (Li - 1.0)**2)
            
            Ni_contrib = 200.0 * Ni
            
            Co_contrib = 100.0 * Co
            
            Mn_contrib = 50.0 * Mn
            
            composition_sum = Ni + Co + Mn
            balance_penalty = -100.0 * (composition_sum - 1.0)**2
            
            noise = np.random.normal(0, 5, len(Li))
            
            capacity = base_capacity + Li_contrib + Ni_contrib + Co_contrib + Mn_contrib + balance_penalty + noise
            
            return capacity
        
        else:  # Hartmann
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




class ExternalProgramConfig:
    """Configuration for external program evaluation."""
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        
        self.variables = config_dict.get('variables', [])
        self.template_content = config_dict.get('template_content', '')
        self.template_filename = config_dict.get('template_filename', 'input.txt')
        self.command = config_dict.get('command', '')
        self.output_filename = config_dict.get('output_filename', 'output.txt')
        self.output_regex = config_dict.get('output_regex', r'objective\s*=\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)')
        self.objective_type = config_dict.get('objective_type', 'maximize')
        self.timeout = config_dict.get('timeout', 300)
        self.n_parallel = config_dict.get('n_parallel', 1)
        self.working_dir_base = config_dict.get('working_dir_base', 'runs')
    
    def to_dict(self):
        return {
            'variables': self.variables,
            'template_content': self.template_content,
            'template_filename': self.template_filename,
            'command': self.command,
            'output_filename': self.output_filename,
            'output_regex': self.output_regex,
            'objective_type': self.objective_type,
            'timeout': self.timeout,
            'n_parallel': self.n_parallel,
            'working_dir_base': self.working_dir_base
        }
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)


class ExternalProgramEvaluator:
    """Evaluator for external programs with parallel execution support."""
    
    def __init__(self, config):
        self.config = config
        self.n_dim = len(config.variables)
        self.eff_dim = list(range(self.n_dim))
        self.name = "External Program"
        
        self.bounds = np.array([[var['lower'], var['upper']] for var in config.variables])
        
        Path(config.working_dir_base).mkdir(parents=True, exist_ok=True)
        
        self.output_pattern = re.compile(config.output_regex)
    
    def _create_run_dir(self, iteration, point_id):
        """Create a unique run directory for this evaluation."""
        timestamp = int(time.time() * 1000)
        run_dir = Path(self.config.working_dir_base) / f"iter_{iteration:04d}_point_{point_id:04d}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def _render_template(self, x, run_dir):
        """Render input file from template using Jinja2."""
        context = {}
        for i, var in enumerate(self.config.variables):
            context[var['name']] = float(x[i])
            context[f'x{i}'] = float(x[i])
        
        template = Template(self.config.template_content)
        rendered = template.render(**context)
        
        input_file = run_dir / self.config.template_filename
        with open(input_file, 'w') as f:
            f.write(rendered)
        
        return input_file
    
    def _run_external(self, run_dir):
        """Execute external command in the run directory."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                self.config.command,
                shell=True,
                cwd=str(run_dir),
                timeout=self.config.timeout,
                capture_output=True,
                text=True
            )
            
            duration = time.time() - start_time
            
            with open(run_dir / 'stdout.txt', 'w') as f:
                f.write(result.stdout)
            with open(run_dir / 'stderr.txt', 'w') as f:
                f.write(result.stderr)
            
            return {
                'returncode': result.returncode,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                'returncode': -1,
                'duration': duration,
                'stdout': '',
                'stderr': f'Timeout after {self.config.timeout}s'
            }
        
        except Exception as e:
            duration = time.time() - start_time
            return {
                'returncode': -2,
                'duration': duration,
                'stdout': '',
                'stderr': str(e)
            }
    
    def _parse_output(self, run_dir):
        """Parse output file to extract objective value."""
        output_file = run_dir / self.config.output_filename
        
        if not output_file.exists():
            raise FileNotFoundError(f"Output file not found: {output_file}")
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        match = self.output_pattern.search(content)
        
        if not match:
            raise ValueError(f"Could not find objective value matching pattern: {self.config.output_regex}")
        
        value = float(match.group(1))
        
        if self.config.objective_type == 'minimize':
            value = -value
        
        return value
    
    def _evaluate_single(self, args):
        """Evaluate a single point (for parallel execution)."""
        x, iteration, point_id = args
        
        run_dir = self._create_run_dir(iteration, point_id)
        
        try:
            self._render_template(x, run_dir)
            
            run_result = self._run_external(run_dir)
            
            if run_result['returncode'] == 0:
                y = self._parse_output(run_dir)
                success = True
                error_msg = None
            else:
                y = -np.inf if self.config.objective_type == 'maximize' else np.inf
                success = False
                error_msg = run_result['stderr']
            
            return {
                'x': x,
                'y': y,
                'success': success,
                'run_dir': str(run_dir),
                'returncode': run_result['returncode'],
                'duration': run_result['duration'],
                'error_msg': error_msg
            }
        
        except Exception as e:
            return {
                'x': x,
                'y': -np.inf if self.config.objective_type == 'maximize' else np.inf,
                'success': False,
                'run_dir': str(run_dir),
                'returncode': -3,
                'duration': 0,
                'error_msg': str(e)
            }
    
    def evaluate(self, X, iteration=0):
        """Evaluate multiple points in parallel."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        n_points = X.shape[0]
        
        args_list = [(X[i], iteration, i) for i in range(n_points)]
        
        if self.config.n_parallel > 1 and n_points > 1:
            with Pool(processes=min(self.config.n_parallel, n_points)) as pool:
                results = pool.map(self._evaluate_single, args_list)
        else:
            results = [self._evaluate_single(args) for args in args_list]
        
        y_values = np.array([r['y'] for r in results])
        
        return y_values, results

def expected_improvement(X, gp, y_max, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 1e-9)

    with np.errstate(divide='warn', invalid='warn'):
        imp = mu - y_max - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0

    return ei


def _optimize_single_restart(args):
    """Single restart optimization for parallel execution."""
    x0, bounds, acquisition, gp, y_max = args
    
    def min_obj(X):
        return -acquisition(X.reshape(1, -1), gp, y_max)
    
    res = minimize(min_obj, x0, bounds=bounds, method='L-BFGS-B')
    return res.fun, res.x


def propose_location(acquisition, gp, y_max, bounds, n_restarts=25, n_threads=None):
    """
    Propose the next sampling location by maximizing the acquisition function.
    Uses thread parallelization for faster optimization with multiple restarts.
    
    Parameters:
    - acquisition: Acquisition function (e.g., expected_improvement)
    - gp: Fitted Gaussian Process model
    - y_max: Current best observed value
    - bounds: Array of (lower, upper) bounds for each dimension
    - n_restarts: Number of random restarts for optimization
    - n_threads: Number of threads for parallel optimization (default: min(n_restarts, cpu_count))
    """
    dim = bounds.shape[0]
    
    if n_threads is None:
        n_threads = min(n_restarts, cpu_count())
    
    x0_list = [np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim) 
               for _ in range(n_restarts)]
    
    args_list = [(x0, bounds, acquisition, gp, y_max) for x0 in x0_list]
    
    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            results = list(executor.map(_optimize_single_restart, args_list))
    else:
        results = [_optimize_single_restart(args) for args in args_list]
    
    best_idx = np.argmin([r[0] for r in results])
    return results[best_idx][1]




def get_length_scales(gp_model, n_dim):
    """Extract length scales from GP model, handling both ARD and isotropic kernels."""
    try:
        length_scales = gp_model.kernel_.k2.length_scale
    except Exception:
        length_scales = getattr(gp_model.kernel_, "length_scale", None)

    length_scales = np.atleast_1d(np.array(length_scales, dtype=float)).ravel()
    if length_scales.size == 1:
        length_scales = np.full(n_dim, float(length_scales[0]))
    elif length_scales.size != n_dim:
        length_scales = np.resize(length_scales, n_dim)

    return length_scales


def get_constant_value(gp_model):
    """Extract constant value from GP kernel."""
    try:
        return float(gp_model.kernel_.k1.constant_value)
    except Exception:
        return 1.0


def plot_relevance_comparison(X_train, y_train, gp_model, problem):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_train_scaled, y_train.ravel())
    linear_importance = np.abs(lasso.coef_)

    length_scales = get_length_scales(gp_model, problem.n_dim)

    ard_importance = np.divide(1.0, length_scales, out=np.zeros_like(length_scales), where=length_scales > 0)
    max_ard = np.max(ard_importance)
    if not np.isfinite(max_ard) or max_ard <= 0:
        max_ard = 1.0
    ard_importance = ard_importance / (max_ard + 1e-12)

    max_linear = np.max(linear_importance)
    if not np.isfinite(max_linear) or max_linear <= 0:
        max_linear = 1.0
    linear_importance = linear_importance / (max_linear + 1e-12)

    dim_labels = [f"x{i}" for i in range(problem.n_dim)]
    colors_linear = ['red' if i in problem.eff_dim else 'lightgray' for i in range(problem.n_dim)]
    colors_ard = ['blue' if i in problem.eff_dim else 'lightgray' for i in range(problem.n_dim)]

    linear_importance = linear_importance.tolist()
    ard_importance = ard_importance.tolist()

    assert len(ard_importance) == len(dim_labels), "Length-scale-derived relevance must match number of dimensions"
    assert len(linear_importance) == len(dim_labels), "Lasso coefficients must match number of dimensions"

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
    """Plot optimization progress showing both observed values and best so far."""
    iterations = list(range(len(history)))
    best_so_far = np.maximum.accumulate(history)

    fig = go.Figure()

    # Add observed values
    fig.add_trace(go.Scatter(
        x=iterations,
        y=history,
        mode='markers',
        name='Observed y',
        marker=dict(size=6, color='lightblue'),
        hovertemplate='Iteration: %{x}<br>Observed y: %{y:.4f}<extra></extra>'
    ))

    # Add best so far line
    fig.add_trace(go.Scatter(
        x=iterations,
        y=best_so_far,
        mode='lines+markers',
        name='Best so far',
        line=dict(width=2, color='darkblue'),
        marker=dict(size=8),
        hovertemplate='Iteration: %{x}<br>Best f(x): %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Optimization Progress: {method_name}",
        xaxis_title="Iteration",
        yaxis_title="f(x)",
        hovermode='closest',
        height=400,
        legend=dict(x=0.02, y=0.98)
    )

    return fig




def plot_ard_evolution(iteration_log, problem):
    """Plot heatmap showing how ARD relevance evolves over iterations."""
    if not iteration_log:
        return None

    iterations = []
    ard_matrix = []

    for log in iteration_log:
        if 'ard_importance' in log and log['ard_importance'] is not None:
            iterations.append(log['iteration'])
            ard_matrix.append(log['ard_importance'])

    if not ard_matrix:
        return None

    ard_matrix = np.array(ard_matrix).T
    dim_labels = [f"x{i}" for i in range(problem.n_dim)]

    fig = go.Figure(data=go.Heatmap(
        z=ard_matrix,
        x=iterations,
        y=dim_labels,
        colorscale='YlOrRd',
        zmin=0,
        zmax=1,
        colorbar=dict(
            title=dict(
                text="Relevance<br>(1/ℓ)",
                side="right"
            )
        ),
        hovertemplate='Iteration: %{x}<br>Dimension: %{y}<br>Relevance: %{z:.3f}<extra></extra>'
    ))

    # Highlight effective dimensions with blue boxes
    for i in problem.eff_dim:
        fig.add_shape(
            type="rect",
            x0=min(iterations) - 0.5,
            x1=max(iterations) + 0.5,
            y0=i - 0.5,
            y1=i + 0.5,
            line=dict(color="blue", width=2),
            fillcolor="rgba(0,0,0,0)"
        )

    fig.update_layout(
        title="ARD Evolution: Dimension Relevance Over Time",
        xaxis_title="Iteration",
        yaxis_title="Dimension",
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


def plot_1d_sweep(problem, X_train, y_train):
    """
    Create 1D sweep plots showing which dimensions actually affect the output.
    Fix all variables at the best point found so far, then sweep each dimension.
    """
    if len(y_train) == 0:
        return None
    
    # Use the best point found so far as the base point
    best_idx = np.argmax(y_train)
    x_base = X_train[best_idx].copy()
    
    # Create subplots for a few dimensions (show first 6)
    n_dims_to_show = min(6, problem.n_dim)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"x{i}" + (" (有効)" if i in problem.eff_dim else " (無効)") 
                       for i in range(n_dims_to_show)]
    )
    
    for idx in range(n_dims_to_show):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        # Sweep this dimension
        x_sweep = np.linspace(problem.bounds[idx, 0], problem.bounds[idx, 1], 50)
        y_sweep = []
        
        for x_val in x_sweep:
            x_test = x_base.copy()
            x_test[idx] = x_val
            y_test = problem.evaluate(x_test.reshape(1, -1))[0]
            y_sweep.append(y_test)
        
        # Plot the sweep
        color = 'blue' if idx in problem.eff_dim else 'lightgray'
        fig.add_trace(
            go.Scatter(x=x_sweep, y=y_sweep, mode='lines', 
                      line=dict(color=color, width=2),
                      showlegend=False),
            row=row, col=col
        )
        
        # Mark the current value
        fig.add_trace(
            go.Scatter(x=[x_base[idx]], y=[y_train[best_idx]], 
                      mode='markers', marker=dict(color='red', size=10),
                      showlegend=False),
            row=row, col=col
        )
    
    fig.update_layout(
        title="1次元スイープ: 各次元を動かしたときの出力変化",
        height=500,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="変数値")
    fig.update_yaxes(title_text="f(x)")
    
    return fig


def plot_length_scale_timeseries(iteration_log, problem, mode='ell', normalize=True, log_y=False, top_k=None):
    """
    Plot time series of length scales over iterations.
    
    Parameters:
    - mode: 'ell' (ℓ), 'inv_ell' (1/ℓ), or 'inv_ell_sq' (1/ℓ²)
    - normalize: normalize by domain width for comparability
    - log_y: use log scale for y-axis
    - top_k: show only top k dimensions by final relevance
    """
    if not iteration_log:
        return None
    
    # Extract length scales from log
    iterations = []
    length_scales_matrix = []
    
    for log in iteration_log:
        if 'length_scales' in log and log['length_scales'] is not None:
            iterations.append(log['iteration'])
            length_scales_matrix.append(log['length_scales'])
    
    if not length_scales_matrix:
        return None
    
    length_scales_matrix = np.array(length_scales_matrix)  # shape: (n_iters, n_dims)
    
    # Normalize by domain width if requested
    if normalize:
        domain_widths = problem.bounds[:, 1] - problem.bounds[:, 0]
        length_scales_matrix = length_scales_matrix / domain_widths
        y_label_suffix = " (normalized by domain width)"
    else:
        y_label_suffix = ""
    
    # Transform based on mode
    if mode == 'inv_ell':
        data_matrix = np.divide(1.0, length_scales_matrix, 
                               out=np.zeros_like(length_scales_matrix), 
                               where=length_scales_matrix > 0)
        y_label = f"1/ℓ{y_label_suffix}"
    elif mode == 'inv_ell_sq':
        data_matrix = np.divide(1.0, length_scales_matrix**2, 
                               out=np.zeros_like(length_scales_matrix), 
                               where=length_scales_matrix > 0)
        y_label = f"1/ℓ²{y_label_suffix}"
    else:  # mode == 'ell'
        data_matrix = length_scales_matrix
        y_label = f"ℓ{y_label_suffix}"
    
    # Select top k dimensions if requested
    if top_k is not None and top_k < problem.n_dim:
        # Use final relevance (1/ℓ) to select top k
        final_relevance = 1.0 / (length_scales_matrix[-1] + 1e-12)
        top_indices = np.argsort(final_relevance)[-top_k:][::-1]
    else:
        top_indices = range(problem.n_dim)
    
    # Create plot
    fig = go.Figure()
    
    for idx in top_indices:
        color = 'blue' if idx in problem.eff_dim else 'lightgray'
        line_width = 2 if idx in problem.eff_dim else 1
        
        fig.add_trace(go.Scatter(
            x=iterations,
            y=data_matrix[:, idx],
            mode='lines+markers',
            name=f"x{idx}" + (" (有効)" if idx in problem.eff_dim else ""),
            line=dict(color=color, width=line_width),
            marker=dict(size=4),
            hovertemplate=f'x{idx}<br>Iteration: %{{x}}<br>{y_label}: %{{y:.4f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Length Scale Transitions: {y_label}",
        xaxis_title="Iteration",
        yaxis_title=y_label,
        height=500,
        hovermode='closest',
        legend=dict(x=1.02, y=1, xanchor='left')
    )
    
    if log_y:
        fig.update_yaxes(type='log')
    
    return fig



def plot_summary_statistics(X, y, problem):
    """Display summary statistics table for the dataset"""
    dim_labels = [f"x{i}" for i in range(problem.n_dim)]
    
    df = pd.DataFrame(X, columns=dim_labels)
    df['Objective'] = y
    
    stats = df.describe().T
    stats['range'] = stats['max'] - stats['min']
    stats = stats[['mean', 'std', 'min', 'max', 'range']]
    
    # Highlight effective dimensions
    def highlight_effective(row):
        dim_idx = int(row.name[1:]) if row.name != 'Objective' else -1
        if dim_idx in problem.eff_dim:
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
    
    styled_df = stats.style.apply(highlight_effective, axis=1).format("{:.4f}")
    
    return styled_df


def plot_correlation_heatmap(X, y, problem, method='spearman'):
    """Plot correlation heatmap with feature-feature and feature-target correlations"""
    dim_labels = [f"x{i}" for i in range(problem.n_dim)]
    
    df = pd.DataFrame(X, columns=dim_labels)
    
    if method == 'spearman':
        corr_matrix = df.corr(method='spearman')
        target_corr = np.array([spearmanr(X[:, i], y)[0] for i in range(problem.n_dim)])
    else:
        corr_matrix = df.corr(method='pearson')
        target_corr = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(problem.n_dim)])
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=dim_labels,
        y=dim_labels,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title=dict(text=f"{method.capitalize()}<br>Correlation", side="right"))
    ))
    
    fig.update_layout(
        title=f"Feature-Feature Correlation ({method.capitalize()})",
        xaxis_title="Dimension",
        yaxis_title="Dimension",
        height=500,
        width=600
    )
    
    colors = ['red' if i in problem.eff_dim else 'lightgray' for i in range(problem.n_dim)]
    
    fig_target = go.Figure(data=go.Bar(
        x=dim_labels,
        y=target_corr,
        marker_color=colors,
        hovertemplate='Dimension: %{x}<br>Correlation: %{y:.3f}<extra></extra>'
    ))
    
    fig_target.update_layout(
        title=f"Feature-Target Correlation ({method.capitalize()})",
        xaxis_title="Dimension",
        yaxis_title=f"{method.capitalize()} Correlation with Objective",
        height=400,
        showlegend=False
    )
    
    return fig, fig_target, target_corr


def plot_scatter_matrix(X, y, problem, selected_dims=None):
    """Plot scatter matrix for selected dimensions"""
    if selected_dims is None or len(selected_dims) == 0:
        selected_dims = problem.eff_dim[:min(6, len(problem.eff_dim))]
    
    selected_dims = list(selected_dims)
    n_dims = len(selected_dims)
    
    if n_dims < 2:
        st.warning("少なくとも2つの次元を選択してください")
        return None
    
    dim_labels = [f"x{i}" for i in selected_dims]
    
    # Create subplot grid
    fig = make_subplots(
        rows=n_dims, cols=n_dims,
        subplot_titles=[f"{dim_labels[j]} vs {dim_labels[i]}" if i != j else f"{dim_labels[i]}" 
                       for i in range(n_dims) for j in range(n_dims)],
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    colorscale = 'Viridis'
    
    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                fig.add_trace(
                    go.Histogram(x=X[:, selected_dims[i]], nbinsx=20, 
                               marker_color='lightblue', showlegend=False),
                    row=i+1, col=j+1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=X[:, selected_dims[j]],
                        y=X[:, selected_dims[i]],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=y,
                            colorscale=colorscale,
                            showscale=(i == 0 and j == n_dims - 1),
                            colorbar=dict(title="Objective", x=1.1) if (i == 0 and j == n_dims - 1) else None
                        ),
                        showlegend=False,
                        hovertemplate=f'{dim_labels[j]}: %{{x:.3f}}<br>{dim_labels[i]}: %{{y:.3f}}<br>Objective: %{{marker.color:.3f}}<extra></extra>'
                    ),
                    row=i+1, col=j+1
                )
            
            if i == n_dims - 1:
                fig.update_xaxes(title_text=dim_labels[j], row=i+1, col=j+1)
            if j == 0:
                fig.update_yaxes(title_text=dim_labels[i], row=i+1, col=j+1)
    
    fig.update_layout(
        title="Scatter Plot Matrix (色 = 目的関数値)",
        height=150 * n_dims,
        showlegend=False
    )
    
    return fig


def plot_pca_biplot(X, y, problem, show_all_labels=False):
    """Plot PCA biplot with scores and loadings"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    
    fig_var = go.Figure(data=go.Bar(
        x=[f"PC{i+1}" for i in range(len(explained_var))],
        y=explained_var * 100,
        marker_color='steelblue'
    ))
    
    fig_var.update_layout(
        title="PCA Explained Variance Ratio",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        height=300
    )
    
    fig_biplot = go.Figure()
    
    fig_biplot.add_trace(go.Scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=y,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Objective")
        ),
        name='Samples',
        hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Objective: %{marker.color:.3f}<extra></extra>'
    ))
    
    loadings = pca.components_[:2, :].T * np.sqrt(pca.explained_variance_[:2])
    
    max_score = max(np.abs(X_pca[:, :2]).max(), 1e-6)
    max_loading = max(np.abs(loadings).max(), 1e-6)
    scale_factor = 0.8 * max_score / max_loading
    
    loadings_scaled = loadings * scale_factor
    
    dim_labels = [f"x{i}" for i in range(problem.n_dim)]
    
    loading_magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    threshold = np.percentile(loading_magnitudes, 70) if not show_all_labels else 0
    
    for i in range(problem.n_dim):
        arrow_color = 'red' if i in problem.eff_dim else 'blue'
        arrow_width = 3 if i in problem.eff_dim else 1.5
        
        fig_biplot.add_trace(go.Scatter(
            x=[0, loadings_scaled[i, 0]],
            y=[0, loadings_scaled[i, 1]],
            mode='lines',
            line=dict(color=arrow_color, width=arrow_width),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_biplot.add_trace(go.Scatter(
            x=[loadings_scaled[i, 0]],
            y=[loadings_scaled[i, 1]],
            mode='markers',
            marker=dict(symbol='arrow', size=10, color=arrow_color, angleref='previous'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        if loading_magnitudes[i] >= threshold or show_all_labels:
            fig_biplot.add_annotation(
                x=loadings_scaled[i, 0],
                y=loadings_scaled[i, 1],
                text=dim_labels[i],
                showarrow=False,
                font=dict(size=10, color=arrow_color),
                xshift=10 if loadings_scaled[i, 0] > 0 else -10,
                yshift=10 if loadings_scaled[i, 1] > 0 else -10
            )
    
    fig_biplot.update_layout(
        title=f"PCA Biplot (PC1: {explained_var[0]*100:.1f}%, PC2: {explained_var[1]*100:.1f}%, Cumulative: {(explained_var[0]+explained_var[1])*100:.1f}%)",
        xaxis_title=f"PC1 ({explained_var[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({explained_var[1]*100:.1f}%)",
        height=600,
        width=800,
        hovermode='closest'
    )
    
    fig_biplot.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Effective dims',
        showlegend=True
    ))
    
    fig_biplot.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Other dims',
        showlegend=True
    ))
    
    return fig_var, fig_biplot, explained_var


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
        ["Rosenbrock", "Hartmann", "Steel Alloy", "Battery Cathode", "External Program"],
        help="Rosenbrock: 10次元空間に埋め込まれた2次元関数\nHartmann: 20次元空間に埋め込まれた6次元関数\nSteel Alloy: 鋼合金の引張強度最適化\nBattery Cathode: リチウムイオン電池正極材料の容量最適化"
    )

    method = st.selectbox(
        "Optimization Method",
        ["Random Search", "Standard BO", "Sparse BO (ARD)"],
        help="Random: ランダムサンプリング\nStandard BO: 通常のベイズ最適化\nSparse BO: ARDカーネルを使用"
    )

    n_initial = st.slider("Initial Samples", 5, 20, 10,
                         help="初期ランダムサンプル数")
    n_iterations = st.slider("Optimization Iterations", 10, 500, 20,
                            help="最適化ループの回数（200以上必要な場合もあります）")

# External Program Configuration
if prob_name == "External Program":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 External Program Config")
    
    if 'ext_config' not in st.session_state:
        st.session_state.ext_config = ExternalProgramConfig()
    
    max_cpus = cpu_count()
    st.session_state.ext_config.n_parallel = st.sidebar.slider(
        "Parallel CPUs",
        1, max_cpus, min(4, max_cpus),
        help=f"並列実行するCPU数（最大: {max_cpus}）"
    )
    
    st.session_state.ext_config.timeout = st.sidebar.number_input(
        "Timeout (seconds)",
        min_value=1,
        max_value=3600,
        value=300,
        help="各実行のタイムアウト時間"
    )

if prob_name == "External Program":
    with st.expander("📝 External Program Configuration", expanded=True):
        st.markdown("""
        外部プログラム（Thermo-Calcなど）と連携して最適化を実行します。
        
        **設定手順:**
        1. 変数を定義（名前、範囲）
        2. 入力テンプレートを作成（Jinja2形式）
        3. 実行コマンドを指定
        4. 出力パーシング設定（正規表現）
        5. テスト実行で動作確認
        """)
        
        st.subheader("1. Variables Definition")
        n_vars = st.number_input("Number of variables", min_value=1, max_value=20, value=2)
        
        variables = []
        for i in range(n_vars):
            col1, col2, col3 = st.columns(3)
            with col1:
                var_name = st.text_input(f"Variable {i} name", value=f"x{i}", key=f"var_name_{i}")
            with col2:
                var_lower = st.number_input(f"Lower bound", value=0.0, key=f"var_lower_{i}")
            with col3:
                var_upper = st.number_input(f"Upper bound", value=1.0, key=f"var_upper_{i}")
            
            variables.append({
                'name': var_name,
                'lower': var_lower,
                'upper': var_upper
            })
        
        st.session_state.ext_config.variables = variables
        
        st.subheader("2. Input Template (Jinja2)")
        st.markdown("テンプレート内で変数を参照するには `{{ variable_name }}` または `{{ x0 }}`, `{{ x1 }}` などを使用します。")
        
        template_content = st.text_area(
            "Template content",
            value=st.session_state.ext_config.template_content,
            height=150,
            help="Jinja2形式のテンプレート"
        )
        st.session_state.ext_config.template_content = template_content
        
        template_filename = st.text_input(
            "Input filename",
            value=st.session_state.ext_config.template_filename,
            help="生成される入力ファイル名"
        )
        st.session_state.ext_config.template_filename = template_filename
        
        st.subheader("3. Execution Command")
        command = st.text_input(
            "Command",
            value=st.session_state.ext_config.command,
            help="実行するコマンド（例: ./run_simulation.sh）",
            placeholder="./run_thermo_calc.sh"
        )
        st.session_state.ext_config.command = command
        
        st.subheader("4. Output Parsing")
        output_filename = st.text_input(
            "Output filename",
            value=st.session_state.ext_config.output_filename,
            help="パーシングする出力ファイル名"
        )
        st.session_state.ext_config.output_filename = output_filename
        
        output_regex = st.text_input(
            "Regex pattern",
            value=st.session_state.ext_config.output_regex,
            help="目的関数を抽出する正規表現（1つのキャプチャグループ）"
        )
        st.session_state.ext_config.output_regex = output_regex
        
        objective_type = st.selectbox(
            "Objective",
            ["maximize", "minimize"],
            index=0 if st.session_state.ext_config.objective_type == "maximize" else 1
        )
        st.session_state.ext_config.objective_type = objective_type
        
        st.subheader("5. Test Run")
        if st.button("🧪 Test with sample values"):
            with st.spinner("Testing..."):
                try:
                    test_evaluator = ExternalProgramEvaluator(st.session_state.ext_config)
                    test_x = np.array([(var['lower'] + var['upper']) / 2 for var in variables])
                    
                    st.write("**Test input:**")
                    for i, var in enumerate(variables):
                        st.write(f"- {var['name']}: {test_x[i]:.4f}")
                    
                    y_values, results = test_evaluator.evaluate(test_x.reshape(1, -1), iteration=9999)
                    result = results[0]
                    
                    if result['success']:
                        st.success(f"✅ Test successful! Objective value: {result['y']:.6f}")
                        st.write(f"- Run directory: `{result['run_dir']}`")
                        st.write(f"- Duration: {result['duration']:.2f}s")
                    else:
                        st.error(f"❌ Test failed!")
                        st.write(f"- Run directory: `{result['run_dir']}`")
                        st.write(f"- Error: {result['error_msg']}")
                except Exception as e:
                    st.error(f"❌ Test error: {str(e)}")

if prob_name == "External Program":
    problem = ExternalProgramEvaluator(st.session_state.ext_config)
    run_button = st.button("🚀 Run Optimization", type="primary", 
                          disabled=len(st.session_state.ext_config.command) == 0)
else:
    problem = BlackBoxProblem(prob_name)
    run_button = st.button("🚀 Run Optimization", type="primary")

if prob_name == "Steel Alloy":
    eff_names = [problem.dim_names[i] for i in problem.eff_dim]
    st.info(f"""
    **現在の設定: 鋼合金組成の最適化**
    - **目的**: 引張強度（MPa）の最大化
    - **全元素数**: {problem.n_dim}個
    - **有効元素（主要強化元素）**: {len(problem.eff_dim)}個 ({', '.join(eff_names)})
    - **最適化手法**: {method}
    
    **有効元素の役割:**
    - **C (炭素)**: 最も重要な強化元素。最適値0.3%付近で最大強度
    - **Cr (クロム)**: 固溶強化と耐食性向上。線形的に強度増加
    - **Ni (ニッケル)**: 靭性と強度の向上。Crとの相互作用あり
    - **Mo (モリブデン)**: 析出強化。最適値1.5%付近で効果最大
    
    **その他の元素**: Mn, Si, V, Ti, Al, Cu, Nb, B, N, P, S は微量添加元素や不純物で、強度への影響は小さい
    
    **ARDの期待動作**: データから学習して、C, Cr, Ni, Mo の重要度が高くなることを確認してください。
    """)
elif prob_name == "Battery Cathode":
    eff_names = [problem.dim_names[i] for i in problem.eff_dim]
    st.info(f"""
    **現在の設定: リチウムイオン電池正極材料の最適化**
    - **目的**: 比容量（mAh/g）の最大化
    - **全元素数**: {problem.n_dim}個
    - **有効元素（NCM組成）**: {len(problem.eff_dim)}個 ({', '.join(eff_names)})
    - **最適化手法**: {method}
    
    **有効元素の役割:**
    - **Li (リチウム)**: 電池の基本元素。最適値1.0付近
    - **Ni (ニッケル)**: 高容量化の鍵。含有量が多いほど容量増加
    - **Co (コバルト)**: 構造安定性と容量のバランス
    - **Mn (マンガン)**: 安定性向上。容量は低めだが重要
    
    **組成バランス**: Ni + Co + Mn ≈ 1.0 が理想的なNCM組成
    
    **その他の元素**: Al, Fe, Ti, Mg, Zr はドーパント、Ca, Na, K は不純物で、容量への影響は小さい
    
    **ARDの期待動作**: データから学習して、Li, Ni, Co, Mn の重要度が高くなることを確認してください。
    """)
else:
    st.info(f"""
    **現在の設定:**
    - 関数: {prob_name}
    - 全次元数: {problem.n_dim}
    - **有効次元（正解ラベル）**: {len(problem.eff_dim)}個 (Index: {problem.eff_dim})
    - 手法: {method}
    
    **「有効次元」とは？**
    このベンチマーク関数が**実際に参照している変数**のことです。
    - Rosenbrock: 10次元のうち x₀, x₁ の2つだけが出力に影響（コード20行目で定義）
    - Hartmann: 20次元のうち x₀〜x₅ の6つだけが出力に影響（コード24行目で定義）
    - 他の次元をどう変えても出力は変わりません（コード31行目: X_eff = X[:, self.eff_dim]）
    
    **目的:** ARDが、データから「どの次元が効いているか」を学習し、この正解ラベルに近づくかを観察してください。
    """)

if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.gp_model = None
    st.session_state.history = None
    st.session_state.iteration_log = []

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

    iteration_log = []
    
    # Log initial samples
    for i in range(len(y_train)):
        log_entry = {
            'iteration': i,
            'x_new': X_train[i].tolist(),
            'y_new': float(y_train[i]),
            'best_so_far': float(np.max(y_train[:i+1])),
            'mu': None,
            'sigma': None,
            'ei': None,
            'constant_value': None,
            'length_scales': None,
            'ard_importance': None
        }
        iteration_log.append(log_entry)

    if method == "Random Search":
        for i in range(n_iterations):
            x_new = np.random.uniform(
                problem.bounds[:, 0],
                problem.bounds[:, 1],
                size=problem.n_dim
            )
            y_new = problem.evaluate(x_new.reshape(1, -1))[0]

            log_entry = {
                'iteration': len(y_train),
                'x_new': x_new.tolist(),
                'y_new': float(y_new),
                'best_so_far': float(max(np.max(y_train), y_new)),
                'mu': None,
                'sigma': None,
                'ei': None,
                'constant_value': None,
                'length_scales': None,
                'ard_importance': None
            }

            X_train = np.vstack([X_train, x_new])
            y_train = np.append(y_train, y_new)
            history.append(np.max(y_train))
            iteration_log.append(log_entry)

            progress_bar.progress((i + 1) / n_iterations)
            status_text.text(f"Iteration {i+1}/{n_iterations}: Best f(x) = {np.max(y_train):.4f}")

        gp_model = None

    else:
        # Normalize X to [0,1] for ARD to work correctly across different feature scales
        bounds_range = problem.bounds[:, 1] - problem.bounds[:, 0]
        bounds_min = problem.bounds[:, 0]
        
        def normalize_X(X):
            """Normalize X to [0,1] using problem bounds"""
            return (X - bounds_min) / bounds_range
        
        def denormalize_X(X_norm):
            """Denormalize X from [0,1] back to original space"""
            return X_norm * bounds_range + bounds_min
        
        # Normalize training data
        X_train_norm = normalize_X(X_train)
        
        if method == "Sparse BO (ARD)":
            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                     Matern(length_scale=[1.0] * problem.n_dim,
                           length_scale_bounds=(1e-3, 1e3),
                           nu=2.5)
        else:
            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                     Matern(length_scale=1.0,
                           length_scale_bounds=(1e-3, 1e3),
                           nu=2.5)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )

        for i in range(n_iterations):
            gp.fit(X_train_norm, y_train)

            bounds_norm = np.array([[0.0, 1.0]] * problem.n_dim)
            x_new_norm = propose_location(
                expected_improvement,
                gp,
                np.max(y_train),
                bounds_norm,
                n_restarts=25
            )
            
            x_new = denormalize_X(x_new_norm)

            # Get GP predictions and EI for logging (in normalized space)
            mu, sigma = gp.predict(x_new_norm.reshape(1, -1), return_std=True)
            ei = expected_improvement(x_new_norm.reshape(1, -1), gp, np.max(y_train))

            y_new = problem.evaluate(x_new.reshape(1, -1))[0]

            # Get length scales and compute ARD importance
            length_scales = get_length_scales(gp, problem.n_dim)
            constant_value = get_constant_value(gp)
            ard_importance = np.divide(1.0, length_scales, out=np.zeros_like(length_scales), where=length_scales > 0)
            max_ard = np.max(ard_importance)
            if np.isfinite(max_ard) and max_ard > 0:
                ard_importance = ard_importance / max_ard

            log_entry = {
                'iteration': len(y_train),
                'x_new': x_new.tolist(),
                'y_new': float(y_new),
                'best_so_far': float(max(np.max(y_train), y_new)),
                'mu': float(mu[0]),
                'sigma': float(sigma[0]),
                'ei': float(ei[0]),
                'constant_value': float(constant_value),
                'length_scales': length_scales.tolist(),
                'ard_importance': ard_importance.tolist()
            }

            X_train = np.vstack([X_train, x_new])
            X_train_norm = np.vstack([X_train_norm, x_new_norm])
            y_train = np.append(y_train, y_new)
            history.append(np.max(y_train))
            iteration_log.append(log_entry)

            progress_bar.progress((i + 1) / n_iterations)
            status_text.text(f"Iteration {i+1}/{n_iterations}: Best f(x) = {np.max(y_train):.4f}")

        gp_model = gp

    st.session_state.optimization_done = True
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.gp_model = gp_model
    st.session_state.history = history
    st.session_state.iteration_log = iteration_log
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
    iteration_log = st.session_state.iteration_log
    method = st.session_state.method
    problem = st.session_state.problem

    st.markdown("---")
    st.header("📊 Analysis Results")

    st.markdown("---")
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    st.markdown("""
    **なぜEDAが重要か？**
    
    最適化の前に、データの性質を確認することで「単純な相関係数」がどうなっているかを見ます。
    これは「相関係数が低くても（0に近い）、実は重要な変数がある」ことを示す前振りです。
    
    **Lasso（線形モデル）の限界:**
    - Lassoは $y = w_1x_1 + w_2x_2 + ...$ という線形性を仮定
    - Rosenbrock関数のような強い相互作用（$x_1$ と $x_2$ が絡み合う）を持つ非線形問題では、
      重要な変数を「重要ではない（係数0）」と誤判定することがあります
    
    **GP-ARD（非線形モデル）の強み:**
    - ARDは局所的な距離の変化を見るため、非線形な関係を正しく捉えられます
    - 長さスケール $\\ell_d$ が大きい次元は無視され、小さい次元は重要と判断されます
    """)
    
    eda_dataset = st.radio(
        "データセット選択",
        ["初期サンプルのみ", "全データ（初期 + BO探索）"],
        index=1,
        help="初期サンプルのみ: 初期設計の品質を確認\n全データ: BOによる探索領域の変化を確認"
    )
    
    if eda_dataset == "初期サンプルのみ":
        X_eda = X_train[:n_initial]
        y_eda = y_train[:n_initial]
        st.info(f"初期サンプル {n_initial} 点を使用")
    else:
        X_eda = X_train
        y_eda = y_train
        st.info(f"全 {len(y_train)} 点を使用（初期 {n_initial} + BO探索 {len(y_train)-n_initial}）")
    
    with st.expander("📊 要約統計量", expanded=False):
        st.markdown("各次元の基本統計量です。赤色の行は有効次元（正解ラベル）を示します。")
        styled_stats = plot_summary_statistics(X_eda, y_eda, problem)
        st.dataframe(styled_stats, use_container_width=True)
    
    with st.expander("🔗 相関分析", expanded=False):
        corr_method = st.radio(
            "相関係数の種類",
            ["Spearman", "Pearson"],
            index=0,
            help="Spearman: 単調な非線形関係も捉える\nPearson: 線形関係のみ"
        )
        
        fig_corr, fig_target_corr, target_corr = plot_correlation_heatmap(
            X_eda, y_eda, problem, method=corr_method.lower()
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_corr, use_container_width=True)
        with col2:
            st.plotly_chart(fig_target_corr, use_container_width=True)
        
        st.markdown(f"""
        **解釈のヒント:**
        - {corr_method}相関は単調な関係の強さを示します（-1〜1）
        - 非線形な相互作用（例: C×Cr）は線形相関では捉えられません
        - ARDはこのような非線形な重要度も学習できます
        """)
    
    with st.expander("📈 対散布図プロット", expanded=False):
        st.markdown("選択した次元間の関係を可視化します。点の色は目的関数値を表します。")
        
        preset_options = {
            "有効次元のみ": list(problem.eff_dim[:min(6, len(problem.eff_dim))]),
            "目的関数との相関上位6次元": list(np.argsort(np.abs(target_corr))[-6:][::-1]),
            "分散上位6次元": list(np.argsort(np.var(X_eda, axis=0))[-6:][::-1])
        }
        
        if method == "Sparse BO (ARD)" and gp_model is not None:
            length_scales = get_length_scales(gp_model, problem.n_dim)
            ard_importance = np.divide(1.0, length_scales, out=np.zeros_like(length_scales), where=length_scales > 0)
            preset_options["ARD重要度上位6次元"] = list(np.argsort(ard_importance)[-6:][::-1])
        
        preset = st.selectbox("プリセット選択", list(preset_options.keys()))
        selected_dims = st.multiselect(
            "次元を選択（2〜8次元）",
            options=list(range(problem.n_dim)),
            default=preset_options[preset],
            format_func=lambda x: f"x{x}" + (" (有効)" if x in problem.eff_dim else "")
        )
        
        if st.button("対散布図を生成", key="generate_pairplot"):
            if len(selected_dims) >= 2 and len(selected_dims) <= 8:
                with st.spinner("対散布図を生成中..."):
                    fig_scatter = plot_scatter_matrix(X_eda, y_eda, problem, selected_dims)
                    if fig_scatter:
                        st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("2〜8次元を選択してください")
    
    with st.expander("🎯 PCA分析とBiplot", expanded=False):
        st.markdown("""
        主成分分析（PCA）により、データの分散を説明する主要な方向を可視化します。
        
        **注意**: PCAは標準化された特徴量で実行されます。
        """)
        
        show_all_labels = st.checkbox("すべての次元ラベルを表示", value=False)
        
        with st.spinner("PCA分析中..."):
            fig_var, fig_biplot, explained_var = plot_pca_biplot(X_eda, y_eda, problem, show_all_labels)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.plotly_chart(fig_var, use_container_width=True)
                st.markdown(f"""
                **累積寄与率:**
                - PC1-2: {(explained_var[0]+explained_var[1])*100:.1f}%
                - PC1-3: {(explained_var[0]+explained_var[1]+explained_var[2])*100:.1f}%
                """)
            with col2:
                st.plotly_chart(fig_biplot, use_container_width=True)
        
        st.markdown("""
        **Biplotの見方:**
        - **点（サンプル）**: 各データポイントの主成分空間での位置。色は目的関数値
        - **矢印（ローディング）**: 各次元が主成分に与える影響の方向と大きさ
        - **赤い矢印**: 有効次元（正解ラベル）
        - **青い矢印**: その他の次元
        
        **解釈のヒント:**
        - 矢印が長い次元ほど、その主成分への寄与が大きい
        - 矢印の方向が近い次元同士は相関が高い
        - PCAは分散を捉えるため、必ずしも目的関数への重要度とは一致しません
        - ARDは目的関数への影響を直接学習するため、より適切な重要度を与えます
        """)

    st.markdown("---")
    st.subheader("1️⃣ Iteration Log")
    st.markdown("""
    ### イテレーションログ
    各ステップで何が起きているかを詳細に確認できます。
    """)

    if iteration_log:
        log_df = pd.DataFrame(iteration_log)

        # Display key columns
        display_cols = ['iteration', 'y_new', 'best_so_far']
        if method != "Random Search" and 'mu' in log_df.columns:
            display_cols.extend(['mu', 'sigma', 'ei'])

        st.dataframe(log_df[display_cols].style.format({
            'y_new': '{:.4f}',
            'best_so_far': '{:.4f}',
            'mu': '{:.4f}',
            'sigma': '{:.4f}',
            'ei': '{:.6f}'
        }), use_container_width=True)

        st.markdown("#### 詳細インスペクター")
        selected_iter = st.slider(
            "イテレーションを選択",
            0,
            len(iteration_log) - 1,
            len(iteration_log) - 1
        )

        selected_log = iteration_log[selected_iter]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Iteration", selected_log['iteration'])
            st.metric("Observed y", f"{selected_log['y_new']:.4f}")
        with col2:
            st.metric("Best so far", f"{selected_log['best_so_far']:.4f}")
            if selected_log.get('mu') is not None:
                st.metric("GP Mean (μ)", f"{selected_log['mu']:.4f}")
        with col3:
            if selected_log.get('sigma') is not None:
                st.metric("GP Std (σ)", f"{selected_log['sigma']:.4f}")
            if selected_log.get('ei') is not None:
                st.metric("EI", f"{selected_log['ei']:.6f}")

        if selected_log.get('x_new'):
            st.markdown("**選択された点 (x):**")
            x_display = {f"x{i}": f"{val:.4f}" for i, val in enumerate(selected_log['x_new'][:5])}
            if len(selected_log['x_new']) > 5:
                x_display['...'] = f"(+{len(selected_log['x_new'])-5} more)"
            st.json(x_display)

        if selected_log.get('length_scales'):
            st.markdown("**Length Scales (ℓ):**")
            ls_data = []
            for i, ls in enumerate(selected_log['length_scales']):
                ls_data.append({
                    'Dimension': f"x{i}",
                    'Length Scale': f"{ls:.4f}",
                    'Relevance (1/ℓ)': f"{1/ls:.4f}" if ls > 0 else "∞",
                    'Type': "Effective" if i in problem.eff_dim else "Ineffective"
                })
            st.dataframe(pd.DataFrame(ls_data), use_container_width=True)

        # Download button
        csv = log_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Log as CSV",
            data=csv,
            file_name=f"optimization_log_{method.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("イテレーションログがありません。最適化を実行してください。")

    st.markdown("---")
    st.subheader("2️⃣ ARD Evolution")
    st.markdown("""
    各次元の重要度が時間とともにどう変化するかを可視化します。
    青い枠は**有効次元（正解ラベル）**を示します。
    """)
    
    with st.expander("📐 ARDの数式（クリックして展開）", expanded=False):
        st.markdown("#### ARD (Automatic Relevance Determination) の数学的定義")
        st.markdown("ARDは、ガウス過程（GP）のカーネル関数に**次元ごとの長さスケール** ℓᵢ を導入することで、各次元の重要度を自動的に学習する手法です。")
        
        st.markdown("**1. ARD距離**")
        st.latex(r"r(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{i=1}^{D} \frac{(x_i - x_i')^2}{\ell_i^2}}")
        st.markdown("- ℓᵢ: 次元 i の長さスケール（相関距離）")
        st.markdown("- ℓᵢ が大きい → 次元 i の変化が距離に寄与しにくい → 重要度が低い")
        st.markdown("- ℓᵢ が小さい → 次元 i の変化が距離に大きく寄与 → 重要度が高い")
        
        st.markdown("**2. Matérn カーネル (ν = 2.5)**")
        st.latex(r"k(\mathbf{x}, \mathbf{x}') = \sigma^2 \left(1 + \sqrt{5}r + \frac{5r^2}{3}\right) \exp(-\sqrt{5}r)")
        st.markdown("- σ²: カーネル振幅（出力のスケール）")
        st.markdown("- r: ARD距離（上記で定義）")
        st.markdown("- ν = 2.5: 滑らかさパラメータ（2回微分可能な関数を仮定）")
        
        st.markdown("**3. ガウス過程の予測**")
        st.latex(r"\mu(\mathbf{x}) = \mathbf{k}(\mathbf{x}, X) [K + \sigma_n^2 I]^{-1} \mathbf{y}")
        st.latex(r"\sigma^2(\mathbf{x}) = k(\mathbf{x}, \mathbf{x}) - \mathbf{k}(\mathbf{x}, X) [K + \sigma_n^2 I]^{-1} \mathbf{k}(X, \mathbf{x})")
        st.markdown("- μ(x): 予測平均")
        st.markdown("- σ²(x): 予測分散（不確実性）")
        st.markdown("- K: 観測点間のカーネル行列（ℓᵢ に依存）")
        
        st.markdown("**4. Expected Improvement (獲得関数)**")
        st.latex(r"Z = \frac{\mu(\mathbf{x}) - y_{\text{best}} - \xi}{\sigma(\mathbf{x})}")
        st.latex(r"EI(\mathbf{x}) = (\mu(\mathbf{x}) - y_{\text{best}} - \xi) \Phi(Z) + \sigma(\mathbf{x}) \phi(Z)")
        st.markdown("- Φ: 標準正規分布の累積分布関数")
        st.markdown("- φ: 標準正規分布の確率密度関数")
        st.markdown("- ξ: 探索パラメータ（デフォルト: 0.01）")
        
        st.markdown("---")
        st.markdown("#### 長さスケール ℓ の直感的な意味")
        st.markdown("**ℓᵢ は次元 i における「相関距離」です：**")
        st.markdown("- **ℓᵢ → ∞** のとき: (xᵢ - x'ᵢ)/ℓᵢ → 0 となり、次元 i の違いが距離 r に寄与しなくなる → カーネルが xᵢ に無感応になる → **次元 i は無視される（スパース性）**")
        st.markdown("- **ℓᵢ → 0** のとき: わずかな xᵢ の変化でも r が大きくなる → カーネルが xᵢ に敏感に反応 → **次元 i は高い重要度を持つ**")
        st.markdown("**重要度の指標:** 1/ℓᵢ（逆長さスケール）または 1/ℓᵢ²（距離への寄与度）")
        st.markdown("このアプリでは、正規化した 1/ℓᵢ を「関連度 (Relevance)」として表示しています。")
    
    # Length scale transitions visualization
    if method != "Random Search" and iteration_log:
        with st.expander("📈 長さスケールの推移（時系列プロット）", expanded=True):
            st.markdown("各次元の長さスケール ℓ がイテレーションとともにどう変化するかを表示します。")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                mode = st.selectbox("表示モード", ["ℓ (長さスケール)", "1/ℓ (関連度)", "1/ℓ² (距離寄与度)"],
                                   key="ls_mode")
                mode_map = {"ℓ (長さスケール)": "ell", "1/ℓ (関連度)": "inv_ell", "1/ℓ² (距離寄与度)": "inv_ell_sq"}
                mode_val = mode_map[mode]
            with col2:
                normalize = st.checkbox("ドメイン幅で正規化", value=True, key="ls_normalize")
            with col3:
                log_y = st.checkbox("対数スケール", value=False, key="ls_log")
            
            top_k = st.slider("表示する次元数（関連度上位）", 3, min(10, problem.n_dim), 
                             min(6, problem.n_dim), key="ls_topk")
            
            fig_ls_ts = plot_length_scale_timeseries(iteration_log, problem, 
                                                     mode=mode_val, normalize=normalize, 
                                                     log_y=log_y, top_k=top_k)
            if fig_ls_ts:
                st.plotly_chart(fig_ls_ts, use_container_width=True)
                
                st.info("""
                **解釈:**
                - **青い線**: 有効次元 → ARDが学習を進めるにつれて、ℓが小さく（または1/ℓが大きく）なることを期待
                - **灰色の線**: 無効次元 → ℓが大きく（または1/ℓが小さく）なり、無視されることを期待
                - 正規化オプションをONにすると、異なるドメイン幅を持つ次元間で比較しやすくなります
                """)
            else:
                st.warning("長さスケール情報が不足しています。")
    
    st.markdown("---")
    st.markdown("""
    #### 「有効次元（正解ラベル）」とは？
    
    青枠で示している次元は、**ベンチマーク関数の定義で実際に使われている変数**です。
    
    - **定義場所**: `sparse_bayesian_optimization_app.py` の20行目（Rosenbrock）、24行目（Hartmann）
    - **使用箇所**: 31行目で `X_eff = X[:, self.eff_dim]` として有効次元だけを取り出して計算
    - **重要**: これは「モデルが知っている情報」ではなく、「ベンチマーク設計上の正解」です
    
    ARDは**データから学習**して、どの次元が重要かを推定します。学習が進むと、
    ARDの重要度（1/length_scale）が青枠の次元に集中していく様子が観察できます。
    """)

    # Add 1D sweep visualization
    if X_train is not None and len(y_train) > 0:
        with st.expander("🔍 1次元スイープで「有効次元」を確認", expanded=False):
            st.markdown("""
            **なぜこれらの次元が「有効」なのか？**
            
            下のグラフは、現在の最良点を基準に、各次元を個別に動かしたときの出力変化を示します。
            - **青い線**: 有効次元 → 変数を動かすと出力が変化
            - **灰色の線**: 無効次元 → 変数を動かしても出力は変化しない（フラット）
            
            これが「有効次元」の意味です。関数定義上、これらの次元だけが出力に影響します。
            """)
            
            fig_sweep = plot_1d_sweep(problem, X_train, y_train)
            if fig_sweep:
                st.plotly_chart(fig_sweep, use_container_width=True)

    st.markdown("---")

    if method != "Random Search" and iteration_log:
        fig_ard = plot_ard_evolution(iteration_log, problem)
        if fig_ard:
            st.plotly_chart(fig_ard, use_container_width=True)

            st.info("""
            **ARD進化ヒートマップの解釈:**
            - **濃い赤色（値1.0に近い）**: 高い重要度 → ARDがその次元を重要と判断
            - **薄い黄色（値0.0に近い）**: 低い重要度 → ARDがその次元を無視
            - **青枠**: 有効次元（正解ラベル）
            - **カラーバー**: 右端（1.0）= 高い重要度、左端（0.0）= 低い重要度
            - **期待される結果**: ARDが学習を進めるにつれて、青枠の次元が濃い赤色になる
            
            もし青枠以外の次元が濃い赤色になっている場合、それはARDの学習が不十分か、
            データにノイズが多いことを示しています。
            """)
        else:
            st.warning("ARD情報が不足しています。")
    else:
        st.warning("Random Searchでは変数重要度の推定は行われません。")

    st.markdown("---")
    st.subheader("3️⃣ Mental Model: Lasso vs GP-ARD")
    
    with st.expander("📚 なぜLassoは非線形関数で失敗するのか？", expanded=True):
        st.markdown("""
        ### 数学的背景
        
        #### A. Lasso (Least Absolute Shrinkage and Selection Operator)
        Lassoは以下の式を最小化します：
        """)
        st.latex(r"\text{Minimize} \sum (y - \mathbf{w}^T \mathbf{x})^2 + \lambda ||\mathbf{w}||_1")
        st.markdown("""
        **特徴:**
        - 重みベクトル $\\mathbf{w}$ の L1 ノルム（絶対値和）にペナルティをかけます
        - これにより、不要な特徴量の重みが完全に **0** になります（スパース性）
        
        **弱点:**
        - モデルが **線形** であること
        - $y = x_1^2$ や $y = x_1 x_2$ のような関係を $w_1 x_1$ で表現しようとすると失敗します
        
        #### B. GP-ARD (Gaussian Process with ARD Kernel)
        GPはカーネル関数 $k(\\mathbf{x}, \\mathbf{x}')$ を用いて類似度を測ります。ARDカーネル（Matérn や RBF）は以下のように定義されます：
        """)
        st.latex(r"k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp \left( - \sum_{d=1}^{D} \frac{(x_d - x_d')^2}{2 \ell_d^2} \right)")
        st.markdown("""
        **$\\ell_d$ (Length Scale):** 次元 $d$ 方向の「特徴的な長さ」です。
        
        **スパース性のメカニズム:**
        - もし次元 $d$ が重要でなければ、その次元の値 $x_d$ がどう変わろうと $y$ は変わりません
        - GP はこれを **$\\ell_d \\to \\infty$** （非常に大きな値）にすることで表現します
        - 分母が無限大になれば、$\\frac{(x_d - x_d')^2}{2 \\ell_d^2} \\to 0$ となり、その次元の距離は無視されます
        
        **重要度:** 逆数 $\\frac{1}{\\ell_d}$ が「重要度（Relevance）」として解釈できます。
        
        ---
        
        ### Rosenbrock関数での失敗例
        
        Rosenbrock関数: $f(x_0, x_1) = (1-x_0)^2 + 100(x_1 - x_0^2)^2$
        
        - **原点付近で対称性**があるため、$x_0$ と $y$ の線形相関が出にくい
        - Lassoは $x_0, x_1$ の重要度を**ゼロと誤判定**することがある
        - これはバグではなく、**線形モデルの限界**を示す良い例です
        """)
    
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

    st.markdown("---")
    st.subheader("4️⃣ Optimization & Exploration")
    st.markdown("""
    ### 最適化の収束過程と探索空間
    """)

    st.markdown("""
    横軸にイテレーション、縦軸に目的関数値を表示します。
    - 水色の点: 各イテレーションで観測された値
    - 青い線: 現在までの最良値
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


    st.markdown("---")
    st.markdown("""
    ### 探索空間の可視化 (Rosenbrock関数のみ)
    
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
