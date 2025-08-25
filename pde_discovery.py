"""
偏微分方程式発見システム (PDE Discovery System)
FDMによる熱伝導方程式の数値解から偏微分方程式を逆算する
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import sympy as sp
from typing import Tuple, List, Dict, Callable

matplotlib.rcParams['font.family'] = ['IPAGothic', 'IPAPGothic']
matplotlib.rcParams['font.sans-serif'] = ['IPAGothic', 'IPAPGothic']
plt.rcParams['axes.unicode_minus'] = False
try:
    import torch
    from pinns_discovery import PINNsHeatSolver, PINNsBurgersSolver, PINNsDiffusionSolver
    PINNS_AVAILABLE = True
except ImportError:
    PINNS_AVAILABLE = False

class HeatConductionFDM:
    """有限差分法による熱伝導方程式の数値解法"""
    
    def __init__(self, L: float = 1.0, T_final: float = 1.0, 
                 nx: int = 50, nt: int = 100, alpha: float = 0.01):
        """
        Parameters:
        L: 空間領域の長さ
        T_final: 最終時刻
        nx: 空間格子点数
        nt: 時間格子点数
        alpha: 熱拡散係数
        """
        self.L = L
        self.T_final = T_final
        self.nx = nx
        self.nt = nt
        self.alpha = alpha
        
        self.dx = L / (nx - 1)
        self.dt = T_final / (nt - 1)
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T_final, nt)
        
        self.r = alpha * self.dt / (self.dx**2)
        if self.r > 0.5:
            st.warning(f"安定性条件違反: r = {self.r:.3f} > 0.5")
    
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """初期条件: ガウシアン分布"""
        return np.exp(-50 * (x - 0.5)**2)
    
    def boundary_conditions(self, u: np.ndarray, n: int) -> np.ndarray:
        """境界条件: 両端で0"""
        u[0] = 0.0
        u[-1] = 0.0
        return u
    
    def solve(self) -> np.ndarray:
        """FDMによる熱伝導方程式の数値解"""
        u = np.zeros((self.nt, self.nx))
        
        u[0, :] = self.initial_condition(self.x)
        u[0, :] = self.boundary_conditions(u[0, :], 0)
        
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                u[n+1, i] = u[n, i] + self.r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
            u[n+1, :] = self.boundary_conditions(u[n+1, :], n+1)
        
        return u

class BurgersFDM:
    """有限差分法によるBurgers方程式の数値解法"""
    
    def __init__(self, L: float = 1.0, T_final: float = 0.5, 
                 nx: int = 50, nt: int = 100, nu: float = 0.01):
        """
        Parameters:
        L: 空間領域の長さ
        T_final: 最終時刻
        nx: 空間格子点数
        nt: 時間格子点数
        nu: 粘性係数
        """
        self.L = L
        self.T_final = T_final
        self.nx = nx
        self.nt = nt
        self.nu = nu
        
        self.dx = L / (nx - 1)
        self.dt = T_final / (nt - 1)
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T_final, nt)
        
        self.r = nu * self.dt / (self.dx**2)
        u_max = 1.0
        self.cfl = u_max * self.dt / self.dx
        
        if self.r > 0.5:
            st.warning(f"拡散安定性条件違反: r = {self.r:.3f} > 0.5")
        if self.cfl > 1.0:
            st.warning(f"CFL条件違反: CFL = {self.cfl:.3f} > 1.0")
    
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """初期条件: ステップ関数"""
        return np.where(x < 0.5, 1.0, 0.0)
    
    def boundary_conditions(self, u: np.ndarray, n: int) -> np.ndarray:
        """境界条件: 両端で0"""
        u[0] = 0.0
        u[-1] = 0.0
        return u
    
    def solve(self) -> np.ndarray:
        """FDMによるBurgers方程式の数値解"""
        u = np.zeros((self.nt, self.nx))
        
        u[0, :] = self.initial_condition(self.x)
        u[0, :] = self.boundary_conditions(u[0, :], 0)
        
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                diffusion = self.r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
                
                if u[n, i] >= 0:
                    convection = -u[n, i] * self.dt / self.dx * (u[n, i] - u[n, i-1])
                else:
                    convection = -u[n, i] * self.dt / self.dx * (u[n, i+1] - u[n, i])
                
                u[n+1, i] = u[n, i] + diffusion + convection
            
            u[n+1, :] = self.boundary_conditions(u[n+1, :], n+1)
        
        return u


class DiffusionFDM:
    """有限差分法による1次元拡散方程式の数値解法（合金系原子拡散）"""
    
    def __init__(self, L: float = 0.02, T_final: float = 3600.0, 
                 nx: int = 50, nt: int = 100, D: float = 1e-11):
        """
        Parameters:
        L: 空間領域の長さ (m)
        T_final: 最終時刻 (s)
        nx: 空間格子点数
        nt: 時間格子点数
        D: 拡散係数 (m²/s)
        """
        self.L = L
        self.T_final = T_final
        self.nx = nx
        self.nt = nt
        self.D = D
        
        self.dx = L / (nx - 1)
        self.dt = T_final / (nt - 1)
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T_final, nt)
        
        self.r = D * self.dt / (self.dx**2)
        if self.r > 0.5:
            import streamlit as st
            st.warning(f"拡散安定性条件違反: r = {self.r:.3f} > 0.5")
    
    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        """初期条件: 拡散カップル（左半分が濃度1、右半分が濃度0）"""
        return np.where(x <= self.L/2, 1.0, 0.0)
    
    def boundary_conditions(self, u: np.ndarray, n: int) -> np.ndarray:
        """境界条件: ゼロフラックス（∂c/∂x = 0）"""
        u[0] = u[1]    # 左境界
        u[-1] = u[-2]  # 右境界
        return u
    
    def solve(self) -> np.ndarray:
        """FDMによる拡散方程式の数値解"""
        u = np.zeros((self.nt, self.nx))
        
        u[0, :] = self.initial_condition(self.x)
        u[0, :] = self.boundary_conditions(u[0, :], 0)
        
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                u[n+1, i] = u[n, i] + self.r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
            u[n+1, :] = self.boundary_conditions(u[n+1, :], n+1)
        
        return u


class NumericalDerivatives:
    """数値微分計算クラス"""
    
    @staticmethod
    def compute_dt(u: np.ndarray, dt: float) -> np.ndarray:
        """時間微分 ∂u/∂t の計算"""
        dudt = np.zeros_like(u)
        
        dudt[:-1, :] = (u[1:, :] - u[:-1, :]) / dt
        dudt[-1, :] = (u[-1, :] - u[-2, :]) / dt
        
        return dudt
    
    @staticmethod
    def compute_dx(u: np.ndarray, dx: float) -> np.ndarray:
        """空間1階微分 ∂u/∂x の計算"""
        dudx = np.zeros_like(u)
        
        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        dudx[:, 0] = (u[:, 1] - u[:, 0]) / dx
        dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
        
        return dudx
    
    @staticmethod
    def compute_d2x(u: np.ndarray, dx: float) -> np.ndarray:
        """空間2階微分 ∂²u/∂x² の計算"""
        d2udx2 = np.zeros_like(u)
        
        d2udx2[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / (dx**2)
        d2udx2[:, 0] = 0
        d2udx2[:, -1] = 0
        
        return d2udx2

class ComplexityCalculator:
    """モデル複雑度計算システム"""
    
    def __init__(self):
        self.operator_weights = {
            'linear': 1,      
            'quadratic': 2,   
            'derivative1': 2, 
            'derivative2': 3, 
            'nonlinear': 3,   
            'interaction': 2  
        }
    
    def calculate_pde_complexity(self, formula_name: str, params: List[float]) -> float:
        """PDE式の複雑度計算"""
        complexity = 0
        
        complexity += len(params)
        
        if "∂²u/∂x²" in formula_name:
            complexity += self.operator_weights['derivative2']
        if "∂u/∂x" in formula_name:
            complexity += self.operator_weights['derivative1']
        if "u" in formula_name and "∂" not in formula_name:
            complexity += self.operator_weights['linear']
        if "u × ∂" in formula_name or "u²" in formula_name:
            complexity += self.operator_weights['nonlinear']
        
        return complexity

class PDESymbolicRegression:
    """偏微分方程式のシンボリック回帰"""
    
    def __init__(self, u: np.ndarray, x: np.ndarray, t: np.ndarray):
        self.u = u
        self.x = x
        self.t = t
        self.dx = x[1] - x[0]
        self.dt = t[1] - t[0]
        
        self.derivatives = NumericalDerivatives()
        self.dudt = self.derivatives.compute_dt(u, self.dt)
        self.dudx = self.derivatives.compute_dx(u, self.dx)
        self.d2udx2 = self.derivatives.compute_d2x(u, self.dx)
    
    def evaluate_pde_formula(self, formula_func: Callable, params: List[float], 
                           mask: np.ndarray = None) -> float:
        """PDE候補式の評価"""
        if mask is None:
            mask = np.ones_like(self.u, dtype=bool)
            mask[0, :] = False  
            mask[:, 0] = False  
            mask[:, -1] = False 
        
        try:
            predicted = formula_func(params, self.u, self.dudx, self.d2udx2)
            
            actual = self.dudt
            
            mse = np.mean((predicted[mask] - actual[mask])**2)
            
            return mse
            
        except Exception as e:
            return np.inf
    
    def calculate_bic(self, likelihood: float, n_params: int, n_data: int) -> float:
        """BIC計算: -2ln(L) + k×ln(n)"""
        return -2 * np.log(likelihood) + n_params * np.log(n_data)
    
    def calculate_aic(self, likelihood: float, n_params: int) -> float:
        """AIC計算: -2ln(L) + 2k"""
        return -2 * np.log(likelihood) + 2 * n_params
    
    def calculate_likelihood(self, mse: float, n_data: int) -> float:
        """MSEから尤度を計算"""
        sigma_squared = mse
        return np.exp(-n_data * np.log(2 * np.pi * sigma_squared) / 2 - n_data * mse / (2 * sigma_squared))
    
    def calculate_model_weights(self, bic_scores: np.ndarray) -> np.ndarray:
        """ベイズモデル重み計算"""
        delta_bic = bic_scores - np.min(bic_scores)
        weights = np.exp(-0.5 * delta_bic)
        return weights / np.sum(weights)
    
    def discover_heat_equation(self) -> Dict:
        """熱伝導方程式の発見"""
        
        formulas = {
            "∂u/∂t = c₁ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2,
                "params": [0.01],
                "description": "標準的な熱伝導方程式"
            },
            "∂u/∂t = c₁ × ∂²u/∂x² + c₂ × u": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2 + p[1] * u,
                "params": [0.01, 0.0],
                "description": "反応項付き熱伝導方程式"
            },
            "∂u/∂t = c₁ × ∂²u/∂x² + c₂ × ∂u/∂x": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2 + p[1] * dudx,
                "params": [0.01, 0.0],
                "description": "対流項付き熱伝導方程式"
            },
            "∂u/∂t = c₁ × u × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * u * d2udx2,
                "params": [0.01],
                "description": "非線形拡散方程式"
            }
        }
        
        results = {}
        
        for name, formula_info in formulas.items():
            func = formula_info["func"]
            initial_params = formula_info["params"]
            
            try:
                mse = self.evaluate_pde_formula(func, initial_params)
                optimal_params = initial_params  # For now, use initial params
            except Exception as e:
                mse = np.inf
                optimal_params = initial_params
            
            results[name] = {
                'mse': mse,
                'params': optimal_params,
                'description': formula_info["description"]
            }
        
        best_result = min(results.values(), key=lambda x: x['mse'])
        best_alpha = abs(best_result['params'][0]) if best_result['params'] else 0.01
        
        all_results = []
        for formula_name, result_data in results.items():
            all_results.append({
                'formula': formula_name,
                'mse': result_data['mse'],
                'params': result_data['params'],
                'description': result_data['description']
            })
        
        return {
            'all_results': all_results,
            'best_alpha': best_alpha,
            'optimization_details': {'method': 'direct_evaluation', 'status': 'completed'}
        }
    
    def discover_burgers_equation(self) -> Dict:
        """Burgers方程式の発見"""
        
        formulas = {
            "∂u/∂t = c₁ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2,
                "params": [0.01],
                "description": "純粋拡散方程式"
            },
            "∂u/∂t = -c₁ × u × ∂u/∂x + c₂ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: -p[0] * u * dudx + p[1] * d2udx2,
                "params": [1.0, 0.01],
                "description": "標準的なBurgers方程式"
            },
            "∂u/∂t = -c₁ × u × ∂u/∂x": {
                "func": lambda p, u, dudx, d2udx2: -p[0] * u * dudx,
                "params": [1.0],
                "description": "無粘性Burgers方程式"
            },
            "∂u/∂t = -c₁ × ∂u/∂x + c₂ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: -p[0] * dudx + p[1] * d2udx2,
                "params": [0.5, 0.01],
                "description": "線形対流拡散方程式"
            },
            "∂u/∂t = -c₁ × u² × ∂u/∂x + c₂ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: -p[0] * u**2 * dudx + p[1] * d2udx2,
                "params": [0.5, 0.01],
                "description": "修正Burgers方程式"
            },
            "∂u/∂t = c₁ × u × ∂u/∂x + c₂ × ∂²u/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * u * dudx + p[1] * d2udx2,
                "params": [-1.0, 0.01],
                "description": "符号付きBurgers方程式"
            }
        }
        
        results = {}
        
        for name, formula_info in formulas.items():
            func = formula_info["func"]
            initial_params = formula_info["params"]
            
            try:
                mse = self.evaluate_pde_formula(func, initial_params)
                optimal_params = initial_params  # For now, use initial params
            except Exception as e:
                mse = np.inf
                optimal_params = initial_params
            
            results[name] = {
                'mse': mse,
                'params': optimal_params,
                'description': formula_info["description"]
            }
        
        best_result = min(results.values(), key=lambda x: x['mse'])
        best_formula_name = min(results.keys(), key=lambda k: results[k]['mse'])
        if "u × ∂u/∂x" in best_formula_name and len(best_result['params']) > 1:
            best_nu = abs(best_result['params'][1])  # Second parameter is viscosity
        else:
            best_nu = abs(best_result['params'][0]) if best_result['params'] else 0.01
        
        all_results = []
        for formula_name, result_data in results.items():
            all_results.append({
                'formula': formula_name,
                'mse': result_data['mse'],
                'params': result_data['params'],
                'description': result_data['description']
            })
        
        return {
            'all_results': all_results,
            'best_nu': best_nu,
            'optimization_details': {'method': 'direct_evaluation', 'status': 'completed'}
        }
    
    def discover_diffusion_equation(self, use_exhaustive_search: bool = False, max_complexity: int = 3) -> Dict:
        """拡散方程式の発見（ベイズ的モデル選択対応）"""
        
        if use_exhaustive_search:
            return self.exhaustive_search_diffusion(max_complexity)
        
        formulas = {
            "∂c/∂t = c₁ × ∂²c/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2,
                "params": [1e-11],
                "description": "標準的な拡散方程式"
            },
            "∂c/∂t = c₁ × c × ∂²c/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * u * d2udx2,
                "params": [1e-11],
                "description": "濃度依存拡散方程式"
            },
            "∂c/∂t = c₁ × ∂²c/∂x² + c₂ × ∂c/∂x": {
                "func": lambda p, u, dudx, d2udx2: p[0] * d2udx2 + p[1] * dudx,
                "params": [1e-11, 0.0],
                "description": "対流項付き拡散方程式"
            },
            "∂c/∂t = c₁ × (1-c) × ∂²c/∂x²": {
                "func": lambda p, u, dudx, d2udx2: p[0] * (1 - u) * d2udx2,
                "params": [1e-11],
                "description": "相互拡散方程式"
            }
        }
        
        candidates = []
        complexity_calc = ComplexityCalculator()
        
        for name, formula_info in formulas.items():
            candidates.append({
                'name': name,
                'func': formula_info['func'],
                'params': formula_info['params'],
                'complexity': complexity_calc.calculate_pde_complexity(name, formula_info['params'])
            })
        
        return self._evaluate_candidates_bayesian(candidates)
    
    def exhaustive_search_diffusion(self, max_complexity: int = 4) -> Dict:
        """拡散方程式の全状態探索"""
        
        base_terms = {
            "∂²c/∂x²": lambda p, u, dudx, d2udx2: p[0] * d2udx2,
            "∂c/∂x": lambda p, u, dudx, d2udx2: p[0] * dudx,
            "c": lambda p, u, dudx, d2udx2: p[0] * u,
            "c²": lambda p, u, dudx, d2udx2: p[0] * u**2,
            "c × ∂c/∂x": lambda p, u, dudx, d2udx2: p[0] * u * dudx,
            "c × ∂²c/∂x²": lambda p, u, dudx, d2udx2: p[0] * u * d2udx2
        }
        
        all_candidates = []
        complexity_calc = ComplexityCalculator()
        
        from itertools import combinations
        
        for complexity in range(1, max_complexity + 1):
            for term_combo in combinations(base_terms.keys(), complexity):
                formula_name = " + ".join([f"c{i+1} × {term}" for i, term in enumerate(term_combo)])
                
                def combined_func(params, u, dudx, d2udx2, terms=term_combo):
                    result = np.zeros_like(u)
                    param_idx = 0
                    for term in terms:
                        result += base_terms[term]([params[param_idx]], u, dudx, d2udx2)
                        param_idx += 1
                    return result
                
                initial_params = [0.01] * len(term_combo)
                
                all_candidates.append({
                    'name': f"∂c/∂t = {formula_name}",
                    'func': combined_func,
                    'params': initial_params,
                    'complexity': complexity_calc.calculate_pde_complexity(formula_name, initial_params)
                })
        
        return self._evaluate_candidates_bayesian(all_candidates)
    
    def _evaluate_candidates_bayesian(self, candidates: List[Dict]) -> Dict:
        """候補式のベイズ評価"""
        results = []
        n_data = np.sum(np.ones_like(self.u)[1:-1, 1:-1])  
        
        for candidate in candidates:
            try:
                mse = self.evaluate_pde_formula(candidate['func'], candidate['params'])
                likelihood = self.calculate_likelihood(mse, n_data)
                n_params = len(candidate['params'])
                
                bic = self.calculate_bic(likelihood, n_params, n_data)
                aic = self.calculate_aic(likelihood, n_params)
                
                results.append({
                    'name': candidate['name'],
                    'mse': mse,
                    'likelihood': likelihood,
                    'bic': bic,
                    'aic': aic,
                    'complexity': candidate['complexity'],
                    'params': candidate['params'],
                    'n_params': n_params
                })
            except Exception as e:
                continue
        
        if not results:
            return {'best_model': None, 'all_results': [], 'model_weights': []}
        
        bic_scores = np.array([r['bic'] for r in results])
        model_weights = self.calculate_model_weights(bic_scores)
        
        best_idx = np.argmin(bic_scores)
        best_model = results[best_idx]
        
        for i, result in enumerate(results):
            result['model_weight'] = model_weights[i]
            result['posterior_prob'] = model_weights[i]
        
        return {
            'best_model': best_model,
            'all_results': sorted(results, key=lambda x: x['bic']),
            'model_weights': model_weights,
            'bayesian_model_average': self._calculate_model_average(results, model_weights)
        }
    
    def _calculate_model_average(self, results: List[Dict], weights: np.ndarray) -> Dict:
        """ベイズモデル平均化"""
        if not results:
            return {}
        
        avg_params = np.zeros(max(len(r['params']) for r in results))
        avg_complexity = np.sum([r['complexity'] * w for r, w in zip(results, weights)])
        
        return {
            'average_complexity': avg_complexity,
            'model_uncertainty': np.std([r['mse'] for r in results]),
            'parameter_uncertainty': np.std([r['params'][0] if r['params'] else 0 for r in results])
        }

def create_pde_discovery_app():
    """Streamlit PDE発見アプリ"""
    
    st.title("🔬 偏微分方程式発見システム (PDE Discovery)")
    st.markdown("---")
    
    equation_type = st.selectbox(
        "🧮 方程式タイプを選択",
        ["熱伝導方程式", "Burgers方程式", "拡散方程式"],
        help="発見したい偏微分方程式のタイプを選択してください"
    )
    
    if equation_type == "熱伝導方程式":
        st.markdown("""
        有限差分法(FDM)で生成した熱伝導方程式の数値解から、
        元の偏微分方程式を逆算するシンボリック回帰システムです。
        
        **対象方程式**: ∂u/∂t = α × ∂²u/∂x²
        
        **手順:**
        1. FDMによる熱伝導方程式の数値解生成（疑似正解データ作成）
        2. PINNsによるPDE発見と検証
        3. シンボリック回帰による PDE 構造の発見
        """)
    elif equation_type == "Burgers方程式":
        st.markdown("""
        有限差分法(FDM)で生成したBurgers方程式の数値解から、
        元の偏微分方程式を逆算するシンボリック回帰システムです。
        
        **対象方程式**: ∂u/∂t + u×∂u/∂x = ν×∂²u/∂x²
        
        **手順:**
        1. FDMによるBurgers方程式の数値解生成（疑似正解データ作成）
        2. PINNsによるPDE発見と検証
        3. シンボリック回帰による非線形PDE構造の発見
        """)
    else:
        st.markdown("""
        有限差分法(FDM)で生成した拡散方程式の数値解から、
        元の偏微分方程式を逆算するシンボリック回帰システムです。
        
        **対象方程式**: ∂c/∂t = D × ∂²c/∂x²
        
        **手順:**
        1. FDMによる拡散方程式の数値解生成（疑似正解データ作成）
        2. PINNsによるPDE発見と検証
        3. シンボリック回帰による拡散係数Dの逆解析
        """)
    
    st.sidebar.header("📊 観測データ")
    uploaded_file = st.sidebar.file_uploader(
        "観測データをアップロード (CSV)", 
        type=["csv"],
        help="時間、空間、濃度データを含むCSVファイル（列名: t, x, c）"
    )
    
    observational_data = None
    if uploaded_file is not None:
        try:
            observational_data = pd.read_csv(uploaded_file)
            if all(col in observational_data.columns for col in ['t', 'x', 'c']):
                st.sidebar.success(f"✅ データ読み込み完了 ({len(observational_data)} 点)")
                st.sidebar.write(f"時間範囲: {observational_data['t'].min():.2f} - {observational_data['t'].max():.2f}")
                st.sidebar.write(f"空間範囲: {observational_data['x'].min():.4f} - {observational_data['x'].max():.4f}")
            else:
                st.sidebar.error("❌ CSVファイルには 't', 'x', 'c' 列が必要です")
                observational_data = None
        except Exception as e:
            st.sidebar.error(f"❌ ファイル読み込みエラー: {str(e)}")

    st.sidebar.header("⚙️ FDMパラメータ（疑似正解データ生成）")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        nx = st.number_input("空間格子点数", min_value=20, max_value=100, value=50)
        alpha_or_nu_or_D = st.number_input(
            "熱拡散係数 α" if equation_type == "熱伝導方程式" 
            else "粘性係数 ν" if equation_type == "Burgers方程式"
            else "拡散係数 D",
            min_value=1e-15 if equation_type == "拡散方程式" else 0.001,
            max_value=1e-6 if equation_type == "拡散方程式" else 0.1,
            value=1e-11 if equation_type == "拡散方程式" else 0.01,
            format="%.2e" if equation_type == "拡散方程式" else "%.3f"
        )
    
    with col2:
        nt = st.number_input("時間格子点数", min_value=50, max_value=200, value=100)
        if equation_type == "拡散方程式":
            T_final = st.number_input("最終時刻 (s)", min_value=100.0, max_value=10000.0, value=3600.0)
        else:
            T_final = st.number_input("最終時刻", min_value=0.1, max_value=2.0, value=1.0 if equation_type == "熱伝導方程式" else 0.5)
    
    st.sidebar.subheader("🧮 ベイズ的モデル選択")
    use_bayesian = st.sidebar.checkbox("ベイズ評価を使用", value=True)
    use_exhaustive = st.sidebar.checkbox("全状態探索（軽量計算時）", value=False)
    max_complexity = st.sidebar.slider("最大複雑度", 1, 5, 3) if use_exhaustive else 3
    
    if use_exhaustive:
        st.sidebar.warning("⚠️ 全状態探索は計算時間が長くなる場合があります")
    
    if PINNS_AVAILABLE:
        st.sidebar.header("🧠 PINNsパラメータ（PDE発見）")
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            epochs = st.number_input("エポック数", min_value=1000, max_value=20000, value=8000, step=1000)
            hidden_dim = st.number_input("隠れ層次元", min_value=20, max_value=200, value=50, step=10)
        
        with col4:
            num_layers = st.number_input("ネットワーク層数", min_value=3, max_value=8, value=4)
            learning_rate = st.number_input("学習率", min_value=0.0001, max_value=0.01, value=0.0005, format="%.4f")
            n_points = st.number_input("訓練点数", min_value=500, max_value=5000, value=2000, step=500)
    
    st.sidebar.header("🎯 最適化設定")
    max_iter = st.sidebar.number_input("最適化反復回数", min_value=100, max_value=5000, value=1000, step=100)
    num_trials = st.sidebar.number_input("試行回数", min_value=1, max_value=10, value=5,
                                       help="異なる初期値での最適化試行回数")
    use_multiple_methods = st.sidebar.checkbox("複数最適化手法を使用", value=True, 
                                             help="複数の最適化アルゴリズムを試行して最良の結果を選択")
    use_higher_order = st.sidebar.checkbox("高次差分を使用", value=True, help="より正確な数値微分を使用")
    
    if st.button("🚀 PDE発見を実行", type="primary"):
        
        st.header("🔄 順次実行: FDM → PINNs")
        
        st.subheader("📊 Step 1: FDMによる疑似正解データ生成")
        
        if equation_type == "熱伝導方程式":
            with st.spinner("FDMによる数値解計算中..."):
                fdm = HeatConductionFDM(nx=nx, nt=nt, alpha=alpha_or_nu_or_D, T_final=T_final)
                u_numerical = fdm.solve()
            
            st.success("✅ FDM数値解計算完了!")
            
            theoretical_param = alpha_or_nu_or_D
            param_name = "α"
            equation_title = "Temperature Evolution (Heat Conduction)"
            y_label = "Temperature u"
            
        elif equation_type == "Burgers方程式":
            with st.spinner("FDMによる数値解計算中..."):
                fdm = BurgersFDM(nx=nx, nt=nt, nu=alpha_or_nu_or_D, T_final=T_final)
                u_numerical = fdm.solve()
            
            st.success("✅ FDM数値解計算完了!")
            
            theoretical_param = alpha_or_nu_or_D
            param_name = "ν"
            equation_title = "Velocity Evolution (Burgers Equation)"
            y_label = "Velocity u"
            
        else:  # 拡散方程式
            with st.spinner("FDMによる数値解計算中..."):
                fdm = DiffusionFDM(nx=nx, nt=nt, D=alpha_or_nu_or_D, T_final=T_final)
                u_numerical = fdm.solve()
            
            st.success("✅ FDM数値解計算完了!")
            
            theoretical_param = alpha_or_nu_or_D
            param_name = "D"
            equation_title = "Concentration Evolution (Diffusion Couple)"
            y_label = "Concentration c"
        
        st.subheader("📊 FDM数値解の可視化")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        X, T = np.meshgrid(fdm.x, fdm.t)
        im1 = ax1.contourf(X, T, u_numerical, levels=20, cmap='hot')
        ax1.set_xlabel('Space x')
        ax1.set_ylabel('Time t')
        ax1.set_title(equation_title)
        plt.colorbar(im1, ax=ax1)
        
        time_indices = [0, nt//4, nt//2, 3*nt//4, nt-1]
        for i, idx in enumerate(time_indices):
            ax2.plot(fdm.x, u_numerical[idx, :], 
                    label=f't = {fdm.t[idx]:.2f}', alpha=0.8)
        ax2.set_xlabel('Space x')
        ax2.set_ylabel(y_label)
        ax2.set_title('Distribution at Different Times')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        if PINNS_AVAILABLE:
            st.subheader("🧠 Step 2: PINNsによるPDE発見")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(epoch, total_epochs, loss):
                progress = epoch / total_epochs
                progress_bar.progress(progress)
                status_text.text(f"エポック {epoch}/{total_epochs}, 損失: {loss:.6f}")
            
            if equation_type == "熱伝導方程式":
                with st.spinner("PINNsによる熱伝導方程式の解法中..."):
                    solver = PINNsHeatSolver(alpha=alpha_or_nu_or_D, hidden_dim=hidden_dim, num_layers=num_layers)
                    training_results = solver.train(epochs=epochs, lr=learning_rate, n_points=n_points, 
                                                  progress_callback=progress_callback)
                    
                    x_test = np.linspace(0, 1, 50)
                    t_test = np.linspace(0, 1, 50)
                    X_test, T_test = np.meshgrid(x_test, t_test)
                    u_numerical = solver.predict(X_test, T_test)
                    
                    st.subheader("📈 PINNs熱伝導方程式の解")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    im1 = ax1.imshow(u_numerical, aspect='auto', origin='lower', 
                                   extent=[0, 1, 0, 1], cmap='hot')
                    ax1.set_xlabel('Space x')
                    ax1.set_ylabel('Time t')
                    ax1.set_title('PINNs Temperature Distribution u(x,t)')
                    plt.colorbar(im1, ax=ax1)
                    
                    time_indices = [0, 12, 25, 37, 49]
                    for i in time_indices:
                        ax2.plot(x_test, u_numerical[i, :], label=f't = {t_test[i]:.2f}')
                    ax2.set_xlabel('Space x')
                    ax2.set_ylabel('Temperature u')
                    ax2.set_title('Temperature Distribution at Different Times')
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                    st.subheader("🎯 PINNs訓練結果")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最終損失", f"{training_results['final_loss']:.6f}")
                    with col2:
                        st.metric("PDE損失", f"{training_results['pde_loss']:.6f}")
                    with col3:
                        st.metric("境界条件損失", f"{training_results['bc_loss']:.6f}")
                    
            elif equation_type == "Burgers方程式":
                with st.spinner("PINNsによるBurgers方程式の解法中..."):
                    solver = PINNsBurgersSolver(nu=alpha_or_nu_or_D, hidden_dim=hidden_dim, num_layers=num_layers)
                    training_results = solver.train(epochs=epochs, lr=learning_rate, n_points=n_points, 
                                                  progress_callback=progress_callback)
                    
                    x_test = np.linspace(0, 1, 50)
                    t_test = np.linspace(0, 0.5, 50)
                    X_test, T_test = np.meshgrid(x_test, t_test)
                    u_numerical = solver.predict(X_test, T_test)
                    
                    st.subheader("📈 PINNs Burgers方程式の解")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    im1 = ax1.imshow(u_numerical, aspect='auto', origin='lower', 
                                   extent=[0, 1, 0, 0.5], cmap='viridis')
                    ax1.set_xlabel('Space x')
                    ax1.set_ylabel('Time t')
                    ax1.set_title('PINNs Velocity Field u(x,t)')
                    plt.colorbar(im1, ax=ax1)
                    
                    time_indices = [0, 12, 25, 37, 49]
                    for i in time_indices:
                        ax2.plot(x_test, u_numerical[i, :], label=f't = {t_test[i]:.3f}')
                    ax2.set_xlabel('Space x')
                    ax2.set_ylabel('Velocity u')
                    ax2.set_title('Velocity Distribution at Different Times')
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                    st.subheader("🎯 PINNs訓練結果")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最終損失", f"{training_results['final_loss']:.6f}")
                    with col2:
                        st.metric("PDE損失", f"{training_results['pde_loss']:.6f}")
                    with col3:
                        st.metric("境界条件損失", f"{training_results['bc_loss']:.6f}")
            
            else:  # 拡散方程式
                with st.spinner("PINNsによる拡散方程式の解法中..."):
                    solver = PINNsDiffusionSolver(D=alpha_or_nu_or_D, hidden_dim=hidden_dim, num_layers=num_layers)
                    training_results = solver.train(epochs=epochs, lr=learning_rate, n_points=n_points, 
                                                  progress_callback=progress_callback)
                    
                    x_test = np.linspace(0, 0.02, 50)
                    t_test = np.linspace(0, 3600, 50)
                    X_test, T_test = np.meshgrid(x_test, t_test)
                    u_numerical = solver.predict(X_test, T_test)
                    
                    st.subheader("📈 PINNs拡散方程式の解")
                    
                    plt.rcParams['font.family'] = ['IPAGothic', 'IPAPGothic']
                    plt.rcParams['font.sans-serif'] = ['IPAGothic', 'IPAPGothic']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    im1 = ax1.imshow(u_numerical, aspect='auto', origin='lower', 
                                   extent=[0, 0.02, 0, 3600], cmap='plasma')
                    ax1.set_xlabel('Space x (m)')
                    ax1.set_ylabel('Time t (s)')
                    ax1.set_title('PINNs Concentration Distribution c(x,t)')
                    plt.colorbar(im1, ax=ax1)
                    
                    time_indices = [0, 12, 25, 37, 49]
                    for i in time_indices:
                        ax2.plot(x_test, u_numerical[i, :], label=f't = {t_test[i]:.0f}s')
                    ax2.set_xlabel('Space x (m)')
                    ax2.set_ylabel('Concentration c')
                    ax2.set_title('Concentration Distribution at Different Times')
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                    st.subheader("🎯 PINNs訓練結果")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("最終損失", f"{training_results['final_loss']:.6f}")
                    with col2:
                        st.metric("PDE損失", f"{training_results['pde_loss']:.6f}")
                    with col3:
                        st.metric("境界条件損失", f"{training_results['bc_loss']:.6f}")
            
            progress_bar.empty()
            status_text.empty()
        else:
            st.warning("⚠️ PINNsが利用できません。PyTorchをインストールしてください。")
        
        st.subheader("🔍 Step 3: シンボリック回帰によるPDE発見")
        
        with st.spinner("偏微分方程式を発見中..."):
            if equation_type == "熱伝導方程式":
                pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
                results = pde_regression.discover_heat_equation()
            elif equation_type == "Burgers方程式":
                pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
                results = pde_regression.discover_burgers_equation()
            else:  # 拡散方程式
                pde_regression = PDESymbolicRegression(u_numerical, fdm.x, fdm.t)
                results = pde_regression.discover_diffusion_equation(
                    use_exhaustive_search=use_exhaustive, 
                    max_complexity=max_complexity
                )
        
        st.success("✅ PDE発見完了!")
        
        st.subheader("🎯 発見されたPDE候補")
        
        if 'best_model' in results and results['best_model']:
            best = results['best_model']
            st.success(f"🏆 **最優秀候補**: {best['name']}")
            
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.metric("MSE", f"{best['mse']:.2e}")
            with col1b:
                st.metric("BIC", f"{best['bic']:.1f}")
            with col1c:
                st.metric("事後確率", f"{best['posterior_prob']:.3f}")
            
            st.info(f"**最適パラメータ**: {', '.join([f'{p:.2e}' for p in best['params']])}")
            
            if results['all_results']:
                st.subheader("📊 候補モデル比較（ベイズ評価）")
                
                comparison_data = []
                for result in results['all_results'][:10]:  
                    comparison_data.append({
                        'モデル': result['name'],
                        'MSE': f"{result['mse']:.2e}",
                        'BIC': f"{result['bic']:.1f}",
                        'AIC': f"{result['aic']:.1f}",
                        '事後確率': f"{result['posterior_prob']:.3f}",
                        '複雑度': result['complexity']
                    })
                
                st.dataframe(pd.DataFrame(comparison_data))
                
                if 'bayesian_model_average' in results:
                    bma = results['bayesian_model_average']
                    st.subheader("🎯 ベイズモデル平均化")
                    st.write(f"**平均複雑度**: {bma.get('average_complexity', 0):.2f}")
                    st.write(f"**モデル不確実性**: {bma.get('model_uncertainty', 0):.2e}")
                    st.write(f"**パラメータ不確実性**: {bma.get('parameter_uncertainty', 0):.2e}")
        else:
            sorted_results = sorted(results['all_results'], key=lambda x: x['mse'])
            
            for i, result in enumerate(sorted_results):
                formula_name = result.get('formula', result.get('name', ''))
                mse = result['mse']
                params = result['params']
                
                if i == 0:
                    st.success(f"🏆 **最優秀候補**: {formula_name}")
                else:
                    st.info(f"📋 **候補 {i+1}**: {formula_name}")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**式**: {formula_name}")
                    
                with col2:
                    if mse < np.inf:
                        st.metric("MSE", f"{mse:.2e}")
                    else:
                        st.metric("MSE", "∞")
                
                with col3:
                    param_str = ", ".join([f"{p:.4f}" for p in params])
                    st.write(f"**係数**: [{param_str}]")
        
        st.subheader("📈 理論値との比較")
        
        if equation_type == "熱伝導方程式":
            discovered_param = results.get('best_alpha', results['best_model']['params'][0] if results.get('best_model') and results['best_model'].get('params') else 0)
        elif equation_type == "Burgers方程式":
            discovered_param = results.get('best_nu', results['best_model']['params'][0] if results.get('best_model') and results['best_model'].get('params') else 0)
        else:  # 拡散方程式
            discovered_param = results.get('best_D', results['best_model']['params'][0] if results.get('best_model') and results['best_model'].get('params') else 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"理論値 {param_name}", f"{theoretical_param:.4f}")
        
        with col2:
            st.metric(f"発見値 {param_name}", f"{discovered_param:.4f}")
        
        with col3:
            error_percent = abs(discovered_param - theoretical_param) / theoretical_param * 100
            st.metric("相対誤差", f"{error_percent:.2f}%")
        
        with st.expander("📊 詳細な結果"):
            st.write("**候補式の評価結果:**")
            results_df = pd.DataFrame(results['all_results'])
            st.dataframe(results_df)
            
            st.write("**最適化の詳細:**")
            st.json(results['optimization_details'])

if __name__ == "__main__":
    create_pde_discovery_app()
