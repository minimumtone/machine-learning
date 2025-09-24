"""
発展課題：自動式生成とシンボリック回帰
SymPyを使用した式の自動生成と複雑度ペナルティの実装
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sympy as sp
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any, Generator
import itertools
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="発展課題：自動シンボリック回帰",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 発展課題：自動シンボリック回帰")
st.markdown("""
このアプリは、式の自動生成と複雑度ペナルティを導入した高度なシンボリック回帰システムです。
SymPyを使用して数式を自動的に生成し、オッカムの剃刀の原理に基づいて最適な式を選択します。
""")

class SymbolicRegressor:
    """シンボリック回帰クラス"""
    
    def __init__(self, variables: List[str], max_complexity: int = 3, alpha: float = 0.01):
        """
        初期化
        
        :param variables: 使用する変数のリスト
        :param max_complexity: 最大複雑度
        :param alpha: 複雑度ペナルティの重み
        """
        self.variables = variables
        self.max_complexity = max_complexity
        self.alpha = alpha
        self.symbols = {var: sp.Symbol(var) for var in variables}
        self.constants = []
        
        self.operators = {
            'add': lambda x, y: x + y,
            'mul': lambda x, y: x * y,
            'div': lambda x, y: x / y,
            'pow': lambda x, y: x ** y,
            'sqrt': lambda x: sp.sqrt(x),
            'sin': lambda x: sp.sin(x),
            'cos': lambda x: sp.cos(x),
            'exp': lambda x: sp.exp(x),
            'log': lambda x: sp.log(x)
        }
    
    def generate_expressions(self, complexity: int) -> Generator[sp.Expr, None, None]:
        """
        指定された複雑度の式を生成
        
        :param complexity: 式の複雑度（ノード数）
        :yield: SymPy式
        """
        if complexity == 1:
            for var in self.variables:
                yield self.symbols[var]
            yield sp.Symbol('c0')
        
        elif complexity == 2:
            for expr in self.generate_expressions(1):
                if expr != sp.Symbol('c0'):  # 定数には単項演算子を適用しない
                    yield sp.sqrt(expr)
                    yield sp.sin(expr)
                    yield sp.cos(expr)
        
        elif complexity >= 3:
            for i in range(1, complexity):
                j = complexity - i - 1
                if j >= 1:
                    for expr1 in self.generate_expressions(i):
                        for expr2 in self.generate_expressions(j):
                            yield expr1 + expr2
                            yield expr1 * expr2
                            if not expr2.equals(0):
                                yield expr1 / expr2
                            if expr2.is_number and abs(float(expr2)) <= 3:
                                yield expr1 ** expr2
    
    def calculate_complexity(self, expr: sp.Expr) -> int:
        """式の複雑度を計算"""
        return len(expr.atoms(sp.Symbol)) + len(expr.atoms(sp.Function))
    
    def expression_to_function(self, expr: sp.Expr) -> Tuple[Callable, List[str]]:
        """
        SymPy式をPython関数に変換
        
        :param expr: SymPy式
        :return: (評価関数, 定数名のリスト)
        """
        constants = [str(atom) for atom in expr.atoms(sp.Symbol) if str(atom).startswith('c')]
        constants.sort()
        
        variables = [str(atom) for atom in expr.atoms(sp.Symbol) if str(atom) in self.variables]
        
        try:
            all_symbols = constants + variables
            func = sp.lambdify(all_symbols, expr, 'numpy')
            
            def evaluation_function(params: np.ndarray, X: pd.DataFrame) -> np.ndarray:
                args = list(params) + [X[var].values for var in variables]
                return func(*args)
            
            return evaluation_function, constants
        
        except Exception:
            return None, []
    
    def evaluate_expression(self, expr: sp.Expr, X: pd.DataFrame, y: pd.Series) -> Tuple[float, np.ndarray]:
        """
        式を評価してMSEと最適定数を返す
        
        :param expr: SymPy式
        :param X: 説明変数データ
        :param y: 目的変数データ
        :return: (MSE, 最適定数)
        """
        func, constants = self.expression_to_function(expr)
        
        if func is None or len(constants) == 0:
            return 1e10, np.array([])
        
        initial_params = np.ones(len(constants))
        
        def objective(params):
            try:
                y_pred = func(params, X)
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    return 1e10
                return np.mean((y - y_pred)**2)
            except:
                return 1e10
        
        try:
            result = minimize(objective, initial_params, method='Nelder-Mead')
            if result.success:
                return result.fun, result.x
            else:
                return 1e10, initial_params
        except:
            return 1e10, initial_params
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        シンボリック回帰を実行
        
        :param X: 説明変数データ
        :param y: 目的変数データ
        :return: 結果辞書
        """
        best_score = float('inf')
        best_expr = None
        best_params = None
        best_mse = None
        
        results = []
        
        for complexity in range(1, self.max_complexity + 1):
            st.write(f"複雑度 {complexity} の式を探索中...")
            
            expr_count = 0
            for expr in self.generate_expressions(complexity):
                expr_count += 1
                if expr_count > 100:  # 計算時間制限
                    break
                
                try:
                    simplified_expr = sp.simplify(expr)
                except:
                    continue
                
                mse, params = self.evaluate_expression(simplified_expr, X, y)
                
                complexity_penalty = self.alpha * self.calculate_complexity(simplified_expr)
                score = mse + complexity_penalty
                
                results.append({
                    'expression': str(simplified_expr),
                    'complexity': self.calculate_complexity(simplified_expr),
                    'mse': mse,
                    'penalty': complexity_penalty,
                    'score': score,
                    'params': params
                })
                
                if score < best_score:
                    best_score = score
                    best_expr = simplified_expr
                    best_params = params
                    best_mse = mse
        
        return {
            'best_expression': best_expr,
            'best_params': best_params,
            'best_mse': best_mse,
            'best_score': best_score,
            'all_results': results
        }

def load_physics_data():
    """物理法則データを読み込み"""
    try:
        kinetic_data = pd.read_csv('kinetic_energy.csv')
        pendulum_data = pd.read_csv('pendulum.csv')
        gravity_data = pd.read_csv('gravity.csv')
        return kinetic_data, pendulum_data, gravity_data
    except FileNotFoundError:
        st.error("データファイルが見つかりません。先にdata_generation.pyを実行してください。")
        return None, None, None

def main():
    """メインアプリケーション"""
    
    st.sidebar.header("🤖 設定")
    
    dataset_choice = st.sidebar.selectbox(
        "データセットを選択:",
        ["運動エネルギー", "単振り子の周期", "万有引力"]
    )
    
    max_complexity = st.sidebar.slider("最大複雑度", 1, 5, 3)
    alpha = st.sidebar.slider("複雑度ペナルティ (α)", 0.0, 0.1, 0.01, 0.001)
    
    kinetic_data, pendulum_data, gravity_data = load_physics_data()
    
    if kinetic_data is None:
        st.stop()
    
    if dataset_choice == "運動エネルギー":
        data = kinetic_data
        X = data[['m', 'v']]
        y = data['K']
        variables = ['m', 'v']
        true_formula = "K = 0.5 × m × v²"
    
    elif dataset_choice == "単振り子の周期":
        data = pendulum_data
        X = data[['L', 'g']]  # mは除外（無関係な変数）
        y = data['T']
        variables = ['L', 'g']
        true_formula = "T = 2π√(L/g)"
    
    else:  # 万有引力
        data = gravity_data
        X = data[['m1', 'm2', 'r']]
        y = data['F']
        variables = ['m1', 'm2', 'r']
        true_formula = "F = G × (m₁×m₂)/r²"
    
    st.header(f"📊 {dataset_choice}データの自動解析")
    st.markdown(f"**理論式**: {true_formula}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 データ概要")
        st.dataframe(data.head())
        st.write(data.describe())
    
    with col2:
        st.subheader("🔍 データ可視化")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if len(variables) == 2:
            ax.scatter(X.iloc[:, 0], y, alpha=0.6, label=f'{variables[0]} vs target')
            ax.set_xlabel(variables[0])
            ax.set_ylabel('Target')
        else:
            ax.scatter(X.iloc[:, 0], y, alpha=0.6)
            ax.set_xlabel(variables[0])
            ax.set_ylabel('Target')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    if st.button("🚀 自動シンボリック回帰を実行"):
        with st.spinner("式を自動生成中..."):
            regressor = SymbolicRegressor(variables, max_complexity, alpha)
            results = regressor.fit(X, y)
        
        st.success("✅ 解析完了！")
        
        st.header("🏆 最良の発見式")
        if results['best_expression'] is not None:
            st.success(f"**発見式**: {results['best_expression']}")
            st.info(f"**MSE**: {results['best_mse']:.6f}")
            st.info(f"**複雑度ペナルティ付きスコア**: {results['best_score']:.6f}")
            st.info(f"**最適定数**: {', '.join([f'{p:.4f}' for p in results['best_params']])}")
        else:
            st.error("適切な式が見つかりませんでした。")
        
        st.header("📊 全候補式の結果")
        if results['all_results']:
            results_df = pd.DataFrame(results['all_results'])
            results_df = results_df.sort_values('score').head(20)  # 上位20件
            
            display_df = results_df.copy()
            display_df['mse'] = display_df['mse'].apply(lambda x: f"{x:.6f}")
            display_df['penalty'] = display_df['penalty'].apply(lambda x: f"{x:.6f}")
            display_df['score'] = display_df['score'].apply(lambda x: f"{x:.6f}")
            
            st.dataframe(display_df[['expression', 'complexity', 'mse', 'penalty', 'score']])
            
            st.subheader("📈 複雑度 vs スコア")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            complexities = [r['complexity'] for r in results['all_results']]
            scores = [r['score'] for r in results['all_results']]
            mses = [r['mse'] for r in results['all_results']]
            
            ax.scatter(complexities, scores, alpha=0.6, label='Total Score')
            ax.scatter(complexities, mses, alpha=0.6, label='MSE only')
            ax.set_xlabel('Complexity')
            ax.set_ylabel('Score')
            ax.set_title('Complexity vs Score (Occam\'s Razor Effect)')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.header("🔍 結果の解釈")
        st.markdown("""
        **オッカムの剃刀の効果**:
        - 複雑度ペナルティにより、同程度の精度なら単純な式が選ばれます
        - αパラメータで精度と単純さのバランスを調整できます
        - 真の物理法則は通常、単純で美しい形をしています
        
        **発見された式の評価**:
        - 理論式と比較して構造的類似性を確認してください
        - 定数値が物理的に妥当かを検証してください
        - 複雑度が適切かを評価してください
        """)

if __name__ == "__main__":
    main()
