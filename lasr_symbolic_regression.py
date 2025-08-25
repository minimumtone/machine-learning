"""
LaSR (Library-Augmented Symbolic Regression) Implementation

Based on the NeurIPS 2024 paper "Symbolic Regression with a Learned Concept Library"
by Arya Grayeli et al.

This implementation combines evolutionary algorithms with LLM-guided concept discovery
through three main phases:
1. Concept-directed hypothesis evolution
2. LLM-based abstraction of patterns into new concepts  
3. LLM-directed evolution of concepts
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Callable, Any, Optional, Generator
import itertools
import warnings
import json
import random
import time
from dataclasses import dataclass
import openai
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class LaSRConfig:
    """Configuration for LaSR algorithm"""
    num_iterations: int = 40
    num_populations: int = 10
    population_size: int = 50
    concept_evolution_steps: int = 3
    llm_probability: float = 0.01
    max_complexity: int = 5
    alpha: float = 0.01
    llm_model: str = "gpt-3.5-turbo"
    max_concept_library_size: int = 100
    temperature: float = 0.7
    max_tokens: int = 150

class ConceptLibrary:
    """Manages the natural language concept library for LaSR"""
    
    def __init__(self, initial_concepts: Optional[List[str]] = None):
        self.concepts = initial_concepts or []
        self.concept_scores = {}
        
    def add_concept(self, concept: str, score: float = 0.0):
        """Add a new concept to the library"""
        if concept not in self.concepts:
            self.concepts.append(concept)
            self.concept_scores[concept] = score
            
    def sample_concepts(self, num_concepts: int = 3) -> List[str]:
        """Sample concepts from the library"""
        if not self.concepts:
            return []
        return random.sample(self.concepts, min(num_concepts, len(self.concepts)))
    
    def get_all_concepts(self) -> List[str]:
        """Get all concepts in the library"""
        return self.concepts.copy()
    
    def size(self) -> int:
        """Get the size of the concept library"""
        return len(self.concepts)

class LLMInterface:
    """Interface for LLM interactions"""
    
    def __init__(self, config: LaSRConfig):
        self.config = config
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
            else:
                st.warning("OpenAI API key not found. LLM features will be disabled.")
        except Exception as e:
            st.warning(f"Failed to initialize OpenAI client: {e}")
            
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Make a call to the LLM"""
        if not self.client:
            return None
            
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"LLM call failed: {e}")
            return None
    
    def llm_initialize(self, variables: List[str], concepts: List[str]) -> Optional[str]:
        """LLM-guided initialization of expressions"""
        if not concepts:
            concepts = ["basic mathematical operations", "polynomial relationships"]
            
        prompt = f"""Generate a mathematical expression using the variables {variables}.

Consider these concepts: {', '.join(concepts)}

The expression should be in Python/SymPy format using these variables: {', '.join(variables)}
Use basic operations: +, -, *, /, **, sqrt(), sin(), cos(), exp(), log()

Return only the mathematical expression, no explanation.

Example format: x**2 + y*sin(z)"""

        return self._call_llm(prompt)
    
    def llm_mutate(self, expression: str, variables: List[str], concepts: List[str]) -> Optional[str]:
        """LLM-guided mutation of expressions"""
        if not concepts:
            concepts = ["mathematical transformations", "function composition"]
            
        prompt = f"""Mutate this mathematical expression: {expression}

Variables available: {', '.join(variables)}
Consider these concepts: {', '.join(concepts)}

Create a similar but modified expression. You can:
- Change operators (+, -, *, /, **)
- Add/remove terms
- Apply functions (sqrt, sin, cos, exp, log)
- Modify structure slightly

Return only the mutated mathematical expression, no explanation.

Original: {expression}
Mutated:"""

        return self._call_llm(prompt)
    
    def llm_crossover(self, expr1: str, expr2: str, variables: List[str], concepts: List[str]) -> Optional[str]:
        """LLM-guided crossover of expressions"""
        if not concepts:
            concepts = ["combining mathematical structures", "hybrid expressions"]
            
        prompt = f"""Combine these two mathematical expressions: 
Expression 1: {expr1}
Expression 2: {expr2}

Variables available: {', '.join(variables)}
Consider these concepts: {', '.join(concepts)}

Create a new expression that combines elements from both expressions.
You can mix terms, operators, and structures from both expressions.

Return only the combined mathematical expression, no explanation.

Combined:"""

        return self._call_llm(prompt)
    
    def generate_concepts(self, good_expressions: List[str], bad_expressions: List[str], variables: List[str]) -> List[str]:
        """Generate new concepts from expression analysis"""
        if not good_expressions:
            return []
            
        prompt = f"""Analyze these mathematical expressions and identify patterns:

GOOD expressions (high performance):
{chr(10).join(f"- {expr}" for expr in good_expressions[:5])}

BAD expressions (low performance):
{chr(10).join(f"- {expr}" for expr in bad_expressions[:3])}

Variables: {', '.join(variables)}

Identify 2-3 abstract concepts or patterns that make the good expressions successful.
Focus on mathematical structures, relationships, or principles.

Return concepts as a JSON list of strings.
Example: ["trigonometric functions enhance periodic relationships", "polynomial terms capture growth patterns"]

Concepts:"""

        response = self._call_llm(prompt)
        if response:
            try:
                concepts = json.loads(response)
                return concepts if isinstance(concepts, list) else []
            except json.JSONDecodeError:
                lines = response.strip().split('\n')
                concepts = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('[') and not line.startswith(']'):
                        line = line.strip('"-,')
                        if line:
                            concepts.append(line)
                return concepts[:3]
        return []
    
    def evolve_concepts(self, existing_concepts: List[str]) -> List[str]:
        """Evolve existing concepts into new ones"""
        if not existing_concepts:
            return []
            
        sampled_concepts = random.sample(existing_concepts, min(3, len(existing_concepts)))
        
        prompt = f"""Given these mathematical concepts:
{chr(10).join(f"- {concept}" for concept in sampled_concepts)}

Generate 2-3 new related concepts that logically follow or extend these ideas.
Focus on mathematical principles, relationships, or structures.

Return concepts as a JSON list of strings.

New concepts:"""

        response = self._call_llm(prompt)
        if response:
            try:
                concepts = json.loads(response)
                return concepts if isinstance(concepts, list) else []
            except json.JSONDecodeError:
                lines = response.strip().split('\n')
                concepts = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('[') and not line.startswith(']'):
                        line = line.strip('"-,')
                        if line:
                            concepts.append(line)
                return concepts[:3]
        return []

class LaSRRegressor:
    """Library-Augmented Symbolic Regression"""
    
    def __init__(self, config: LaSRConfig = None):
        self.config = config or LaSRConfig()
        self.concept_library = ConceptLibrary()
        self.llm_interface = LLMInterface(self.config)
        self.variables = []
        self.populations = []
        self.best_expressions = []
        self.history = []
        
    def _string_to_sympy(self, expr_str: str) -> Optional[sp.Expr]:
        """Convert string expression to SymPy expression"""
        try:
            symbols = {var: sp.Symbol(var) for var in self.variables}
            symbols.update({
                'sqrt': sp.sqrt,
                'sin': sp.sin,
                'cos': sp.cos,
                'exp': sp.exp,
                'log': sp.log,
                'pi': sp.pi,
                'e': sp.E
            })
            
            expr = eval(expr_str, {"__builtins__": {}}, symbols)
            return sp.sympify(expr)
        except:
            return None
    
    def _sympy_to_function(self, expr: sp.Expr) -> Callable:
        """Convert SymPy expression to evaluable function"""
        symbols = [sp.Symbol(var) for var in self.variables]
        try:
            func = sp.lambdify(symbols, expr, 'numpy')
            return func
        except:
            return lambda *args: np.full(len(args[0]) if hasattr(args[0], '__len__') else 1, np.inf)
    
    def _evaluate_expression(self, expr: sp.Expr, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        """Evaluate expression fitness"""
        try:
            func = self._sympy_to_function(expr)
            
            if len(self.variables) == 1:
                y_pred = func(X[self.variables[0]].values)
            else:
                args = [X[var].values for var in self.variables]
                y_pred = func(*args)
            
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return np.inf, np.inf
                
            mse = np.mean((y.values - y_pred) ** 2)
            complexity = len(expr.atoms(sp.Symbol)) + len(expr.atoms(sp.Function))
            score = mse + self.config.alpha * complexity
            
            return mse, score
        except:
            return np.inf, np.inf
    
    def _generate_random_expression(self) -> Optional[sp.Expr]:
        """Generate a random expression"""
        if random.random() < self.config.llm_probability and self.llm_interface.client:
            concepts = self.concept_library.sample_concepts()
            expr_str = self.llm_interface.llm_initialize(self.variables, concepts)
            if expr_str:
                expr = self._string_to_sympy(expr_str)
                if expr is not None:
                    return expr
        
        symbols = [sp.Symbol(var) for var in self.variables]
        if len(symbols) == 1:
            return symbols[0] ** random.randint(1, 3)
        else:
            return symbols[0] * symbols[1] if len(symbols) >= 2 else symbols[0]
    
    def _mutate_expression(self, expr: sp.Expr) -> sp.Expr:
        """Mutate an expression"""
        if random.random() < self.config.llm_probability and self.llm_interface.client:
            concepts = self.concept_library.sample_concepts()
            expr_str = self.llm_interface.llm_mutate(str(expr), self.variables, concepts)
            if expr_str:
                new_expr = self._string_to_sympy(expr_str)
                if new_expr is not None:
                    return new_expr
        
        symbols = [sp.Symbol(var) for var in self.variables]
        operations = [
            lambda e: e + random.choice(symbols),
            lambda e: e * random.choice(symbols),
            lambda e: e ** 2 if random.random() < 0.3 else e,
            lambda e: sp.sqrt(e) if random.random() < 0.2 else e
        ]
        
        try:
            return random.choice(operations)(expr)
        except:
            return expr
    
    def _crossover_expressions(self, expr1: sp.Expr, expr2: sp.Expr) -> sp.Expr:
        """Crossover two expressions"""
        if random.random() < self.config.llm_probability and self.llm_interface.client:
            concepts = self.concept_library.sample_concepts()
            expr_str = self.llm_interface.llm_crossover(str(expr1), str(expr2), self.variables, concepts)
            if expr_str:
                new_expr = self._string_to_sympy(expr_str)
                if new_expr is not None:
                    return new_expr
        
        try:
            return expr1 + expr2
        except:
            return expr1
    
    def _extract_pareto_frontier(self, population: List[Tuple[sp.Expr, float, float]]) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """Extract Pareto frontier (good and bad expressions)"""
        if not population:
            return [], []
        
        population.sort(key=lambda x: x[2])
        
        good_expressions = [expr for expr, mse, score in population[:len(population)//4]]
        bad_expressions = [expr for expr, mse, score in population[-len(population)//4:]]
        
        return good_expressions, bad_expressions
    
    def _concept_abstraction(self, good_expressions: List[sp.Expr], bad_expressions: List[sp.Expr]):
        """Extract concepts from expression analysis"""
        if not good_expressions or not self.llm_interface.client:
            return
            
        good_strs = [str(expr) for expr in good_expressions]
        bad_strs = [str(expr) for expr in bad_expressions]
        
        new_concepts = self.llm_interface.generate_concepts(good_strs, bad_strs, self.variables)
        
        for concept in new_concepts:
            if concept and len(concept) > 10:
                self.concept_library.add_concept(concept)
    
    def _concept_evolution(self):
        """Evolve the concept library"""
        if self.concept_library.size() == 0 or not self.llm_interface.client:
            return
            
        for _ in range(self.config.concept_evolution_steps):
            new_concepts = self.llm_interface.evolve_concepts(self.concept_library.get_all_concepts())
            for concept in new_concepts:
                if concept and len(concept) > 10:
                    self.concept_library.add_concept(concept)
        
        if self.concept_library.size() > self.config.max_concept_library_size:
            self.concept_library.concepts = self.concept_library.concepts[:self.config.max_concept_library_size]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, user_hints: List[str] = None) -> Tuple[sp.Expr, float]:
        """Fit the LaSR model"""
        self.variables = list(X.columns)
        
        if user_hints:
            for hint in user_hints:
                self.concept_library.add_concept(hint)
        
        all_expressions = []
        
        for iteration in range(self.config.num_iterations):
            st.write(f"LaSR Iteration {iteration + 1}/{self.config.num_iterations}")
            
            if iteration == 0:
                population = []
                for _ in range(self.config.population_size):
                    expr = self._generate_random_expression()
                    if expr is not None:
                        mse, score = self._evaluate_expression(expr, X, y)
                        population.append((expr, mse, score))
            else:
                new_population = []
                for _ in range(self.config.population_size):
                    if random.random() < 0.3:
                        expr = self._generate_random_expression()
                    elif random.random() < 0.6 and len(population) > 0:
                        parent = random.choice(population)[0]
                        expr = self._mutate_expression(parent)
                    else:
                        if len(population) >= 2:
                            parent1, parent2 = random.sample(population, 2)
                            expr = self._crossover_expressions(parent1[0], parent2[0])
                        else:
                            expr = self._generate_random_expression()
                    
                    if expr is not None:
                        mse, score = self._evaluate_expression(expr, X, y)
                        new_population.append((expr, mse, score))
                
                population.extend(new_population)
                population.sort(key=lambda x: x[2])
                population = population[:self.config.population_size]
            
            all_expressions.extend(population)
            
            good_expressions, bad_expressions = self._extract_pareto_frontier(population)
            
            self._concept_abstraction(good_expressions, bad_expressions)
            
            if iteration % 5 == 0:
                self._concept_evolution()
            
            best_expr, best_mse, best_score = min(population, key=lambda x: x[2])
            self.history.append({
                'iteration': iteration + 1,
                'best_mse': best_mse,
                'best_score': best_score,
                'best_expression': str(best_expr),
                'concept_library_size': self.concept_library.size()
            })
            
            if iteration % 10 == 0:
                st.write(f"Best expression: {best_expr}")
                st.write(f"MSE: {best_mse:.6f}, Score: {best_score:.6f}")
                st.write(f"Concept library size: {self.concept_library.size()}")
        
        all_expressions.sort(key=lambda x: x[2])
        best_expr, best_mse, best_score = all_expressions[0]
        
        return best_expr, best_mse

def load_physics_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load physics datasets"""
    if dataset_name == "kinetic_energy":
        try:
            data = pd.read_csv('kinetic_energy.csv')
            return data[['m', 'v']], data['K']
        except:
            m = np.random.uniform(1, 10, 100)
            v = np.random.uniform(1, 20, 100)
            K = 0.5 * m * v**2 + np.random.normal(0, 0.1, 100)
            return pd.DataFrame({'m': m, 'v': v}), pd.Series(K)
    
    elif dataset_name == "pendulum":
        try:
            data = pd.read_csv('pendulum.csv')
            return data[['L', 'g']], data['T']
        except:
            L = np.random.uniform(0.5, 5, 100)
            g = np.random.uniform(9.8, 10.2, 100)
            T = 2 * np.pi * np.sqrt(L / g) + np.random.normal(0, 0.01, 100)
            return pd.DataFrame({'L': L, 'g': g}), pd.Series(T)
    
    elif dataset_name == "gravity":
        try:
            data = pd.read_csv('gravity.csv')
            return data[['m1', 'm2', 'r']], data['F']
        except:
            G = 6.674e-11
            m1 = 1e10 * np.random.uniform(1, 10, 100)
            m2 = 1e10 * np.random.uniform(1, 10, 100)
            r = np.random.uniform(100, 1000, 100)
            F = G * (m1 * m2) / r**2 + np.random.normal(0, 1e-5, 100)
            return pd.DataFrame({'m1': m1, 'm2': m2, 'r': r}), pd.Series(F)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
