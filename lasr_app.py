"""
LaSR (Library-Augmented Symbolic Regression) Streamlit Application

Interactive web application for testing and demonstrating the LaSR algorithm
for symbolic regression with learned concept libraries.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lasr_symbolic_regression import LaSRRegressor, LaSRConfig, load_physics_data
import sympy as sp
import time

plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="LaSR: Library-Augmented Symbolic Regression",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LaSR: Library-Augmented Symbolic Regression")
st.markdown("""
This application demonstrates **LaSR (Library-Augmented Symbolic Regression)**, a novel approach that combines 
evolutionary algorithms with Large Language Model (LLM) guided concept discovery.

**Key Features:**
- 🔄 **Concept-directed hypothesis evolution**: Mix traditional genetic operations with LLM-guided steps
- 🧠 **Concept abstraction**: Extract natural language concepts from high-performing expressions
- 🌱 **Concept evolution**: Evolve concepts into more general and useful forms

**Target Physical Laws:**
1. **Kinetic Energy**: K = 0.5 × m × v²
2. **Pendulum Period**: T = 2π√(L/g)  
3. **Gravitational Force**: F = G × (m₁×m₂)/r²
""")

st.sidebar.header("🔧 LaSR Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password",
    help="Enter your OpenAI API key to enable LLM features"
)

if api_key:
    import os
    os.environ['OPENAI_API_KEY'] = api_key
    st.sidebar.success("✅ API Key configured")
else:
    st.sidebar.warning("⚠️ LLM features disabled without API key")

dataset_choice = st.sidebar.selectbox(
    "Select Physics Dataset",
    ["kinetic_energy", "pendulum", "gravity"],
    help="Choose which physical law to discover"
)

st.sidebar.subheader("Algorithm Parameters")

num_iterations = st.sidebar.slider("Number of Iterations", 10, 100, 40)
llm_probability = st.sidebar.slider("LLM Probability (p)", 0.0, 0.2, 0.01, 0.001)
population_size = st.sidebar.slider("Population Size", 20, 100, 50)
max_complexity = st.sidebar.slider("Max Complexity", 3, 8, 5)
alpha = st.sidebar.slider("Complexity Penalty (α)", 0.0, 0.1, 0.01, 0.001)

llm_model = st.sidebar.selectbox(
    "LLM Model",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    help="Choose the LLM model for concept generation"
)

user_hints = st.sidebar.text_area(
    "User Hints (optional)",
    placeholder="Enter domain-specific hints, one per line:\ne.g., 'quadratic relationships are important'\n'trigonometric functions may be relevant'",
    help="Provide natural language hints to guide the search"
)

config = LaSRConfig(
    num_iterations=num_iterations,
    llm_probability=llm_probability,
    population_size=population_size,
    max_complexity=max_complexity,
    alpha=alpha,
    llm_model=llm_model
)

if st.button("🚀 Run LaSR Discovery", type="primary"):
    
    with st.spinner("Loading dataset..."):
        X, y = load_physics_data(dataset_choice)
    
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Input Variables (X):**")
        st.dataframe(X.head(10))
        
    with col2:
        st.write("**Target Variable (y):**")
        st.dataframe(y.head(10).to_frame('target'))
    
    st.subheader("🔍 LaSR Algorithm Execution")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    hints = [hint.strip() for hint in user_hints.split('\n') if hint.strip()] if user_hints else None
    
    regressor = LaSRRegressor(config)
    
    start_time = time.time()
    
    with st.expander("📝 Algorithm Progress", expanded=True):
        progress_container = st.container()
        
        with progress_container:
            best_expr, best_mse = regressor.fit(X, y, user_hints=hints)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    progress_bar.progress(1.0)
    status_text.success(f"✅ LaSR completed in {execution_time:.2f} seconds!")
    
    st.subheader("🎯 Discovery Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Best Expression", str(best_expr))
        st.metric("Final MSE", f"{best_mse:.6f}")
        st.metric("Concept Library Size", regressor.concept_library.size())
        st.metric("Execution Time", f"{execution_time:.2f}s")
    
    with col2:
        if regressor.concept_library.size() > 0:
            st.write("**Learned Concepts:**")
            concepts = regressor.concept_library.get_all_concepts()
            for i, concept in enumerate(concepts[:10], 1):
                st.write(f"{i}. {concept}")
            if len(concepts) > 10:
                st.write(f"... and {len(concepts) - 10} more concepts")
        else:
            st.write("No concepts learned (LLM not available)")
    
    st.subheader("📈 Algorithm Performance")
    
    if regressor.history:
        history_df = pd.DataFrame(regressor.history)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        ax1.plot(history_df['iteration'], history_df['best_mse'])
        ax1.set_title('Best MSE Over Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MSE')
        ax1.set_yscale('log')
        
        ax2.plot(history_df['iteration'], history_df['best_score'])
        ax2.set_title('Best Score Over Iterations')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Score (MSE + α×Complexity)')
        ax2.set_yscale('log')
        
        ax3.plot(history_df['iteration'], history_df['concept_library_size'])
        ax3.set_title('Concept Library Growth')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Number of Concepts')
        
        final_expr = best_expr
        try:
            func = sp.lambdify(list(X.columns), final_expr, 'numpy')
            if len(X.columns) == 1:
                y_pred = func(X.iloc[:, 0].values)
            else:
                args = [X[col].values for col in X.columns]
                y_pred = func(*args)
            
            ax4.scatter(y.values, y_pred, alpha=0.6)
            ax4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax4.set_title('Predicted vs Actual')
            ax4.set_xlabel('Actual')
            ax4.set_ylabel('Predicted')
            
            r2 = 1 - np.sum((y.values - y_pred)**2) / np.sum((y.values - y.mean())**2)
            ax4.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax4.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            ax4.text(0.5, 0.5, 'Could not evaluate\nfinal expression', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("📋 Detailed History")
        st.dataframe(history_df)
    
    st.subheader("🔬 Expression Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Expression Complexity Analysis:**")
        try:
            symbols = len(best_expr.atoms(sp.Symbol))
            functions = len(best_expr.atoms(sp.Function))
            total_complexity = symbols + functions
            
            st.write(f"- Number of variables: {symbols}")
            st.write(f"- Number of functions: {functions}")
            st.write(f"- Total complexity: {total_complexity}")
            st.write(f"- Complexity penalty: {config.alpha * total_complexity:.6f}")
        except:
            st.write("Could not analyze expression complexity")
    
    with col2:
        st.write("**Expected vs Discovered:**")
        expected_formulas = {
            "kinetic_energy": "K = 0.5 × m × v²",
            "pendulum": "T = 2π√(L/g)",
            "gravity": "F = G × (m₁×m₂)/r²"
        }
        
        if dataset_choice in expected_formulas:
            st.write(f"**Expected:** {expected_formulas[dataset_choice]}")
            st.write(f"**Discovered:** {best_expr}")
            
            if dataset_choice == "kinetic_energy" and "v**2" in str(best_expr) and "m" in str(best_expr):
                st.success("✅ Correctly identified quadratic velocity dependence!")
            elif dataset_choice == "pendulum" and "sqrt" in str(best_expr) and "L" in str(best_expr):
                st.success("✅ Correctly identified square root relationship!")
            elif dataset_choice == "gravity" and "r**2" in str(best_expr) or "r**(-2)" in str(best_expr):
                st.success("✅ Correctly identified inverse square law!")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About LaSR:**

LaSR is based on the NeurIPS 2024 paper "Symbolic Regression with a Learned Concept Library" by Arya Grayeli et al.

The algorithm alternates between:
1. **Hypothesis Evolution** - Evolving mathematical expressions
2. **Concept Abstraction** - Learning patterns from successful expressions  
3. **Concept Evolution** - Evolving the concept library itself

This creates a feedback loop where discovered concepts guide future expression generation.
""")

if st.sidebar.button("ℹ️ Show Algorithm Details"):
    st.subheader("🔬 LaSR Algorithm Details")
    
    st.markdown("""
    
    LaSR extends traditional genetic programming for symbolic regression by incorporating a **learned concept library**. 
    The algorithm maintains a library of natural language concepts that capture useful patterns and guide the search process.
    
    
    - Traditional genetic operations (mutation, crossover, initialization) are augmented with LLM-guided versions
    - With probability `p`, LLM operations replace standard genetic operations
    - LLM operations use sampled concepts from the library to guide expression generation
    
    - After each iteration, extract Pareto frontier of best and worst performing expressions
    - Use LLM to analyze patterns in successful expressions vs unsuccessful ones
    - Generate new natural language concepts that capture these patterns
    - Add concepts to the library for future use
    
    - Periodically evolve the concept library itself
    - Sample existing concepts and use LLM to generate related/extended concepts
    - This allows the library to grow and become more sophisticated over time
    
    - **p (LLM Probability)**: Frequency of LLM-guided operations vs traditional genetic operations
    - **α (Complexity Penalty)**: Weight for expression complexity in fitness function
    - **Iterations**: Number of evolution cycles
    - **Population Size**: Number of expressions maintained per iteration
    
    - **Semantic Guidance**: Natural language concepts provide semantic meaning to guide search
    - **Knowledge Transfer**: Concepts learned from one problem can help with related problems  
    - **Interpretability**: The concept library provides insight into what patterns the algorithm finds useful
    - **Performance**: Significantly outperforms traditional symbolic regression on benchmark problems
    """)
