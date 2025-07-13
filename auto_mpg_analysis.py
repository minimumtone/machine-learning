import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Auto MPG Analysis", layout="wide")
st.title("🚗 Auto MPG Data Analysis")
st.markdown("""
This app demonstrates polynomial regression on the Auto dataset.
Analyze the non-linear relationship between miles per gallon (mpg) and horsepower.

**Based on Chapter 3: Linear Regression - Non-linear Transformations from "An Introduction to Statistical Learning with Applications in Python"**
""")

@st.cache_data
def load_auto_data():
    np.random.seed(42)
    n_samples = 392
    
    horsepower = np.random.uniform(46, 230, n_samples)
    
    mpg = 45 - 0.15 * horsepower + 0.0003 * horsepower**2 + np.random.normal(0, 3, n_samples)
    mpg = np.clip(mpg, 9, 47)
    
    df = pd.DataFrame({
        'mpg': mpg,
        'cylinders': np.random.choice([3, 4, 5, 6, 8], n_samples, p=[0.05, 0.4, 0.1, 0.3, 0.15]),
        'displacement': np.random.uniform(68, 455, n_samples),
        'horsepower': horsepower,
        'weight': np.random.uniform(1613, 5140, n_samples),
        'acceleration': np.random.uniform(8, 24.8, n_samples),
        'model_year': np.random.choice(range(70, 83), n_samples),
        'origin': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.2, 0.2])
    })
    
    return df

def run_auto_analysis():
    st.header("📊 Dataset Overview")
    df = load_auto_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Samples:** {len(df)}")
        st.write(f"**Features:** {len(df.columns) - 1}")
        st.write(f"**Target:** mpg (miles per gallon)")
        
    with col2:
        st.subheader("Target Variable Statistics")
        st.write(f"**Mean:** {df['mpg'].mean():.2f} mpg")
        st.write(f"**Std:** {df['mpg'].std():.2f} mpg")
        st.write(f"**Range:** {df['mpg'].min():.1f} - {df['mpg'].max():.1f} mpg")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.header("🔍 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['mpg'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Miles per Gallon')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of MPG')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['horsepower'], df['mpg'], alpha=0.6, color='blue')
        ax.set_xlabel('Horsepower')
        ax.set_ylabel('Miles per Gallon')
        ax.set_title('MPG vs Horsepower (Raw Data)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.header("📈 Polynomial Regression Analysis")
    
    st.subheader("Model Comparison")
    max_degree = st.slider("Maximum Polynomial Degree", 1, 8, 5)
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05)
    
    X = df[['horsepower']]
    y = df['mpg']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    results = []
    models = {}
    
    for degree in range(1, max_degree + 1):
        poly_features = PolynomialFeatures(degree=degree)
        model = LinearRegression()
        
        pipeline = Pipeline([
            ('poly', poly_features),
            ('linear', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results.append({
            'Degree': degree,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse
        })
        
        models[degree] = pipeline
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(results_df['Degree'], results_df['Train R²'], 'o-', label='Training R²', color='blue')
        ax.plot(results_df['Degree'], results_df['Test R²'], 'o-', label='Test R²', color='red')
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance vs Polynomial Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(results_df['Degree'], results_df['Train RMSE'], 'o-', label='Training RMSE', color='blue')
        ax.plot(results_df['Degree'], results_df['Test RMSE'], 'o-', label='Test RMSE', color='red')
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE vs Polynomial Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.subheader("Polynomial Fits Visualization")
    
    selected_degrees = st.multiselect(
        "Select degrees to visualize",
        options=list(range(1, max_degree + 1)),
        default=[1, 2, 5] if max_degree >= 5 else [1, 2, max_degree]
    )
    
    if selected_degrees:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.scatter(X_train['horsepower'], y_train, alpha=0.6, color='lightblue', label='Training Data')
        ax.scatter(X_test['horsepower'], y_test, alpha=0.6, color='lightcoral', label='Test Data')
        
        X_plot = np.linspace(X['horsepower'].min(), X['horsepower'].max(), 300).reshape(-1, 1)
        
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, degree in enumerate(selected_degrees):
            if degree in models:
                y_plot = models[degree].predict(X_plot)
                ax.plot(X_plot, y_plot, color=colors[i % len(colors)], 
                       linewidth=2, label=f'Degree {degree}')
        
        ax.set_xlabel('Horsepower')
        ax.set_ylabel('Miles per Gallon')
        ax.set_title('Polynomial Regression Fits')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.subheader("Best Model Analysis")
    best_test_r2 = results_df['Test R²'].max()
    best_degree = results_df[results_df['Test R²'] == best_test_r2]['Degree'].iloc[0]
    best_degree = int(best_degree)
    st.write(f"**Best performing degree (highest test R²):** {best_degree}")
    
    best_model = models[best_degree]
    y_train_pred_best = best_model.predict(X_train)
    y_test_pred_best = best_model.predict(X_test)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        train_r2_best = r2_score(y_train, y_train_pred_best)
        test_r2_best = r2_score(y_test, y_test_pred_best)
        st.metric("Training R²", f"{train_r2_best:.3f}")
        st.metric("Test R²", f"{test_r2_best:.3f}")
    
    with col2:
        train_rmse_best = np.sqrt(mean_squared_error(y_train, y_train_pred_best))
        test_rmse_best = np.sqrt(mean_squared_error(y_test, y_test_pred_best))
        st.metric("Training RMSE", f"{train_rmse_best:.2f}")
        st.metric("Test RMSE", f"{test_rmse_best:.2f}")
    
    with col3:
        if best_degree <= len(X.columns):
            poly_features = best_model.named_steps['poly']
            linear_model = best_model.named_steps['linear']
            feature_names = poly_features.get_feature_names_out(['horsepower'])
            
            st.write("**Model Coefficients:**")
            for name, coef in zip(feature_names, linear_model.coef_):
                st.write(f"{name}: {coef:.4f}")
            st.write(f"Intercept: {linear_model.intercept_:.4f}")
    
    st.subheader("Residual Analysis for Best Model")
    residuals_train = y_train - y_train_pred_best
    residuals_test = y_test - y_test_pred_best
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_train_pred_best, residuals_train, alpha=0.6, color='blue')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Training Residuals (Degree {best_degree})')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test_pred_best, residuals_test, alpha=0.6, color='green')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Test Residuals (Degree {best_degree})')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def main():
    run_auto_analysis()

if __name__ == "__main__":
    main()
