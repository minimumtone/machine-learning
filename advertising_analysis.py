import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Advertising Analysis", layout="wide")
st.title("📺 Advertising Data Analysis")
st.markdown("""
This app demonstrates multiple linear regression on the Advertising dataset.
Analyze the relationship between advertising budgets (TV, Radio, Newspaper) and Sales.

**Based on Chapter 3: Linear Regression - Multiple Linear Regression from "An Introduction to Statistical Learning with Applications in Python"**
""")

@st.cache_data
def load_advertising_data():
    np.random.seed(42)
    n_samples = 200
    
    tv = np.random.uniform(0, 300, n_samples)
    radio = np.random.uniform(0, 50, n_samples)
    newspaper = np.random.uniform(0, 115, n_samples)
    
    sales = (7.03 + 0.0475 * tv + 0.189 * radio + 0.001 * newspaper + 
             0.001 * tv * radio + np.random.normal(0, 1.5, n_samples))
    sales = np.clip(sales, 1, 30)
    
    df = pd.DataFrame({
        'TV': tv,
        'Radio': radio,
        'Newspaper': newspaper,
        'Sales': sales
    })
    
    return df

def run_advertising_analysis():
    st.header("📊 Dataset Overview")
    df = load_advertising_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Samples:** {len(df)}")
        st.write("**Features:** TV, Radio, Newspaper budgets")
        st.write("**Target:** Sales (in thousands of units)")
        
    with col2:
        st.subheader("Target Variable Statistics")
        st.write(f"**Mean Sales:** {df['Sales'].mean():.2f}k units")
        st.write(f"**Std:** {df['Sales'].std():.2f}k units")
        st.write(f"**Range:** {df['Sales'].min():.1f} - {df['Sales'].max():.1f}k units")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.header("🔍 Exploratory Data Analysis")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(df['Sales'], bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_xlabel('Sales (thousands of units)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Sales')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(df['TV'], df['Sales'], alpha=0.6, color='red')
    axes[0, 1].set_xlabel('TV Budget ($1000s)')
    axes[0, 1].set_ylabel('Sales (thousands of units)')
    axes[0, 1].set_title('Sales vs TV Budget')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(df['Radio'], df['Sales'], alpha=0.6, color='green')
    axes[1, 0].set_xlabel('Radio Budget ($1000s)')
    axes[1, 0].set_ylabel('Sales (thousands of units)')
    axes[1, 0].set_title('Sales vs Radio Budget')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(df['Newspaper'], df['Sales'], alpha=0.6, color='orange')
    axes[1, 1].set_xlabel('Newspaper Budget ($1000s)')
    axes[1, 1].set_ylabel('Sales (thousands of units)')
    axes[1, 1].set_title('Sales vs Newspaper Budget')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Correlation Matrix")
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
    
    st.header("📈 Linear Regression Analysis")
    
    st.subheader("Model Comparison")
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05)
    
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    models = {
        'TV Only': ['TV'],
        'Radio Only': ['Radio'],
        'Newspaper Only': ['Newspaper'],
        'TV + Radio': ['TV', 'Radio'],
        'TV + Newspaper': ['TV', 'Newspaper'],
        'Radio + Newspaper': ['Radio', 'Newspaper'],
        'All Features': ['TV', 'Radio', 'Newspaper']
    }
    
    results = []
    fitted_models = {}
    
    for model_name, features in models.items():
        X_train_subset = X_train[features]
        X_test_subset = X_test[features]
        
        model = LinearRegression()
        model.fit(X_train_subset, y_train)
        
        y_train_pred = model.predict(X_train_subset)
        y_test_pred = model.predict(X_test_subset)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results.append({
            'Model': model_name,
            'Features': len(features),
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse
        })
        
        fitted_models[model_name] = (model, features)
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = range(len(results_df))
        ax.bar([x - 0.2 for x in x_pos], results_df['Train R²'], 0.4, label='Training R²', alpha=0.7)
        ax.bar([x + 0.2 for x in x_pos], results_df['Test R²'], 0.4, label='Test R²', alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([x - 0.2 for x in x_pos], results_df['Train RMSE'], 0.4, label='Training RMSE', alpha=0.7)
        ax.bar([x + 0.2 for x in x_pos], results_df['Test RMSE'], 0.4, label='Test RMSE', alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader("Best Model Analysis")
    best_model_name = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
    st.write(f"**Best performing model:** {best_model_name}")
    
    best_model, best_features = fitted_models[best_model_name]
    X_train_best = X_train[best_features]
    X_test_best = X_test[best_features]
    
    y_train_pred_best = best_model.predict(X_train_best)
    y_test_pred_best = best_model.predict(X_test_best)
    
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
        train_mae_best = mean_absolute_error(y_train, y_train_pred_best)
        test_mae_best = mean_absolute_error(y_test, y_test_pred_best)
        st.metric("Training MAE", f"{train_mae_best:.2f}")
        st.metric("Test MAE", f"{test_mae_best:.2f}")
    
    st.subheader("Model Coefficients")
    coefficients_df = pd.DataFrame({
        'Feature': best_features,
        'Coefficient': best_model.coef_,
        'Abs_Coefficient': np.abs(best_model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(coefficients_df, use_container_width=True)
        st.write(f"**Intercept:** {best_model.intercept_:.4f}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['green' if x > 0 else 'red' for x in coefficients_df['Coefficient']]
        ax.barh(range(len(coefficients_df)), coefficients_df['Coefficient'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(coefficients_df)))
        ax.set_yticklabels(coefficients_df['Feature'])
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Feature Coefficients')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.subheader("Interaction Effects Analysis")
    
    X_interaction = X.copy()
    X_interaction['TV_Radio'] = X['TV'] * X['Radio']
    X_interaction['TV_Newspaper'] = X['TV'] * X['Newspaper']
    X_interaction['Radio_Newspaper'] = X['Radio'] * X['Newspaper']
    
    X_train_int, X_test_int, _, _ = train_test_split(X_interaction, y, test_size=test_size, random_state=42)
    
    model_interaction = LinearRegression()
    model_interaction.fit(X_train_int, y_train)
    
    y_train_pred_int = model_interaction.predict(X_train_int)
    y_test_pred_int = model_interaction.predict(X_test_int)
    
    train_r2_int = r2_score(y_train, y_train_pred_int)
    test_r2_int = r2_score(y_test, y_test_pred_int)
    
    st.write("**Model with Interaction Terms:**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training R² (with interactions)", f"{train_r2_int:.3f}")
        st.metric("Test R² (with interactions)", f"{test_r2_int:.3f}")
        st.metric("R² Improvement", f"{test_r2_int - test_r2_best:.3f}")
    
    with col2:
        interaction_coef_df = pd.DataFrame({
            'Feature': X_interaction.columns,
            'Coefficient': model_interaction.coef_
        })
        st.dataframe(interaction_coef_df, use_container_width=True)
    
    st.subheader("Prediction vs Actual")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_train, y_train_pred_best, alpha=0.6, color='blue', label='Training')
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Sales')
        ax.set_ylabel('Predicted Sales')
        ax.set_title('Training Set: Predicted vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_test_pred_best, alpha=0.6, color='green', label='Test')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Sales')
        ax.set_ylabel('Predicted Sales')
        ax.set_title('Test Set: Predicted vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def main():
    run_advertising_analysis()

if __name__ == "__main__":
    main()
