import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Boston Housing Analysis", layout="wide")
st.title("🏠 Boston Housing Data Analysis")
st.markdown("""
This app demonstrates linear regression analysis on the Boston Housing dataset.
Predict median house value (medv) using 13 predictors including crime rate, 
average number of rooms, and socioeconomic status.

**Based on Chapter 3: Linear Regression from "An Introduction to Statistical Learning with Applications in Python"**
""")

@st.cache_data
def load_boston_data():
    np.random.seed(42)
    n_samples = 506
    
    crim = np.random.exponential(3, n_samples)
    rm = np.random.normal(6.3, 0.7, n_samples)
    age = np.random.uniform(0, 100, n_samples)
    lstat = np.random.exponential(12, n_samples)
    
    features = {
        'crim': crim,
        'zn': np.random.exponential(11, n_samples),
        'indus': np.random.normal(11, 7, n_samples),
        'chas': np.random.binomial(1, 0.07, n_samples),
        'nox': np.random.normal(0.55, 0.12, n_samples),
        'rm': rm,
        'age': age,
        'dis': np.random.exponential(3.8, n_samples),
        'rad': np.random.choice(range(1, 25), n_samples),
        'tax': np.random.normal(408, 169, n_samples),
        'ptratio': np.random.normal(18.5, 2.2, n_samples),
        'b': np.random.normal(356, 91, n_samples),
        'lstat': lstat
    }
    
    medv = (50 - 0.5 * crim + 5 * rm - 0.1 * age - 0.5 * lstat + 
            np.random.normal(0, 5, n_samples))
    medv = np.clip(medv, 5, 50)
    
    df = pd.DataFrame(features)
    df['medv'] = medv
    return df

def run_boston_analysis():
    st.header("📊 Dataset Overview")
    df = load_boston_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Samples:** {len(df)}")
        st.write(f"**Features:** {len(df.columns) - 1}")
        st.write("**Target:** medv (median house value)")
        
    with col2:
        st.subheader("Target Variable Statistics")
        st.write(f"**Mean:** ${df['medv'].mean():.2f}k")
        st.write(f"**Std:** ${df['medv'].std():.2f}k")
        st.write(f"**Range:** ${df['medv'].min():.1f}k - ${df['medv'].max():.1f}k")
    
    st.subheader("Feature Descriptions")
    feature_descriptions = {
        'crim': 'per capita crime rate by town',
        'zn': 'proportion of residential land zoned for lots over 25,000 sq.ft.',
        'indus': 'proportion of non-retail business acres per town',
        'chas': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)',
        'nox': 'nitric oxides concentration (parts per 10 million)',
        'rm': 'average number of rooms per dwelling',
        'age': '% of owner-occupied units built prior to 1940',
        'dis': 'weighted distances to employment centres',
        'rad': 'index of accessibility to radial highways',
        'tax': 'full-value property-tax rate per $10,000',
        'ptratio': 'pupil-teacher ratio by town',
        'b': '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town',
        'lstat': '% lower status of the population'
    }
    
    desc_df = pd.DataFrame(list(feature_descriptions.items()), columns=['Feature', 'Description'])
    st.dataframe(desc_df, use_container_width=True)
    
    st.header("🔍 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['medv'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Median House Value ($1000s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Median House Values')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        correlation_with_target = df.corr()['medv'].sort_values(ascending=False)
        correlation_with_target = correlation_with_target[correlation_with_target.index != 'medv']
        
        colors = ['green' if x > 0 else 'red' for x in correlation_with_target.values]
        ax.barh(range(len(correlation_with_target)), list(correlation_with_target.values), color=colors, alpha=0.7)
        ax.set_yticks(range(len(correlation_with_target)))
        ax.set_yticklabels(correlation_with_target.index)
        ax.set_xlabel('Correlation with medv')
        ax.set_title('Feature Correlations with Target')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.header("📈 Linear Regression Analysis")
    
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05)
    
    X = df.drop('medv', axis=1)
    y = df['medv']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training R²", f"{train_r2:.3f}")
        st.metric("Test R²", f"{test_r2:.3f}")
    with col2:
        st.metric("Training RMSE", f"{train_rmse:.2f}")
        st.metric("Test RMSE", f"{test_rmse:.2f}")
    with col3:
        st.metric("Training MAE", f"{train_mae:.2f}")
        st.metric("Test MAE", f"{test_mae:.2f}")
    
    st.subheader("Feature Coefficients")
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in coefficients['Coefficient']]
    ax.barh(range(len(coefficients)), coefficients['Coefficient'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(coefficients)))
    ax.set_yticklabels(coefficients['Feature'])
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Linear Regression Coefficients')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.subheader("Prediction vs Actual")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training')
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Training Set: Predicted vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_test_pred, alpha=0.6, color='green', label='Test')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Test Set: Predicted vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.subheader("Residual Analysis")
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_train_pred, residuals_train, alpha=0.6, color='blue')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Training Set: Residual Plot')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test_pred, residuals_test, alpha=0.6, color='green')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Test Set: Residual Plot')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def main():
    run_boston_analysis()

if __name__ == "__main__":
    main()
