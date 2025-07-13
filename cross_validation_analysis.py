import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, validation_curve
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cross-Validation Analysis", layout="wide")
st.title("🔄 Cross-Validation and Bootstrap Analysis")
st.markdown("""
This app demonstrates model selection and performance evaluation techniques.
Explore validation set approach, k-fold cross-validation, and bootstrap methods.

**Based on Chapter 5: Resampling Methods from "An Introduction to Statistical Learning with Applications in Python"**
""")

@st.cache_data
def load_auto_data_cv():
    np.random.seed(42)
    n_samples = 392
    
    horsepower = np.random.uniform(46, 230, n_samples)
    mpg = 45 - 0.15 * horsepower + 0.0003 * horsepower**2 + np.random.normal(0, 3, n_samples)
    mpg = np.clip(mpg, 9, 47)
    
    df = pd.DataFrame({
        'horsepower': horsepower,
        'mpg': mpg
    })
    
    return df

def run_validation_set_approach():
    st.subheader("🎯 Validation Set Approach")
    st.write("Split data into training and validation sets to estimate test error.")
    
    df = load_auto_data_cv()
    X = df[['horsepower']]
    y = df['mpg']
    
    validation_size = st.slider("Validation Set Size", 0.2, 0.8, 0.5, 0.1, key="val_size")
    max_degree = st.slider("Maximum Polynomial Degree", 1, 10, 6, key="val_degree")
    n_trials = st.slider("Number of Random Splits", 1, 50, 10, key="val_trials")
    
    all_results = []
    
    for trial in range(n_trials):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=trial
        )
        
        trial_results = []
        for degree in range(1, max_degree + 1):
            poly_features = PolynomialFeatures(degree=degree)
            model = LinearRegression()
            
            pipeline = Pipeline([
                ('poly', poly_features),
                ('linear', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_val_pred = pipeline.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            
            trial_results.append({
                'Trial': trial,
                'Degree': degree,
                'Validation_MSE': val_mse
            })
        
        all_results.extend(trial_results)
    
    results_df = pd.DataFrame(all_results)
    
    mean_mse = results_df.groupby('Degree')['Validation_MSE'].mean()
    std_mse = results_df.groupby('Degree')['Validation_MSE'].std()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if n_trials > 1:
            for trial in range(min(10, n_trials)):
                trial_data = results_df[results_df['Trial'] == trial]
                ax.plot(trial_data['Degree'], trial_data['Validation_MSE'], 
                       alpha=0.3, color='gray', linewidth=1)
        
        ax.errorbar(list(mean_mse.index), list(mean_mse.values), yerr=list(std_mse.values), 
                   fmt='o-', color='red', linewidth=2, capsize=5, label='Mean ± Std')
        
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('Validation MSE')
        ax.set_title('Validation Set Approach')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        summary_df = pd.DataFrame({
            'Degree': mean_mse.index,
            'Mean_MSE': mean_mse.values,
            'Std_MSE': std_mse.values
        })
        st.dataframe(summary_df, use_container_width=True)
        
        best_degree = mean_mse.idxmin()
        st.write(f"**Best Degree:** {best_degree}")
        st.write(f"**Best MSE:** {mean_mse[best_degree]:.3f} ± {std_mse[best_degree]:.3f}")

def run_cross_validation():
    st.subheader("🔄 K-Fold Cross-Validation")
    st.write("Use k-fold CV to get more stable estimates of test error.")
    
    df = load_auto_data_cv()
    X = df[['horsepower']]
    y = df['mpg']
    
    k_folds = st.slider("Number of Folds (k)", 3, 20, 10, key="cv_folds")
    max_degree = st.slider("Maximum Polynomial Degree", 1, 10, 6, key="cv_degree")
    
    cv_results = []
    
    for degree in range(1, max_degree + 1):
        poly_features = PolynomialFeatures(degree=degree)
        model = LinearRegression()
        
        pipeline = Pipeline([
            ('poly', poly_features),
            ('linear', model)
        ])
        
        cv_scores = cross_val_score(pipeline, X, y, cv=k_folds, 
                                   scoring='neg_mean_squared_error')
        cv_mse = -cv_scores
        
        cv_results.append({
            'Degree': degree,
            'Mean_CV_MSE': cv_mse.mean(),
            'Std_CV_MSE': cv_mse.std(),
            'CV_Scores': cv_mse
        })
    
    cv_df = pd.DataFrame(cv_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(cv_df['Degree'], cv_df['Mean_CV_MSE'], yerr=cv_df['Std_CV_MSE'],
                   fmt='o-', color='blue', linewidth=2, capsize=5, label=f'{k_folds}-Fold CV')
        ax.set_xlabel('Polynomial Degree')
        ax.set_ylabel('Cross-Validation MSE')
        ax.set_title(f'{k_folds}-Fold Cross-Validation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        display_df = cv_df[['Degree', 'Mean_CV_MSE', 'Std_CV_MSE']].copy()
        st.dataframe(display_df, use_container_width=True)
        
        best_degree_cv = cv_df.loc[cv_df['Mean_CV_MSE'].idxmin(), 'Degree']
        best_mse_cv = cv_df.loc[cv_df['Mean_CV_MSE'].idxmin(), 'Mean_CV_MSE']
        best_std_cv = cv_df.loc[cv_df['Mean_CV_MSE'].idxmin(), 'Std_CV_MSE']
        
        st.write(f"**Best Degree:** {best_degree_cv}")
        st.write(f"**Best CV MSE:** {best_mse_cv:.3f} ± {best_std_cv:.3f}")

def run_bootstrap():
    st.subheader("🥾 Bootstrap Method")
    st.write("Use bootstrap resampling to estimate sampling distribution.")
    
    df = load_auto_data_cv()
    X = df[['horsepower']]
    y = df['mpg']
    
    n_bootstrap = st.slider("Number of Bootstrap Samples", 100, 2000, 1000, key="boot_samples")
    degree = st.slider("Polynomial Degree", 1, 5, 2, key="boot_degree")
    
    bootstrap_results = []
    bootstrap_coefs = []
    
    for i in range(n_bootstrap):
        X_boot, y_boot = resample(X, y, random_state=i)
        
        poly_features = PolynomialFeatures(degree=degree)
        model = LinearRegression()
        
        pipeline = Pipeline([
            ('poly', poly_features),
            ('linear', model)
        ])
        
        pipeline.fit(X_boot, y_boot)
        
        y_pred = pipeline.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        bootstrap_results.append({
            'Bootstrap_Sample': i,
            'MSE': mse,
            'R2': r2
        })
        
        coefficients = pipeline.named_steps['linear'].coef_
        intercept = pipeline.named_steps['linear'].intercept_
        bootstrap_coefs.append(np.concatenate([[intercept], coefficients]))
    
    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_coefs_array = np.array(bootstrap_coefs)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        axes[0].hist(bootstrap_df['MSE'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(bootstrap_df['MSE'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {bootstrap_df["MSE"].mean():.3f}')
        axes[0].set_xlabel('MSE')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Bootstrap Distribution of MSE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(bootstrap_df['R2'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].axvline(bootstrap_df['R2'].mean(), color='red', linestyle='--',
                       label=f'Mean: {bootstrap_df["R2"].mean():.3f}')
        axes[1].set_xlabel('R²')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Bootstrap Distribution of R²')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.write("**Bootstrap Statistics:**")
        
        mse_stats = {
            'Mean': bootstrap_df['MSE'].mean(),
            'Std': bootstrap_df['MSE'].std(),
            '95% CI Lower': np.percentile(bootstrap_df['MSE'], 2.5),
            '95% CI Upper': np.percentile(bootstrap_df['MSE'], 97.5)
        }
        
        r2_stats = {
            'Mean': bootstrap_df['R2'].mean(),
            'Std': bootstrap_df['R2'].std(),
            '95% CI Lower': np.percentile(bootstrap_df['R2'], 2.5),
            '95% CI Upper': np.percentile(bootstrap_df['R2'], 97.5)
        }
        
        stats_df = pd.DataFrame({
            'MSE': mse_stats,
            'R²': r2_stats
        })
        st.dataframe(stats_df, use_container_width=True)
        
        st.write("**Coefficient Statistics:**")
        coef_names = ['Intercept'] + [f'Coef_{i}' for i in range(bootstrap_coefs_array.shape[1]-1)]
        coef_stats = pd.DataFrame({
            'Coefficient': coef_names,
            'Mean': bootstrap_coefs_array.mean(axis=0),
            'Std': bootstrap_coefs_array.std(axis=0),
            '95% CI Lower': np.percentile(bootstrap_coefs_array, 2.5, axis=0),
            '95% CI Upper': np.percentile(bootstrap_coefs_array, 97.5, axis=0)
        })
        st.dataframe(coef_stats, use_container_width=True)

def main():
    st.header("📊 Dataset Overview")
    df = load_auto_data_cv()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Samples:** {len(df)}")
        st.write(f"**Features:** horsepower")
        st.write(f"**Target:** mpg")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['horsepower'], df['mpg'], alpha=0.6, color='blue')
        ax.set_xlabel('Horsepower')
        ax.set_ylabel('MPG')
        ax.set_title('Auto Dataset')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.header("🔍 Resampling Methods")
    
    method = st.selectbox(
        "Select Resampling Method",
        ["Validation Set Approach", "K-Fold Cross-Validation", "Bootstrap"]
    )
    
    if method == "Validation Set Approach":
        run_validation_set_approach()
    elif method == "K-Fold Cross-Validation":
        run_cross_validation()
    elif method == "Bootstrap":
        run_bootstrap()

if __name__ == "__main__":
    main()
