import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Tree-Based Methods", layout="wide")
st.title("🌳 Tree-Based Methods Analysis")
st.markdown("""
This app demonstrates tree-based methods including Decision Trees, Random Forests, and Boosting.
Explore different ensemble methods and their performance characteristics.

**Based on Chapter 8: Tree-Based Methods from "An Introduction to Statistical Learning with Applications in Python"**
""")

@st.cache_data
def load_boston_data_trees():
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

@st.cache_data
def load_heart_data():
    np.random.seed(42)
    n_samples = 303
    
    age = np.random.uniform(29, 77, n_samples)
    chol = np.random.normal(246, 51, n_samples)
    thalach = np.random.normal(150, 22, n_samples)
    oldpeak = np.random.exponential(1, n_samples)
    
    heart_disease_prob = 1 / (1 + np.exp(-(0.05 * age + 0.002 * chol - 0.01 * thalach + 0.5 * oldpeak - 5)))
    heart_disease = np.random.binomial(1, heart_disease_prob, n_samples)
    
    df = pd.DataFrame({
        'age': age,
        'sex': np.random.binomial(1, 0.68, n_samples),
        'cp': np.random.choice([0, 1, 2, 3], n_samples),
        'trestbps': np.random.normal(131, 17, n_samples),
        'chol': chol,
        'fbs': np.random.binomial(1, 0.15, n_samples),
        'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.4, 0.1]),
        'thalach': thalach,
        'exang': np.random.binomial(1, 0.33, n_samples),
        'oldpeak': oldpeak,
        'slope': np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.5, 0.3]),
        'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.05, 0.2, 0.65, 0.1]),
        'target': heart_disease
    })
    
    return df

def run_regression_trees():
    st.subheader("🏠 Regression Trees - Boston Housing")
    
    df = load_boston_data_trees()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Samples: {len(df)}")
        st.write(f"Features: {len(df.columns) - 1}")
        st.write(f"Target: medv (median house value)")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['medv'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Median House Value ($1000s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of House Values')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    X = df.drop('medv', axis=1)
    y = df['medv']
    
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05, key="reg_test_size")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.subheader("Model Comparison")
    
    models = {
        'Decision Tree (Unpruned)': DecisionTreeRegressor(random_state=42),
        'Decision Tree (Max Depth 5)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Decision Tree (Max Depth 3)': DecisionTreeRegressor(max_depth=3, random_state=42),
        'Random Forest (50 trees)': RandomForestRegressor(n_estimators=50, random_state=42),
        'Random Forest (100 trees)': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = []
    fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results.append({
            'Model': name,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std()
        })
        
        fitted_models[name] = model
    
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
        best_model_name = results_df.loc[results_df['Test R²'].idxmax(), 'Model']
        best_model = fitted_models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(len(feature_importance)), feature_importance['Importance'], alpha=0.7)
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['Feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Feature Importance - {best_model_name}')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

def run_classification_trees():
    st.subheader("❤️ Classification Trees - Heart Disease")
    
    df = load_heart_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Samples: {len(df)}")
        st.write(f"Features: {len(df.columns) - 1}")
        st.write(f"Target: heart disease (0/1)")
        
        target_counts = df['target'].value_counts()
        st.write(f"No Disease: {target_counts.get(0, 0)}")
        st.write(f"Disease: {target_counts.get(1, 0)}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        target_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Distribution of Heart Disease')
        ax.set_ylabel('Count')
        ax.set_xlabel('Heart Disease')
        ax.set_xticklabels(['No Disease', 'Disease'], rotation=0)
        st.pyplot(fig)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05, key="clf_test_size")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    st.subheader("Classification Model Comparison")
    
    models = {
        'Decision Tree (Unpruned)': DecisionTreeClassifier(random_state=42),
        'Decision Tree (Max Depth 5)': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Decision Tree (Max Depth 3)': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest (50 trees)': RandomForestClassifier(n_estimators=50, random_state=42),
        'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'CV Accuracy Mean': cv_scores.mean(),
            'CV Accuracy Std': cv_scores.std()
        })
        
        fitted_models[name] = model
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = range(len(results_df))
        ax.bar([x - 0.2 for x in x_pos], results_df['Train Accuracy'], 0.4, label='Training Accuracy', alpha=0.7)
        ax.bar([x + 0.2 for x in x_pos], results_df['Test Accuracy'], 0.4, label='Test Accuracy', alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Classification Model Performance')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        best_model_name = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
        best_model = fitted_models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(range(len(feature_importance)), feature_importance['Importance'], alpha=0.7)
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['Feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Feature Importance - {best_model_name}')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

def main():
    st.header("🌳 Tree-Based Methods")
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Regression Trees", "Classification Trees"]
    )
    
    if analysis_type == "Regression Trees":
        run_regression_trees()
    elif analysis_type == "Classification Trees":
        run_classification_trees()

if __name__ == "__main__":
    main()
