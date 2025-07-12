import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Classification Analysis", layout="wide")
st.title("📊 Classification Analysis")
st.markdown("""
This app demonstrates classification methods including Logistic Regression, LDA, QDA, and KNN.
Analyze different classification algorithms and their performance characteristics.

**Based on Chapter 4: Classification from "An Introduction to Statistical Learning with Applications in Python"**
""")

@st.cache_data
def load_stock_market_data():
    np.random.seed(42)
    n_samples = 1250
    
    lag1 = np.random.normal(0, 1.2, n_samples)
    lag2 = np.random.normal(0, 1.2, n_samples)
    lag3 = np.random.normal(0, 1.2, n_samples)
    lag4 = np.random.normal(0, 1.2, n_samples)
    lag5 = np.random.normal(0, 1.2, n_samples)
    volume = np.random.uniform(0.3, 3.5, n_samples)
    
    direction_prob = 1 / (1 + np.exp(-(0.1 * lag1 + 0.05 * lag2 - 0.03 * volume + np.random.normal(0, 0.5, n_samples))))
    direction = np.where(direction_prob > 0.5, 'Up', 'Down')
    
    today = np.where(direction == 'Up', 
                    np.random.normal(0.5, 1.0, n_samples),
                    np.random.normal(-0.5, 1.0, n_samples))
    
    year = np.random.choice(range(2001, 2006), n_samples)
    
    df = pd.DataFrame({
        'Year': year,
        'Lag1': lag1,
        'Lag2': lag2,
        'Lag3': lag3,
        'Lag4': lag4,
        'Lag5': lag5,
        'Volume': volume,
        'Today': today,
        'Direction': direction
    })
    
    return df

@st.cache_data
def load_iris_data():
    np.random.seed(42)
    n_samples = 150
    
    setosa_sepal_length = np.random.normal(5.0, 0.35, 50)
    setosa_sepal_width = np.random.normal(3.4, 0.38, 50)
    setosa_petal_length = np.random.normal(1.5, 0.17, 50)
    setosa_petal_width = np.random.normal(0.25, 0.1, 50)
    
    versicolor_sepal_length = np.random.normal(5.9, 0.52, 50)
    versicolor_sepal_width = np.random.normal(2.8, 0.31, 50)
    versicolor_petal_length = np.random.normal(4.3, 0.47, 50)
    versicolor_petal_width = np.random.normal(1.3, 0.2, 50)
    
    virginica_sepal_length = np.random.normal(6.6, 0.64, 50)
    virginica_sepal_width = np.random.normal(3.0, 0.32, 50)
    virginica_petal_length = np.random.normal(5.6, 0.55, 50)
    virginica_petal_width = np.random.normal(2.0, 0.27, 50)
    
    df = pd.DataFrame({
        'sepal_length': np.concatenate([setosa_sepal_length, versicolor_sepal_length, virginica_sepal_length]),
        'sepal_width': np.concatenate([setosa_sepal_width, versicolor_sepal_width, virginica_sepal_width]),
        'petal_length': np.concatenate([setosa_petal_length, versicolor_petal_length, virginica_petal_length]),
        'petal_width': np.concatenate([setosa_petal_width, versicolor_petal_width, virginica_petal_width]),
        'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
    })
    
    return df

def run_stock_market_analysis():
    st.subheader("📈 Stock Market Direction Prediction")
    
    df = load_stock_market_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Samples: {len(df)}")
        st.write(f"Features: Lag1-Lag5, Volume")
        st.write(f"Target: Direction (Up/Down)")
        
        direction_counts = df['Direction'].value_counts()
        st.write(f"Up: {direction_counts.get('Up', 0)} ({direction_counts.get('Up', 0)/len(df)*100:.1f}%)")
        st.write(f"Down: {direction_counts.get('Down', 0)} ({direction_counts.get('Down', 0)/len(df)*100:.1f}%)")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        direction_counts.plot(kind='bar', ax=ax, color=['red', 'green'], alpha=0.7)
        ax.set_title('Distribution of Market Direction')
        ax.set_ylabel('Count')
        ax.set_xlabel('Direction')
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)
    
    st.subheader("Feature Analysis")
    feature_cols = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(feature_cols):
        for direction in ['Up', 'Down']:
            data = df[df['Direction'] == direction][col]
            axes[i].hist(data, alpha=0.6, label=direction, bins=20)
        axes[i].set_title(f'{col} by Direction')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Classification Models Comparison")
    
    X = df[feature_cols]
    y = df['Direction']
    
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05, key="stock_test_size")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        'K-Nearest Neighbors (k=3)': KNeighborsClassifier(n_neighbors=3),
        'K-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []
    fitted_models = {}
    
    for name, model in models.items():
        if 'Discriminant' in name or 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            fitted_models[name] = (model, X_test_scaled, y_pred_proba)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            fitted_models[name] = (model, X_test_scaled, y_pred_proba)
        
        accuracy = (y_pred == y_test).mean()
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        results.append({
            'Model': name,
            'Test Accuracy': accuracy,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std()
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, (model, X_test_data, y_pred_proba) in fitted_models.items():
        if y_pred_proba is not None:
            y_test_binary = (y_test == 'Up').astype(int)
            fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Stock Market Direction Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def run_iris_analysis():
    st.subheader("🌸 Iris Species Classification")
    
    df = load_iris_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"Samples: {len(df)}")
        st.write(f"Features: sepal/petal length/width")
        st.write(f"Classes: 3 species")
        
        species_counts = df['species'].value_counts()
        for species, count in species_counts.items():
            st.write(f"{species}: {count}")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        species_counts.plot(kind='bar', ax=ax, color=['purple', 'orange', 'green'], alpha=0.7)
        ax.set_title('Distribution of Iris Species')
        ax.set_ylabel('Count')
        ax.set_xlabel('Species')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    
    st.subheader("Feature Visualization")
    feature_pairs = [('sepal_length', 'sepal_width'), ('petal_length', 'petal_width')]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (x_col, y_col) in enumerate(feature_pairs):
        for species in df['species'].unique():
            species_data = df[df['species'] == species]
            axes[i].scatter(species_data[x_col], species_data[y_col], 
                          label=species, alpha=0.7, s=50)
        axes[i].set_xlabel(x_col.replace('_', ' ').title())
        axes[i].set_ylabel(y_col.replace('_', ' ').title())
        axes[i].set_title(f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Multi-class Classification")
    
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05, key="iris_test_size")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        'K-Nearest Neighbors (k=3)': KNeighborsClassifier(n_neighbors=3),
        'K-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = []
    
    for name, model in models.items():
        if 'Discriminant' in name or 'Logistic' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        accuracy = (y_pred == y_test).mean()
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        results.append({
            'Model': name,
            'Test Accuracy': accuracy,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std()
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    best_model_name = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
    best_model = models[str(best_model_name)]
    best_model.fit(X_train_scaled, y_train)
    y_pred_best = best_model.predict(X_test_scaled)
    
    st.subheader(f"Best Model: {best_model_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cm = confusion_matrix(y_test, y_pred_best)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=best_model.classes_, 
                   yticklabels=best_model.classes_, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    with col2:
        report = classification_report(y_test, y_pred_best, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

def main():
    st.header("📊 Classification Methods")
    
    dataset = st.selectbox(
        "Select Dataset",
        ["Stock Market Data", "Iris Dataset"]
    )
    
    if dataset == "Stock Market Data":
        run_stock_market_analysis()
    elif dataset == "Iris Dataset":
        run_iris_analysis()

if __name__ == "__main__":
    main()
