import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['IPAGothic', 'IPAPGothic']
plt.rcParams['font.sans-serif'] = ['IPAGothic', 'IPAPGothic']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="機械学習モデル評価教育アプリ",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 機械学習モデル評価教育アプリ")
st.markdown("""
このアプリは学生が**良いモデル**と**悪いモデル**を見分ける力を身につけるための教育ツールです。
異なるモデルの性能を比較し、**汎化性能**や**過学習**について学習できます。



材料工学分野では、以下のような予測問題でモデル評価が重要です：
- **熱伝導率予測**: 温度、圧力、処理時間から材料の熱伝導率を予測
- **電気伝導率予測**: 組成比、アニール温度から導電性を予測  
- **機械的強度予測**: 結晶粒径、熱処理条件から強度を予測

**特に注目すべきポイント:**
- ハイパーパラメータの設定ミスによる予測問題（単一値化、階段状予測）
- 過学習の検出と対策
- クロスバリデーションによる性能評価
""")

st.sidebar.header("📊 データセット選択")
dataset_option = st.sidebar.selectbox(
    "使用するデータセット:",
    ["熱伝導率データ", "電気伝導率データ", "機械的強度データ", "ガソリン消費量データ", "合成データ（線形）", "合成データ（非線形）", "合成データ（ノイズ多）"]
)

def generate_synthetic_data(data_type, n_samples=100, noise_level=0.1, random_state=42):
    np.random.seed(random_state)
    
    if data_type == "linear":
        X = np.random.uniform(-2, 2, (n_samples, 2))
        y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, noise_level, n_samples)
        feature_names = ['特徴量1', '特徴量2']
        
    elif data_type == "nonlinear":
        X = np.random.uniform(-2, 2, (n_samples, 2))
        y = X[:, 0]**2 + X[:, 1]**2 + 0.5 * X[:, 0] * X[:, 1] + np.random.normal(0, noise_level, n_samples)
        feature_names = ['特徴量1', '特徴量2']
        
    elif data_type == "noisy":
        X = np.random.uniform(-2, 2, (n_samples, 3))
        y = 2 * X[:, 0] + np.random.normal(0, noise_level * 5, n_samples)  # High noise
        feature_names = ['特徴量1', '特徴量2', '特徴量3']
    
    df = pd.DataFrame(X, columns=feature_names)
    df['目標変数'] = y
    return df

def generate_materials_data(data_type, n_samples=200, random_state=42):
    np.random.seed(random_state)
    
    if data_type == "thermal_conductivity":
        temperature = np.random.uniform(800, 1400, n_samples)
        pressure = np.random.uniform(1.0, 5.0, n_samples) 
        time = np.random.uniform(1.0, 10.0, n_samples)
        
        thermal_conductivity = (
            50 + 0.1 * temperature + 20 * pressure + 5 * time +
            np.random.normal(0, 10, n_samples)
        )
        
        df = pd.DataFrame({
            '温度_K': temperature,
            '圧力_GPa': pressure, 
            '処理時間_h': time,
            '熱伝導率_W_per_mK': thermal_conductivity
        })
        return df, '熱伝導率_W_per_mK'
        
    elif data_type == "electrical_conductivity":
        composition = np.random.uniform(0.1, 0.9, n_samples)
        annealing_temp = np.random.uniform(600, 1200, n_samples)
        grain_size = np.random.uniform(1, 50, n_samples)
        
        electrical_conductivity = (
            1000 * composition + 0.5 * annealing_temp + 10 * grain_size +
            np.random.normal(0, 100, n_samples)
        )
        
        df = pd.DataFrame({
            '組成比': composition,
            'アニール温度_K': annealing_temp,
            '結晶粒径_μm': grain_size,
            '電気伝導率_S_per_m': electrical_conductivity
        })
        return df, '電気伝導率_S_per_m'
        
    elif data_type == "mechanical_strength":
        grain_size = np.random.uniform(1, 50, n_samples)
        treatment_time = np.random.uniform(0.5, 20, n_samples)
        treatment_temp = np.random.uniform(800, 1200, n_samples)
        
        strength = (
            500 - 5 * grain_size + 10 * treatment_time + 0.2 * treatment_temp +
            np.random.normal(0, 50, n_samples)
        )
        
        df = pd.DataFrame({
            '結晶粒径_μm': grain_size,
            '熱処理時間_h': treatment_time,
            '熱処理温度_K': treatment_temp,
            '機械的強度_MPa': strength
        })
        return df, '機械的強度_MPa'

def generate_materials_data(data_type, n_samples=200, random_state=42):
    np.random.seed(random_state)
    
    if data_type == "thermal_conductivity":
        temperature = np.random.uniform(800, 1400, n_samples)
        pressure = np.random.uniform(1.0, 5.0, n_samples) 
        time = np.random.uniform(1.0, 10.0, n_samples)
        
        thermal_conductivity = (
            50 + 0.1 * temperature + 20 * pressure + 5 * time +
            np.random.normal(0, 10, n_samples)
        )
        
        df = pd.DataFrame({
            '温度_K': temperature,
            '圧力_GPa': pressure, 
            '処理時間_h': time,
            '熱伝導率_W_per_mK': thermal_conductivity
        })
        return df, '熱伝導率_W_per_mK'
        
    elif data_type == "electrical_conductivity":
        composition = np.random.uniform(0.1, 0.9, n_samples)
        annealing_temp = np.random.uniform(600, 1200, n_samples)
        grain_size = np.random.uniform(1, 50, n_samples)
        
        electrical_conductivity = (
            1000 * composition + 0.5 * annealing_temp + 10 * grain_size +
            np.random.normal(0, 100, n_samples)
        )
        
        df = pd.DataFrame({
            '組成比': composition,
            'アニール温度_K': annealing_temp,
            '結晶粒径_μm': grain_size,
            '電気伝導率_S_per_m': electrical_conductivity
        })
        return df, '電気伝導率_S_per_m'
        
    elif data_type == "mechanical_strength":
        grain_size = np.random.uniform(1, 50, n_samples)
        treatment_time = np.random.uniform(0.5, 20, n_samples)
        treatment_temp = np.random.uniform(800, 1200, n_samples)
        
        strength = (
            500 - 5 * grain_size + 10 * treatment_time + 0.2 * treatment_temp +
            np.random.normal(0, 50, n_samples)
        )
        
        df = pd.DataFrame({
            '結晶粒径_μm': grain_size,
            '熱処理時間_h': treatment_time,
            '熱処理温度_K': treatment_temp,
            '機械的強度_MPa': strength
        })
        return df, '機械的強度_MPa'

@st.cache_data
def load_data(dataset_option):
    if dataset_option == "熱伝導率データ":
        return generate_materials_data("thermal_conductivity")
    elif dataset_option == "電気伝導率データ":
        return generate_materials_data("electrical_conductivity") 
    elif dataset_option == "機械的強度データ":
        return generate_materials_data("mechanical_strength")
    elif dataset_option == "ガソリン消費量データ":
        try:
            df = pd.read_csv("petrol_consumption.csv")
            df.columns = ['ガソリン税', '平均収入', '舗装道路', '運転免許率', 'ガソリン消費量']
            return df, 'ガソリン消費量'
        except:
            st.error("ガソリン消費量データが見つかりません。合成データを使用します。")
            df = generate_synthetic_data("linear")
            return df, '目標変数'
    elif dataset_option == "合成データ（線形）":
        df = generate_synthetic_data("linear")
        return df, '目標変数'
    elif dataset_option == "合成データ（非線形）":
        df = generate_synthetic_data("nonlinear")
        return df, '目標変数'
    elif dataset_option == "合成データ（ノイズ多）":
        df = generate_synthetic_data("noisy")
        return df, '目標変数'

df, target_col = load_data(dataset_option)
feature_cols = [col for col in df.columns if col != target_col]

st.header("📈 データ概要")
col1, col2 = st.columns(2)

def display_aggrid_table(df, title="データテーブル", height=400):
    st.subheader(title)
    
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children")
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gridOptions = gb.build()
    
    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        theme='streamlit',
        enable_enterprise_modules=False,
        height=height,
        width='100%',
        reload_data=True
    )
    
    return grid_response

with col1:
    display_aggrid_table(df.head(10), "データサンプル", height=300)

with col2:
    display_aggrid_table(df.describe(), "統計情報", height=300)

st.header("🤖 モデル設定")

col1, col2, col3 = st.columns(3)

with col1:
    test_size = st.slider("テストデータ割合", 0.1, 0.5, 0.3, 0.05)

with col2:
    random_state = st.number_input("ランダムシード", 0, 100, 42)

with col3:
    cv_folds = st.slider("クロスバリデーション分割数", 3, 10, 5)

X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

models = {
    "線形回帰（適切）": {
        "model": LinearRegression(),
        "description": "シンプルで解釈しやすいモデル。線形関係に適している。",
        "expected": "good" if dataset_option in ["ガソリン消費量データ", "合成データ（線形）"] else "fair"
    },
    "多項式回帰（2次）": {
        "model": Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler()),
            ('reg', LinearRegression())
        ]),
        "description": "2次の特徴量を追加。非線形関係をある程度捉えられる。",
        "expected": "good" if dataset_option == "合成データ（非線形）" else "fair"
    },
    "多項式回帰（高次・過学習）": {
        "model": Pipeline([
            ('poly', PolynomialFeatures(degree=min(8, len(X_train)-1))),
            ('scaler', StandardScaler()),
            ('reg', LinearRegression())
        ]),
        "description": "高次の多項式。訓練データに過度に適合する可能性が高い（過学習）。",
        "expected": "bad"
    },
    "Ridge回帰（過度な正則化）": {
        "model": Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('scaler', StandardScaler()),
            ('reg', Ridge(alpha=1000.0))  # 極端に大きなalpha
        ]),
        "description": "過度な正則化により予測値が平均値に収束（単一値化）。",
        "expected": "bad"
    },
    "Ridge回帰（適切な正則化）": {
        "model": Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('scaler', StandardScaler()),
            ('reg', Ridge(alpha=1.0))
        ]),
        "description": "適切な正則化により過学習を抑制。汎化性能が向上する可能性。",
        "expected": "good"
    },
    "ランダムフォレスト（適切）": {
        "model": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_state),
        "description": "アンサンブル手法。非線形関係を捉えやすく、過学習に比較的強い。",
        "expected": "good"
    },
    "ランダムフォレスト（階段状予測）": {
        "model": RandomForestRegressor(n_estimators=5, max_depth=2, min_samples_split=20, random_state=random_state),
        "description": "浅い木と少ない木数により階段状の予測パターンが発生。",
        "expected": "bad"
    },
    "ランダムフォレスト（過学習）": {
        "model": RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=random_state),
        "description": "制約の少ないランダムフォレスト。過学習しやすい設定。",
        "expected": "bad"
    }
}

if st.button("🚀 モデル訓練・評価を実行", type="primary"):
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (model_name, model_info) in enumerate(models.items()):
        status_text.text(f"訓練中: {model_name}")
        
        model = model_info["model"]
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        overfitting_score = train_r2 - test_r2
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        pred_variance = np.var(y_test_pred)
        pred_unique_ratio = len(np.unique(np.round(y_test_pred, 2))) / len(y_test_pred)
        
        is_single_value = pred_variance < 0.01 * np.var(y_test)
        
        is_step_like = pred_unique_ratio < 0.1
        
        results.append({
            "モデル": model_name,
            "説明": model_info["description"],
            "期待結果": model_info["expected"],
            "訓練R²": train_r2,
            "テストR²": test_r2,
            "CV平均R²": cv_mean,
            "CV標準偏差": cv_std,
            "訓練RMSE": train_rmse,
            "テストRMSE": test_rmse,
            "訓練MAE": train_mae,
            "テストMAE": test_mae,
            "過学習指標": overfitting_score,
            "予測分散": pred_variance,
            "予測多様性": pred_unique_ratio,
            "単一値化": is_single_value,
            "階段状": is_step_like,
            "テスト予測値": y_test_pred,
            "テスト実測値": y_test
        })
        
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("評価完了！")
    
    results_df = pd.DataFrame(results)
    
    st.header("📊 モデル評価結果")
    
    st.subheader("🔍 予測パターン分析")
    
    problem_models = results_df[
        (results_df["単一値化"] == True) | 
        (results_df["階段状"] == True) | 
        (results_df["過学習指標"] > 0.2)
    ]
    
    if len(problem_models) > 0:
        st.warning(f"⚠️ {len(problem_models)}個のモデルで予測問題が検出されました")
        
        for _, model_row in problem_models.iterrows():
            problems = []
            if model_row["単一値化"]:
                problems.append("**単一値化**: 予測値がほぼ同じ値になっている")
            if model_row["階段状"]:
                problems.append("**階段状**: 予測値が不連続で階段状になっている")
            if model_row["過学習指標"] > 0.2:
                problems.append("**過学習**: 訓練データに過度に適合している")
            
            st.error(f"**{model_row['モデル']}**: {', '.join(problems)}")
    
    st.subheader("性能比較表")
    display_df = results_df[["モデル", "訓練R²", "テストR²", "CV平均R²", "過学習指標", "予測多様性"]].round(4)
    
    display_df["予測問題"] = ""
    for idx, row in results_df.iterrows():
        problems = []
        if row["単一値化"]:
            problems.append("単一値化")
        if row["階段状"]:
            problems.append("階段状")
        if row["過学習指標"] > 0.2:
            problems.append("過学習")
        display_df.loc[idx, "予測問題"] = ", ".join(problems) if problems else "なし"
    
    def color_overfitting(val):
        if val > 0.2:
            return 'background-color: #ffcccc'  # Red for high overfitting
        elif val > 0.1:
            return 'background-color: #ffffcc'  # Yellow for moderate overfitting
        else:
            return 'background-color: #ccffcc'  # Green for low overfitting
    
    def color_diversity(val):
        if val < 0.1:
            return 'background-color: #ffcccc'  # Red for low diversity
        elif val < 0.3:
            return 'background-color: #ffffcc'  # Yellow for moderate diversity
        else:
            return 'background-color: #ccffcc'  # Green for high diversity
    
    display_aggrid_table(display_df, "モデル評価結果", height=300)
    
    st.subheader("📚 評価指標の解説")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **R²（決定係数）**
        - 1に近いほど良い予測性能
        - 0.8以上：優秀
        - 0.6-0.8：良好
        - 0.6未満：改善が必要
        
        **RMSE（平均平方根誤差）**
        - 小さいほど良い
        - 目標変数の単位と同じ
        """)
    
    with col2:
        st.markdown("""
        **過学習指標（訓練R² - テストR²）**
        - 0.1未満：健全
        - 0.1-0.2：注意
        - 0.2以上：過学習の可能性大
        
        **CV（クロスバリデーション）**
        - 訓練データでの汎化性能推定
        - 標準偏差が小さいほど安定
        """)
    
    st.subheader("📈 性能可視化")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    axes[0, 0].bar(range(len(results_df)), results_df["訓練R²"], alpha=0.7, label="訓練R²", color='blue')
    axes[0, 0].bar(range(len(results_df)), results_df["テストR²"], alpha=0.7, label="テストR²", color='red')
    axes[0, 0].set_title("R² Score Comparison")
    axes[0, 0].set_xlabel("Model")
    axes[0, 0].set_ylabel("R²")
    axes[0, 0].set_xticks(range(len(results_df)))
    axes[0, 0].set_xticklabels([name.split("（")[0] for name in results_df["モデル"]], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(range(len(results_df)), results_df["訓練RMSE"], alpha=0.7, label="訓練RMSE", color='blue')
    axes[0, 1].bar(range(len(results_df)), results_df["テストRMSE"], alpha=0.7, label="テストRMSE", color='red')
    axes[0, 1].set_title("RMSE Comparison")
    axes[0, 1].set_xlabel("Model")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_xticks(range(len(results_df)))
    axes[0, 1].set_xticklabels([name.split("（")[0] for name in results_df["モデル"]], rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    colors = ['red' if x > 0.2 else 'orange' if x > 0.1 else 'green' for x in results_df["過学習指標"]]
    axes[1, 0].bar(range(len(results_df)), results_df["過学習指標"], color=colors, alpha=0.7)
    axes[1, 0].set_title("Overfitting Index (Train R² - Test R²)")
    axes[1, 0].set_xlabel("Model")
    axes[1, 0].set_ylabel("Overfitting Index")
    axes[1, 0].set_xticks(range(len(results_df)))
    axes[1, 0].set_xticklabels([name.split("（")[0] for name in results_df["モデル"]], rotation=45)
    axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='注意ライン')
    axes[1, 0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='危険ライン')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].errorbar(range(len(results_df)), results_df["CV平均R²"], 
                       yerr=results_df["CV標準偏差"], fmt='o', capsize=5, capthick=2)
    axes[1, 1].set_title("Cross Validation Results")
    axes[1, 1].set_xlabel("Model")
    axes[1, 1].set_ylabel("CV Mean R² ± Std Dev")
    axes[1, 1].set_xticks(range(len(results_df)))
    axes[1, 1].set_xticklabels([name.split("（")[0] for name in results_df["モデル"]], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    colors_div = ['red' if x < 0.1 else 'orange' if x < 0.3 else 'green' for x in results_df["予測多様性"]]
    axes[2, 0].bar(range(len(results_df)), results_df["予測多様性"], color=colors_div, alpha=0.7)
    axes[2, 0].set_title("Prediction Diversity (Single Value Detection)")
    axes[2, 0].set_xlabel("Model")
    axes[2, 0].set_ylabel("Prediction Diversity")
    axes[2, 0].set_xticks(range(len(results_df)))
    axes[2, 0].set_xticklabels([name.split("（")[0] for name in results_df["モデル"]], rotation=45)
    axes[2, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='単一値化ライン')
    axes[2, 0].axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='注意ライン')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    results_df['総合スコア'] = results_df['テストR²'] - 0.5 * results_df['過学習指標']
    
    worst_model_idx = results_df["総合スコア"].idxmin()
    worst_model = results_df.loc[worst_model_idx]
    axes[2, 1].scatter(worst_model["テスト実測値"], worst_model["テスト予測値"], alpha=0.6, color='red')
    axes[2, 1].plot([worst_model["テスト実測値"].min(), worst_model["テスト実測値"].max()], 
                   [worst_model["テスト実測値"].min(), worst_model["テスト実測値"].max()], 
                   'k--', alpha=0.8, label='理想線')
    axes[2, 1].set_title(f"Worst Model Prediction vs Actual\n({worst_model['モデル']})")
    axes[2, 1].set_xlabel("Actual Values")
    axes[2, 1].set_ylabel("Predicted Values")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("🏆 モデル評価とランキング")
    
    ranked_df = results_df.sort_values('総合スコア', ascending=False)
    
    for i, (_, row) in enumerate(ranked_df.iterrows()):
        rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        rank_emoji = rank_emojis[i] if i < len(rank_emojis) else f"{i+1}️⃣"
        
        problems = []
        if row['単一値化']:
            problems.append("単一値化")
        if row['階段状']:
            problems.append("階段状")
        if row['過学習指標'] > 0.2:
            problems.append("過学習")
        
        if problems:
            quality = f"❌ 悪いモデル（{', '.join(problems)}）"
            quality_color = "red"
        elif row['過学習指標'] > 0.1:
            quality = "⚠️ 注意が必要"
            quality_color = "orange"
        elif row['テストR²'] > 0.7:
            quality = "✅ 良いモデル"
            quality_color = "green"
        elif row['テストR²'] > 0.5:
            quality = "🔶 普通のモデル"
            quality_color = "blue"
        else:
            quality = "❌ 悪いモデル（性能不足）"
            quality_color = "red"
        
        with st.expander(f"{rank_emoji} {row['モデル']} - {quality}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**説明:** {row['説明']}")
                st.markdown(f"**テストR²:** {row['テストR²']:.4f}")
                st.markdown(f"**過学習指標:** {row['過学習指標']:.4f}")
                
            with col2:
                st.markdown(f"**CV平均R²:** {row['CV平均R²']:.4f} ± {row['CV標準偏差']:.4f}")
                st.markdown(f"**総合スコア:** {row['総合スコア']:.4f}")
                
                if row['単一値化']:
                    st.markdown("**問題:** 予測値が単一値に収束 → 正則化パラメータを下げる")
                elif row['階段状']:
                    st.markdown("**問題:** 階段状の予測 → 木の深さを増やすか、木の数を増やす")
                elif row['過学習指標'] > 0.2:
                    st.markdown("**推奨:** 正則化を強化するか、モデルの複雑さを下げる")
                elif row['過学習指標'] > 0.1:
                    st.markdown("**推奨:** クロスバリデーションでさらに検証")
                elif row['テストR²'] > 0.7:
                    st.markdown("**推奨:** このモデルは実用に適している")
                else:
                    st.markdown("**推奨:** 特徴量エンジニアリングやモデル選択を見直す")
    
    st.subheader("🎯 学習のまとめ")
    
    st.markdown("""
    1. **訓練データとテストデータの性能差が小さい**（過学習していない）
    2. **テストデータでの性能が高い**（汎化性能が良い）
    3. **クロスバリデーションの結果が安定している**（標準偏差が小さい）
    
    1. **訓練データの性能 >> テストデータの性能**（過学習）
    2. **テストデータでの性能が低い**（汎化性能が悪い）
    3. **クロスバリデーションの結果が不安定**（標準偏差が大きい）
    4. **予測値が単一値に収束**（過度な正則化）
    5. **予測値が階段状**（決定木の設定不良）
    
    - **Ridge回帰のalpha値が大きすぎる** → 予測値が平均値に収束（単一値化）
    - **ランダムフォレストの木が浅すぎる** → 階段状の予測パターン
    - **正則化が強すぎる** → モデルが学習できない（underfitting）
    - **正則化が弱すぎる** → 過学習（overfitting）
    
    - **過学習対策:** 正則化、早期停止、データ増強
    - **単一値化対策:** 正則化パラメータを下げる、特徴量を増やす
    - **階段状対策:** 木の深さや数を増やす、連続値予測手法を使う
    - **性能向上:** 特徴量エンジニアリング、モデル選択、ハイパーパラメータ調整
    - **評価方法:** 必ずホールドアウト検証とクロスバリデーションを併用
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("📖 学習リソース")
st.sidebar.markdown("""
**重要な概念:**
- 過学習（Overfitting）
- 汎化性能（Generalization）
- バイアス-バリアンストレードオフ
- クロスバリデーション
- 正則化（Regularization）

**評価指標:**
- R²（決定係数）
- RMSE（平均平方根誤差）
- MAE（平均絶対誤差）
""")
