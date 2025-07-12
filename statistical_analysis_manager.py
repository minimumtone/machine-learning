import streamlit as st
import importlib.util
import sys
import os
import subprocess

st.set_page_config(
    page_title="統計学習解析プラットフォーム",
    page_icon="📊",
    layout="wide"
)

st.title("📊 統計学習解析プラットフォーム")
st.markdown("""
「統計学習入門 Python編」の解説事例をプログラム化した統合解析プラットフォームです。
各章の重要な概念と手法を実際のデータで学習できます。
""")

ANALYSIS_PROGRAMS = {
    "Boston住宅価格分析": {
        "file": "boston_housing_analysis.py",
        "description": "線形回帰による住宅価格予測（第3章）",
        "concepts": ["単回帰", "重回帰", "モデル評価", "残差分析"],
        "dataset": "Boston Housing Dataset (506サンプル, 13特徴量)"
    },
    "自動車燃費分析": {
        "file": "auto_mpg_analysis.py", 
        "description": "多項式回帰による非線形関係の分析（第3章）",
        "concepts": ["多項式特徴量", "非線形変換", "モデル比較"],
        "dataset": "Auto Dataset (392サンプル, mpg vs horsepower)"
    },
    "広告売上分析": {
        "file": "advertising_analysis.py",
        "description": "重回帰と交互作用効果の分析（第3章）", 
        "concepts": ["重回帰", "交互作用", "変数選択", "予測精度"],
        "dataset": "Advertising Dataset (TV, Radio, Newspaper vs Sales)"
    },
    "交差検証・ブートストラップ": {
        "file": "cross_validation_analysis.py",
        "description": "モデル選択と性能評価手法（第5章）",
        "concepts": ["交差検証", "ブートストラップ", "バイアス-バリアンス", "モデル選択"],
        "dataset": "複数データセットでの比較分析"
    },
    "分類分析": {
        "file": "classification_analysis.py", 
        "description": "ロジスティック回帰・LDA・QDA・KNN（第4章）",
        "concepts": ["ロジスティック回帰", "線形判別分析", "k近傍法", "ROC曲線"],
        "dataset": "Stock Market Dataset, その他分類データ"
    },
    "決定木・アンサンブル": {
        "file": "tree_methods_analysis.py",
        "description": "決定木・ランダムフォレスト・ブースティング（第8章）", 
        "concepts": ["決定木", "ランダムフォレスト", "ブースティング", "特徴量重要度"],
        "dataset": "Boston Dataset, 分類データセット"
    }
}

def display_program_overview():
    st.header("📚 解析プログラム一覧")
    
    for program_name, info in ANALYSIS_PROGRAMS.items():
        with st.expander(f"📖 {program_name}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**説明:** {info['description']}")
                st.write(f"**データセット:** {info['dataset']}")
            
            with col2:
                st.write("**学習概念:**")
                for concept in info['concepts']:
                    st.write(f"• {concept}")

def run_selected_program(program_name):
    st.header(f"🚀 {program_name}")
    
    program_info = ANALYSIS_PROGRAMS[program_name]
    program_file = program_info["file"]
    
    st.write(f"**説明:** {program_info['description']}")
    st.write(f"**データセット:** {program_info['dataset']}")
    
    st.write("**学習概念:**")
    for concept in program_info['concepts']:
        st.write(f"• {concept}")
    
    st.markdown("---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    program_path = os.path.join(current_dir, program_file)
    
    if os.path.exists(program_path):
        st.success(f"✅ プログラムファイルが見つかりました: {program_file}")
        st.info("このプログラムを実行するには、以下のコマンドを使用してください:")
        st.code(f"streamlit run {program_file}")
        
        st.write("**プログラムの内容:**")
        with open(program_path, 'r', encoding='utf-8') as f:
            content = f.read()
            st.code(content[:1000] + "..." if len(content) > 1000 else content, language='python')
    else:
        st.error(f"❌ プログラムファイルが見つかりません: {program_file}")

def main():
    st.sidebar.title("📊 ナビゲーション")
    
    page_options = ["プログラム一覧"] + list(ANALYSIS_PROGRAMS.keys())
    selected_page = st.sidebar.selectbox("ページを選択", page_options)
    
    if selected_page == "プログラム一覧":
        display_program_overview()
        
        st.header("🎯 クイックアクセス")
        st.write("各解析プログラムに直接アクセスできます:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏠 Boston住宅価格分析"):
                st.info("boston_housing_analysis.py を実行してください")
            if st.button("🔄 交差検証・ブートストラップ"):
                st.info("cross_validation_analysis.py を実行してください")
        
        with col2:
            if st.button("🚗 自動車燃費分析"):
                st.info("auto_mpg_analysis.py を実行してください")
            if st.button("📊 分類分析"):
                st.info("classification_analysis.py を実行してください")
        
        with col3:
            if st.button("📺 広告売上分析"):
                st.info("advertising_analysis.py を実行してください")
            if st.button("🌳 決定木・アンサンブル"):
                st.info("tree_methods_analysis.py を実行してください")
        
        st.header("📖 使用方法")
        st.write("""
        1. **プログラム選択**: 左サイドバーから学習したい解析手法を選択
        2. **データ探索**: 各プログラムでデータセットの概要を確認
        3. **パラメータ調整**: スライダーやセレクトボックスでパラメータを変更
        4. **結果分析**: グラフや表で結果を確認・比較
        5. **概念理解**: 各手法の理論的背景を学習
        """)
        
        st.header("📚 学習の流れ")
        st.write("""
        **推奨学習順序:**
        1. Boston住宅価格分析 (線形回帰の基礎)
        2. 自動車燃費分析 (非線形関係の理解)
        3. 広告売上分析 (重回帰と交互作用)
        4. 交差検証・ブートストラップ (モデル評価)
        5. 分類分析 (分類手法の比較)
        6. 決定木・アンサンブル (高度な手法)
        """)
        
    else:
        run_selected_program(selected_page)

if __name__ == "__main__":
    main()
