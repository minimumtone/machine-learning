import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import json

from pdf_to_markdown import PDFToMarkdownConverter
from chunking_strategies import MarkdownChunker, MeCabSemanticChunker, SelfRouteChunker, Chunk
from rag_system import RAGSystem
from evaluation import ChunkingEvaluator

st.set_page_config(
    page_title="RAG Chunking Comparison",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG チャンク作成手法の比較")
st.markdown("""
このアプリケーションは、PDFからテキストを抽出してRAGシステムに使用する際の、
異なるチャンク作成手法を比較します。

**実装された手法:**
1. **Markdown-based**: 標準的なMarkdownヘッダーベースのチャンク分割
2. **MeCab Semantic**: MeCabを使用した意味的な単位でのチャンク分割
3. **Self-Route**: セマンティック類似度を用いた知的なチャンク境界検出
""")

@st.cache_data
def load_and_convert_pdf(pdf_path: str):
    converter = PDFToMarkdownConverter(pdf_path)
    info = converter.get_pdf_info()
    md_text = converter.convert_to_markdown()
    return md_text, info

@st.cache_data
def create_chunks_all_methods(md_text: str, chunk_size: int, chunk_overlap: int, similarity_threshold: float):
    markdown_chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    mecab_chunker = MeCabSemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    self_route_chunker = SelfRouteChunker(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold
    )
    
    with st.spinner("Markdown-based チャンク作成中..."):
        markdown_chunks = markdown_chunker.chunk(md_text)
    
    with st.spinner("MeCab Semantic チャンク作成中..."):
        mecab_chunks = mecab_chunker.chunk(md_text)
    
    with st.spinner("Self-Route チャンク作成中..."):
        self_route_chunks = self_route_chunker.chunk(md_text)
    
    return {
        'markdown': markdown_chunks,
        'mecab_semantic': mecab_chunks,
        'self_route': self_route_chunks
    }

def main():
    st.sidebar.header("⚙️ 設定")
    
    pdf_path = st.sidebar.text_input(
        "PDFファイルパス",
        value="/home/ubuntu/attachments/30af5be2-cb9a-4976-9c52-fcbb02d7b303/CRDS-FY2024-FR-09.pdf"
    )
    
    chunk_size = st.sidebar.slider("チャンクサイズ", 500, 2000, 1000, 100)
    chunk_overlap = st.sidebar.slider("チャンクオーバーラップ", 0, 500, 200, 50)
    similarity_threshold = st.sidebar.slider("Self-Route 類似度閾値", 0.0, 1.0, 0.7, 0.05)
    
    if not Path(pdf_path).exists():
        st.error(f"PDFファイルが見つかりません: {pdf_path}")
        return
    
    with st.spinner("PDFを読み込み中..."):
        md_text, pdf_info = load_and_convert_pdf(pdf_path)
    
    st.sidebar.success("✅ PDF読み込み完了")
    st.sidebar.write(f"**ページ数:** {pdf_info['num_pages']}")
    st.sidebar.write(f"**ファイルサイズ:** {pdf_info['file_size_mb']:.2f} MB")
    st.sidebar.write(f"**Markdown文字数:** {len(md_text):,}")
    
    tabs = st.tabs([
        "📊 チャンク統計比較", 
        "🔍 RAG検索テスト", 
        "📝 チャンク詳細", 
        "📈 評価メトリクス"
    ])
    
    with st.spinner("全手法でチャンク作成中..."):
        chunks_by_method = create_chunks_all_methods(
            md_text, chunk_size, chunk_overlap, similarity_threshold
        )
    
    with tabs[0]:
        st.header("チャンク統計比較")
        
        stats_data = []
        for method, chunks in chunks_by_method.items():
            chunk_lengths = [len(c.content) for c in chunks]
            stats_data.append({
                '手法': method,
                'チャンク数': len(chunks),
                '平均長': f"{sum(chunk_lengths) / len(chunk_lengths):.0f}",
                '最小長': min(chunk_lengths),
                '最大長': max(chunk_lengths),
                '標準偏差': f"{pd.Series(chunk_lengths).std():.0f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                stats_df, 
                x='手法', 
                y='チャンク数',
                title="手法別チャンク数",
                color='手法'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            length_data = []
            for method, chunks in chunks_by_method.items():
                for chunk in chunks:
                    length_data.append({
                        '手法': method,
                        'チャンク長': len(chunk.content)
                    })
            
            length_df = pd.DataFrame(length_data)
            fig = px.box(
                length_df,
                x='手法',
                y='チャンク長',
                title="チャンク長の分布",
                color='手法'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        evaluator = ChunkingEvaluator()
        comparison = evaluator.compare_methods(chunks_by_method)
        
        st.subheader("詳細評価メトリクス")
        
        eval_data = []
        for method, metrics in comparison.items():
            eval_data.append({
                '手法': method,
                'チャンク数': metrics['num_chunks'],
                '平均長': f"{metrics['avg_chunk_length']:.0f}",
                '中央値': f"{metrics['median_chunk_length']:.0f}",
                '長さ分散': f"{metrics['length_variance']:.0f}",
                'チャンク間一貫性': f"{metrics['avg_inter_chunk_coherence']:.3f}"
            })
        
        eval_df = pd.DataFrame(eval_data)
        st.dataframe(eval_df, use_container_width=True)
    
    with tabs[1]:
        st.header("RAG検索テスト")
        
        st.markdown("""
        各手法でチャンクを作成し、RAGシステムで検索を行います。
        クエリを入力して、各手法の検索結果を比較してください。
        """)
        
        query = st.text_input(
            "検索クエリを入力",
            value="科学技術政策の動向について教えてください"
        )
        
        top_k = st.slider("取得するチャンク数", 1, 10, 5)
        
        if st.button("🔍 検索実行"):
            evaluator = ChunkingEvaluator()
            
            results_by_method = {}
            
            for method, chunks in chunks_by_method.items():
                with st.spinner(f"{method} で検索中..."):
                    rag_system = RAGSystem(collection_name=f"rag_{method}")
                    rag_system.initialize_collection(chunks)
                    results = rag_system.retrieve(query, top_k=top_k)
                    results_by_method[method] = results
            
            for method, results in results_by_method.items():
                st.subheader(f"📌 {method} の検索結果")
                
                if results:
                    for i, result in enumerate(results):
                        with st.expander(f"結果 {i+1} (スコア: {result.score:.3f})"):
                            st.write(f"**チャンクID:** {result.chunk.chunk_id}")
                            st.write(f"**メタデータ:** {result.chunk.metadata}")
                            st.write("**内容:**")
                            st.text(result.chunk.content[:500] + "..." if len(result.chunk.content) > 500 else result.chunk.content)
                    
                    eval_metrics = evaluator.evaluate_retrieval_quality(query, results)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("平均スコア", f"{eval_metrics['avg_score']:.3f}")
                    col2.metric("平均関連性", f"{eval_metrics['avg_relevance']:.3f}")
                    col3.metric("Top-1スコア", f"{eval_metrics['top_1_score']:.3f}")
                else:
                    st.warning("検索結果がありません")
                
                st.divider()
    
    with tabs[2]:
        st.header("チャンク詳細")
        
        method_to_view = st.selectbox(
            "表示する手法を選択",
            list(chunks_by_method.keys())
        )
        
        chunks = chunks_by_method[method_to_view]
        
        st.write(f"**総チャンク数:** {len(chunks)}")
        
        chunk_index = st.number_input(
            "チャンクインデックス",
            min_value=0,
            max_value=len(chunks) - 1,
            value=0
        )
        
        if 0 <= chunk_index < len(chunks):
            chunk = chunks[chunk_index]
            
            st.subheader(f"チャンク {chunk_index}")
            st.write(f"**手法:** {chunk.method}")
            st.write(f"**チャンクID:** {chunk.chunk_id}")
            st.write(f"**文字数:** {len(chunk.content)}")
            st.write(f"**メタデータ:** {chunk.metadata}")
            
            st.text_area("内容", chunk.content, height=400)
    
    with tabs[3]:
        st.header("評価メトリクス詳細")
        
        evaluator = ChunkingEvaluator()
        comparison = evaluator.compare_methods(chunks_by_method)
        
        st.subheader("手法別評価")
        
        for method, metrics in comparison.items():
            with st.expander(f"📊 {method} の詳細メトリクス"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("チャンク数", metrics['num_chunks'])
                    st.metric("平均長", f"{metrics['avg_chunk_length']:.0f}")
                    st.metric("標準偏差", f"{metrics['std_chunk_length']:.0f}")
                    st.metric("最小長", metrics['min_chunk_length'])
                
                with col2:
                    st.metric("最大長", metrics['max_chunk_length'])
                    st.metric("中央値", f"{metrics['median_chunk_length']:.0f}")
                    st.metric("長さ分散", f"{metrics['length_variance']:.0f}")
                    st.metric("チャンク間一貫性", f"{metrics['avg_inter_chunk_coherence']:.3f}")
                
                st.json(metrics)
        
        st.subheader("手法間比較")
        
        comparison_data = []
        for method, metrics in comparison.items():
            comparison_data.append({
                '手法': method,
                'チャンク数': metrics['num_chunks'],
                '平均長': metrics['avg_chunk_length'],
                '一貫性': metrics['avg_inter_chunk_coherence'],
                '分散': metrics['length_variance']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        
        for metric in ['チャンク数', '平均長', '一貫性', '分散']:
            normalized_values = (comparison_df[metric] - comparison_df[metric].min()) / (comparison_df[metric].max() - comparison_df[metric].min())
            
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['手法'],
                y=normalized_values,
            ))
        
        fig.update_layout(
            title="正規化されたメトリクス比較",
            barmode='group',
            yaxis_title="正規化値 (0-1)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
