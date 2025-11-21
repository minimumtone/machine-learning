# RAG チャンク作成手法の比較

PDFからテキストを抽出してRAGシステムに使用する際の、異なるチャンク作成手法を比較するプロジェクトです。

## 概要

このプロジェクトは、RAG (Retrieval-Augmented Generation) システムにおいて、PDFドキュメントを効果的にチャンク分割するための3つの異なる手法を実装し、比較評価します。

### 実装された手法

1. **Markdown-based Chunking**
   - 標準的なMarkdownヘッダーベースのチャンク分割
   - `langchain-text-splitters`の`MarkdownHeaderTextSplitter`を使用
   - 構造化されたドキュメントに適している

2. **MeCab Semantic Chunking**
   - MeCab形態素解析を使用した意味的な単位でのチャンク分割
   - 日本語の文境界を正確に検出
   - 意味的なまとまりを保持しながらチャンク化

3. **Self-Route Chunking**
   - セマンティック類似度を用いた知的なチャンク境界検出
   - 文間の意味的類似度を計算し、類似度が閾値以下の場合に境界を設定
   - コンテキストの一貫性を最大化

## ファイル構成

```
rag_chunking_comparison/
├── README.md                    # このファイル
├── requirements.txt             # 依存パッケージ
├── pdf_to_markdown.py          # PDF→Markdown変換
├── chunking_strategies.py      # 3つのチャンク作成手法の実装
├── rag_system.py               # RAGシステム（埋め込み・検索）
├── evaluation.py               # 評価メトリクス
├── streamlit_app.py            # Streamlit比較アプリ
└── test_chunking.py            # 包括的なテストスクリプト
```

## インストール

```bash
cd rag_chunking_comparison
pip install -r requirements.txt
```

## 使用方法

### 1. テストスクリプトの実行

全ての手法を包括的にテストします：

```bash
python test_chunking.py
```

### 2. Streamlitアプリの起動

インタラクティブな比較アプリを起動します：

```bash
streamlit run streamlit_app.py
```

アプリでは以下の機能が利用できます：
- チャンク統計の比較
- RAG検索テスト
- チャンク詳細の確認
- 評価メトリクスの可視化

### 3. Pythonスクリプトでの使用

```python
from pdf_to_markdown import PDFToMarkdownConverter
from chunking_strategies import MarkdownChunker, MeCabSemanticChunker, SelfRouteChunker
from rag_system import RAGSystem
from evaluation import ChunkingEvaluator

# PDFをMarkdownに変換
converter = PDFToMarkdownConverter("path/to/pdf")
md_text = converter.convert_to_markdown()

# チャンク作成
markdown_chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=200)
chunks = markdown_chunker.chunk(md_text)

# RAGシステムで検索
rag_system = RAGSystem(collection_name="my_collection")
rag_system.initialize_collection(chunks)
results = rag_system.retrieve("検索クエリ", top_k=5)

# 評価
evaluator = ChunkingEvaluator()
metrics = evaluator.evaluate_chunk_quality(chunks)
```

## 評価メトリクス

各手法は以下のメトリクスで評価されます：

### チャンク品質メトリクス
- **チャンク数**: 生成されたチャンクの総数
- **平均チャンク長**: チャンクの平均文字数
- **標準偏差**: チャンク長のばらつき
- **長さ分散**: チャンク長の分散
- **チャンク間一貫性**: 隣接チャンク間のセマンティック類似度

### 検索品質メトリクス
- **平均スコア**: 検索結果の平均類似度スコア
- **平均関連性**: クエリとチャンクの平均セマンティック類似度
- **Top-1スコア**: 最上位結果のスコア
- **Top-3平均スコア**: 上位3件の平均スコア

## テスト結果の例

```
--- markdown ---
  Number of chunks: 71
  Average chunk length: 784.13
  Avg inter-chunk coherence: 0.565

--- mecab_semantic ---
  Number of chunks: 4
  Average chunk length: 12462.25
  Avg inter-chunk coherence: 0.476

--- self_route ---
  Number of chunks: 4
  Average chunk length: 11671.25
  Avg inter-chunk coherence: 0.520
```

## 技術スタック

- **PDF処理**: pymupdf4llm, PyMuPDF
- **テキスト分割**: langchain-text-splitters
- **形態素解析**: MeCab (mecab-python3, unidic-lite)
- **埋め込み**: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **ベクトルDB**: ChromaDB
- **可視化**: Streamlit, Plotly

## 参考文献

- [RAGのチャンク作成（PDFからテキスト化するための前処理について）](https://note.com/daigo40215499/n/nbcca777ffe1c)
- pymupdf4llm: PDF to Markdown conversion
- LangChain Text Splitters: Markdown-based chunking
- MeCab: Japanese morphological analysis

## ライセンス

このプロジェクトは研究・教育目的で作成されています。

## 作成者

Devin AI - 2025年11月
