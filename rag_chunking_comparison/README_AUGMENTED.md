# 拡張RAGチャンク作成システム

このシステムは、PDFからテキストを抽出してRAGシステムに使用する際の、異なるチャンク作成手法を比較し、さらにデータ拡張により検索精度を向上させます。

## 新機能: データ拡張による検索精度向上

各チャンクから複数の言い換え文章を自動生成し、ベクトル化することで、RAGシステムの検索精度を大幅に向上させます。

### 仕組み

1. **チャンク作成**: 3つの手法（Markdown-based、MeCab Semantic、Self-Route）でチャンクを作成
2. **言い換え生成**: 各チャンクに対して、OpenAI APIを使用して10パターンの言い換え文章を生成
3. **ベクトル化**: 元のチャンク + 10パターン = 合計11倍のベクトルデータを作成
4. **検索**: ユーザーの問い合わせに対して、全てのバリエーションから検索し、元のチャンクを返す

### メリット

- **検索精度向上**: 様々な表現での問い合わせに対応できる
- **ロバスト性**: ユーザーの問い合わせ表現に依存しない
- **柔軟性**: 言い換えパターン数を調整可能

## セットアップ

### 1. 依存パッケージのインストール

```bash
cd rag_chunking_comparison
pip install -r requirements.txt
```

### 2. OpenAI APIキーの設定

データ拡張機能を使用するには、OpenAI APIキーが必要です。

```bash
export OPENAI_API_KEY='your-api-key-here'
```

または、`.env`ファイルに設定:

```
OPENAI_API_KEY=your-api-key-here
```

## 使用方法

### 基本的なチャンク比較（拡張なし）

```bash
# 元のStreamlitアプリを起動
streamlit run streamlit_app.py
```

### 拡張RAGシステム（データ拡張あり）

```bash
# 拡張版Streamlitアプリを起動
streamlit run streamlit_augmented_app.py
```

### テストスクリプトの実行

```bash
# 基本的なチャンク作成とRAGのテスト
python test_chunking.py

# 拡張RAGシステムのテスト（OpenAI APIキーが必要）
python test_augmented_chunking.py
```

## ファイル構成

### 新規追加ファイル

- `paraphrase_generator.py`: 言い換え文章生成システム
  - OpenAI APIを使用して、各チャンクから複数の言い換えを生成
  - キャッシュ機能により、同じチャンクの再生成を防止

- `augmented_rag_system.py`: 拡張RAGシステム
  - 元のチャンク + 言い換えバリエーションを全てベクトル化
  - 検索時は全バリエーションから検索し、元のチャンクを返す
  - ベースラインとの比較機能

- `augmented_evaluation.py`: 拡張システムの評価メトリクス
  - 言い換え品質の評価（元のチャンクとの類似度、バリエーション間の多様性）
  - 検索精度向上の評価（スコア改善、バリエーション使用率）

- `streamlit_augmented_app.py`: 拡張版Streamlitアプリ
  - 基本的なチャンク比較機能
  - 拡張RAG検索機能
  - ベースラインとの比較可視化

- `test_augmented_chunking.py`: 拡張システムの包括的テスト
  - 言い換え生成のテスト
  - 拡張RAGシステムのテスト
  - ベースラインとの比較テスト

### 既存ファイル

- `pdf_to_markdown.py`: PDF→Markdown変換
- `chunking_strategies.py`: 3つのチャンク作成手法
- `rag_system.py`: 基本RAGシステム
- `evaluation.py`: 基本評価メトリクス
- `streamlit_app.py`: 基本Streamlitアプリ
- `test_chunking.py`: 基本テストスクリプト

## 使用例

### Python APIの使用

```python
from pdf_to_markdown import convert_pdf_to_markdown
from chunking_strategies import MeCabSemanticChunker
from paraphrase_generator import augment_chunks
from augmented_rag_system import AugmentedRAGSystem

# PDFを読み込み
pdf_path = "path/to/your.pdf"
markdown_text = convert_pdf_to_markdown(pdf_path)

# チャンクを作成
chunker = MeCabSemanticChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk(markdown_text)

# 最初の5チャンクを拡張（10パターンの言い換えを生成）
augmented_chunks = augment_chunks(chunks[:5], num_variations=10)

# 拡張RAGシステムを初期化
aug_rag = AugmentedRAGSystem(collection_name="my_augmented_rag")
aug_rag.initialize_collection(augmented_chunks)

# 検索
query = "科学技術政策の動向について教えてください"
results = aug_rag.retrieve(query, top_k=5)

# 結果を表示
for i, result in enumerate(results, 1):
    print(f"{i}. スコア: {result.score:.3f}")
    print(f"   マッチしたバリエーション: {result.variation_index}")
    print(f"   内容: {result.original_chunk.content[:100]}...")
```

## 評価メトリクス

### 言い換え品質

- **元のチャンクとの類似度**: 言い換えが元の意味を保持しているか
- **バリエーション間の多様性**: 言い換えが十分に異なる表現を使用しているか

### 検索精度向上

- **スコア改善**: ベースラインと比較した検索スコアの改善度
- **バリエーション使用率**: 言い換えがどの程度検索に貢献しているか
- **重複率**: ベースラインと拡張システムで同じチャンクが検索されているか

## パフォーマンス

### 処理時間

- **言い換え生成**: 1チャンクあたり約5-10秒（OpenAI API依存）
- **ベクトル化**: チャンク数 × (1 + 言い換え数) に比例
- **検索**: ベースラインとほぼ同等（ChromaDBの効率的な検索）

### コスト

- OpenAI APIの使用料金が発生します
- gpt-4o-miniモデルを使用（コスト効率重視）
- キャッシュ機能により、同じチャンクの再生成を防止

## トラブルシューティング

### OpenAI APIキーが設定されていない

```
ValueError: OpenAI API key is required.
```

→ `OPENAI_API_KEY`環境変数を設定してください

### メモリ不足

大量のチャンクを一度に拡張すると、メモリ不足になる可能性があります。

→ `max_chunks_to_augment`パラメータで拡張するチャンク数を制限してください

### 処理時間が長い

言い換え生成はOpenAI APIを使用するため、時間がかかります。

→ キャッシュ機能を活用し、同じチャンクの再生成を避けてください

## ライセンス

このプロジェクトは、元のRAGチャンク比較システムと同じライセンスに従います。

## 参考資料

- [元の実装記事](https://note.com/daigo40215499/n/nbcca777ffe1c)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
