# 隔離環境検証パッケージ — L1$_2$ Schema-Graph Text-to-SQL

本パッケージは、論文の全評価（ablation・独立評価・転用試験・CTE評価）を
隔離環境で再現するための資材・スクリプト・手順一式です。

## 1. 検証資材一覧

| ディレクトリ/ファイル | 内容 |
|---|---|
| `llm/` | パイプライン本体（SQL生成・スキーマリンキング・辞書・reranker・プロンプト） |
| `llm/mecab_materials.dic` | MeCab材料辞書（497語、コンパイル済み） |
| `graph/` | スキーマグラフ + Steiner木JOIN経路探索 |
| `safety/` | SQLGuard（AST検証・allowed_schema.yaml） |
| `evaluation/` | 評価器・gold SQL・expected results・全評価データセット・既存結果JSON |
| `scripts/` | 全評価スクリプト（下記 §3） |
| `db/` | スキーマSQL + データSQL（L1$_2$ 1,470行 + 純物質89元素 = 1,559行）+ 転用スキーマ |
| `docker/docker-compose.yml` | PostgreSQL 15（初期化SQL自動投入） |
| `few_shot_examples.json` | few-shot例42件（ルート直下） |
| `tests/` | 単体テスト134件 |
| `paper/` | 論文ソース（main.tex 日 / main_en.tex 英）+ PDF + paper_data.json |
| `pyproject.toml` | 依存関係定義 |

### 評価データセット
| ファイル | 件数 | 用途 |
|---|---|---|
| `evaluation/evaluation_dataset.jsonl` | 100 | 著者設計メイン評価（ablation対象） |
| `evaluation/expert_evaluation_dataset.jsonl` | 100 | 独立設計クエリ（外的妥当性） |
| `evaluation/transfer_evaluation_dataset.jsonl` | 20 | 未知スキーマ転用試験 |
| `evaluation/cte_evaluation_dataset.jsonl` | 10 | 新規CTEパターン（既存5件と合わせて15件評価） |

### 既存結果（照合用リファレンス）
- `evaluation/ablation_run_{1..5}.json` + `ablation_multirun_stats.json` — 5-run ablation（full 84.7±0.5%）
- `evaluation/independent_eval_results.json` — 独立評価100件（72.5%）
- `evaluation/transfer_eval_results.json` — 転用試験20件（80.5%）
- `evaluation/cte_eval_results.json` — CTE 15件（54.9%）
- `paper/paper_data.json` — 全論文数値のSSoT（compute_all_figures.py が生成）

## 2. 環境要件

- OS: Linux / macOS（x86_64/arm64）
- Python: **3.11以上**（検証済み: 3.12.8）
- Docker + docker compose（PostgreSQL 15用）
- OpenAI APIキー（gpt-5.5 アクセス権）
- ディスク: ~2GB（reranker用torch含む場合 ~5GB）
- LaTeX（PDF再生成する場合のみ）: lualatex（日本語版）+ pdflatex（英語版）

## 3. セットアップ手順

```bash
# 1. 展開
unzip l12-verification-package.zip -d l12-verify && cd l12-verify

# 2. Python環境
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,reranker]"
# 日本語トークナイズ（辞書機能に必要）
pip install mecab-python3 ipadic

# 3. DB起動（初期化SQLが自動投入され 1,559行になる）
docker compose -f docker/docker-compose.yml up -d
sleep 10
docker exec -i l12_postgres \
  psql -U l12_user -d l12_materials -c "SELECT count(*) FROM material_entry;"
# → 1559 であること

# 4. APIキー
export OPENAI_API_KEY=sk-...
# （任意）モデル切替: export LLM_MODEL=gpt-5.5

# 5. 動作確認
pytest tests/ -q                 # → 134 passed（API・DB不要）
python scripts/compute_all_figures.py   # → paper/paper_data.json 再生成（DB必要）
```

接続情報はデフォルトで `localhost:5432 / l12_user / l12_password / l12_materials`。
変更する場合は環境変数 `POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER /
POSTGRES_PASSWORD / POSTGRES_DB` を設定してください。

## 4. 検証スクリプトと実行手順

### 4.1 メイン評価 / ablation（論文 表: full 84.7±0.5%）
```bash
# 1ラン（7条件×100件、約7時間、API約700回）
python scripts/eval_ablation.py
# 5ラン統計（約35時間、API約3,500回）— 論文のmean±SD+Wilcoxon再現
python scripts/eval_ablation_multirun.py --n-runs 5 --start-run 1
```
出力: `evaluation/ablation_run_N.json`, `ablation_multirun_stats.json`

### 4.2 独立評価100件（論文: 72.5%）
```bash
python scripts/eval_independent.py          # 約70分、API 100回
```
出力: `evaluation/independent_eval_results.json`

### 4.3 未知スキーマ転用試験（論文: 80.5%）
```bash
# 転用DB構築（テーブル名・カラム名を全変更した5テーブル、データはメインDBから複製）
python scripts/build_transfer_db.py
# ゼロ適応評価（辞書・few-shot・グラフの再設計なし。約13分、API 20回）
python scripts/eval_transfer.py
```
出力: `evaluation/transfer_eval_results.json`
※ `eval_transfer.py` が `SQL_PROMPT_TEMPLATE` 環境変数で転用用プロンプト
（`llm/prompt_templates/sql_generation_prompt_transfer.md`）に自動切替します。

### 4.4 CTE 15件評価（論文: 54.9% = 既存5パターン100% + 新規10パターン32.3%）
```bash
# 既存CTE 5件（メイン評価セット内）+ 新規10件を結合して15件データセットを作成
python - <<'EOF'
import json, pathlib
ids = {"q_vhard_009", "q_vhard_016", "q_vhard_018", "q_vhard_019", "q_vhard_020"}
lines = [l for l in pathlib.Path("evaluation/evaluation_dataset.jsonl").read_text().splitlines()
         if json.loads(l)["id"] in ids]
lines += pathlib.Path("evaluation/cte_evaluation_dataset.jsonl").read_text().splitlines()
pathlib.Path("evaluation/cte15_dataset.jsonl").write_text("\n".join(lines) + "\n")
EOF
python scripts/eval_independent.py \
  --dataset evaluation/cte15_dataset.jsonl \
  --output evaluation/cte_eval_results.json    # 約10分、API 15回
```

### 4.5 感度分析（任意、論文 §感度分析）
```bash
python scripts/eval_fewshot_sensitivity.py   # k=1,3,5,10,15（約5時間）
python scripts/eval_dict_sensitivity.py      # 辞書 full/50%/25%/0%（約4時間）
python scripts/eval_model_comparison.py      # GPT-5.5 vs GPT-4o（ANTHROPIC_API_KEYがあればClaudeも）
```

### 4.6 gold SQL / expected results の整合確認
```bash
python scripts/check_expected_results.py            # 全gold SQLをDB実行し期待値と照合
python scripts/check_expected_results.py --update   # 期待値の更新（DB変更時のみ）
```

### 4.7 集計・図表・論文
```bash
python scripts/compute_all_figures.py   # paper/paper_data.json 再生成（SSoT）
python scripts/generate_figures.py      # paper/figures/ の図再生成
cd paper && lualatex main.tex && lualatex main.tex          # 日本語版 23pp
cd paper && pdflatex main_en.tex && bibtex main_en && pdflatex main_en.tex && pdflatex main_en.tex  # 英語版 46pp
```

## 5. 判定基準（期待値）

| 検証 | 期待値 | 許容範囲の目安 |
|---|---|---|
| pytest | 134 passed | 完全一致 |
| DB行数 | material_entry 1,559行 / 31テーブル+1ビュー | 完全一致 |
| ablation full（1ラン） | 84.7% | ±1.5pp（LLM非決定性） |
| ablation 5-run: no_dict / no_fewshot | −7.3pp / −7.4pp（p<0.001） | 有意性の再現 |
| 独立評価100件 | 72.5% | ±3pp |
| 転用試験20件 | 80.5% | ±5pp（20件のため1件=5pp） |
| CTE 15件 | 54.9%（既存5件は100%） | 既存5件100%は再現されること |
| compute_all_figures | エラーなしで paper_data.json 生成 | — |

## 6. 注意事項

- 評価指標は実行結果の行集合recall（gold expected resultsとの照合）。
  LLM出力の非決定性により±1〜2ppの揺らぎは正常です。
- `sentence-transformers` 未導入の場合、few-shot rerankerはTF-IDF順の
  フォールバックで動作します（ログにCross-Encoder unavailableと出ます）。
  論文数値の再現にはreranker extraの導入を推奨します。
- API費用目安: ablation 5-run 全再現で約3,500呼び出し。小規模確認のみなら
  4.2〜4.4（計135呼び出し、~1.5時間）を推奨します。
- ハードコードなし: DB接続・モデル名・プロンプトはすべて環境変数で切替可能。
