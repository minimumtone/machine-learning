#!/bin/bash
# ============================================================
# Step 4: 150クエリ実験（30テーブル・3条件比較）
# 要: OPENAI_API_KEY
#
# Usage:
#   bash scripts/04_run_150query_experiment.sh           # 全150クエリ（30-60分）
#   bash scripts/04_run_150query_experiment.sh --quick    # medium+complex 50件（10-20分）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

QUICK=""
if [ "$1" = "--quick" ]; then
    QUICK="--quick"
fi

if [ -n "$QUICK" ]; then
    echo "============================================================"
    echo "  クイックモード: medium + complex 50件のみ"
    echo "============================================================"
else
    echo "============================================================"
    echo "  150クエリ実験（30テーブル・3条件比較）"
    echo "============================================================"
fi

# APIキー確認
cd "$PROJECT_ROOT"
if [ -f .env ]; then
    source .env 2>/dev/null || true
    export $(grep -v '^#' .env | xargs) 2>/dev/null || true
fi

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_api_key_here" ]; then
    echo ""
    echo "  ※ OPENAI_API_KEY が設定されていません。"
    echo "  .env ファイルに OPENAI_API_KEY を設定してください。"
    echo ""
    echo "  APIキー不要の検証（Step 3）は既に完了していますか？"
    echo "  完了していれば、既存の結果ファイルで Step 7 の検証が可能です。"
    echo ""
    echo "  既存結果で検証する場合:"
    echo "    python3 scripts/07_verify_results.py"
    exit 1
fi

if [ -n "$QUICK" ]; then
    echo ""
    echo "  モデル: gpt-4o-mini"
    echo "  クエリ数: 50 (medium 25件 + complex 25件)"
    echo "  条件: Full Schema / Traversed / No Schema"
    echo "  推定API料金: $0.3-1"
    echo "  推定所要時間: 10-20分"
else
    echo ""
    echo "  モデル: gpt-4o-mini"
    echo "  クエリ数: 150 (6カテゴリ × 25件)"
    echo "  条件: Full Schema / Traversed / No Schema"
    echo "  推定API料金: $1-3"
    echo "  推定所要時間: 30-60分"
fi

echo ""
echo "  続行しますか? (y/n)"
read -r CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "  中止しました。"
    exit 0
fi

echo ""
echo "  実験開始..."
cd "$PROJECT_ROOT"
python3 experiments/run_extended_schema_experiment.py $QUICK 2>&1 | tee /tmp/experiment_log.txt

echo ""
echo "============================================================"
echo "  実験完了"
echo "============================================================"
echo ""

# 結果サマリ（生データから直接計算）
python3 -c "
import json
from pathlib import Path
p = Path('experiments/results/extended_schema_experiment.json')
if not p.exists():
    p = Path('experiments/results/extended_schema_experiment_150q.json')
if p.exists():
    data = json.loads(p.read_text())
    detail = data if isinstance(data, list) else data.get('detailed_results', [])
    N = len(detail)
    full_ok = sum(1 for d in detail if d.get('llm_full_schema', {}).get('success'))
    trav_ok = sum(1 for d in detail if d.get('llm_traversed', {}).get('success'))
    nosc_ok = sum(1 for d in detail if d.get('llm_no_schema', {}).get('success'))
    print(f'  条件           | 成功率')
    print('  ' + '-' * 40)
    print(f'  Full Schema    | {full_ok}/{N} ({full_ok/N*100:.1f}%)')
    print(f'  Traversed      | {trav_ok}/{N} ({trav_ok/N*100:.1f}%)')
    print(f'  No Schema      | {nosc_ok}/{N} ({nosc_ok/N*100:.1f}%)')
    print(f'')
    print(f'  Traversal効果: +{(trav_ok-full_ok)/N*100:.1f}pp')
else:
    print('  結果ファイルが見つかりません。')
"

echo ""
echo "次のステップ: bash scripts/05_run_rb_comparison.sh"
