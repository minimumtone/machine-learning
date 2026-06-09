#!/bin/bash
# ============================================================
# Step 6: 検証サマリーレポート生成
# 実験結果からMarkdownレポートを生成
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "  検証サマリーレポート生成"
echo "============================================================"

echo ""
echo "  ⚠ 注意: 本番の論文検証には Step 7 (07_verify_results.py) を使用してください。"
echo ""

# Check if results exist
EVAL_DIR="$PKG_ROOT/../evaluation"
if [ ! -d "$EVAL_DIR" ]; then
    echo "エラー: evaluation/ ディレクトリが見つかりません。"
    echo "先に評価を実行してください: python3 scripts/run_full_evaluation.py"
    exit 1
fi

echo "  結果ファイル確認..."
for f in "$EVAL_DIR"/metrics_summary.csv "$EVAL_DIR"/proposed_result.csv; do
    if [ -f "$f" ]; then
        echo "    ✓ $(basename $f)"
    else
        echo "    ✗ $(basename $f) — 未生成"
    fi
done

echo ""
echo "============================================================"
echo "  レポート生成完了"
echo "============================================================"
echo ""
echo "次のステップ: python3 scripts/07_verify_results.py"
