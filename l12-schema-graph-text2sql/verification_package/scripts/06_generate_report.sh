#!/bin/bash
# ============================================================
# Step 6: HTMLレポート生成
# 実験結果からHTMLレポートを生成
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "  HTMLレポート生成"
echo "============================================================"

cd "$PROJECT_ROOT"

echo ""
echo "  [1/2] 包括的実験レポート生成中..."
python3 generate_comprehensive_report.py
echo "  → comprehensive_experiment_report.html"

echo ""
echo "  [2/2] 統合検証レポート生成中..."
python3 generate_unified_report.py
echo "  → unified_verification_report.html"

echo ""
echo "============================================================"
echo "  レポート生成完了"
echo "============================================================"
echo ""
echo "  ブラウザで開いて確認:"
echo "    open comprehensive_experiment_report.html"
echo "    open unified_verification_report.html"
echo ""
echo "次のステップ: python3 scripts/07_verify_results.py"
