#!/bin/bash
# ============================================================
# Step 6: HTMLレポート生成
# 実験結果からHTMLレポートを生成
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# verification_packageがリポジトリ内にあるか判定
if [ -f "$PKG_ROOT/../generate_comprehensive_report.py" ]; then
    REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"
else
    echo "エラー: verification_packageをリポジトリ内に配置してください。"
    exit 1
fi
PROJECT_ROOT="$REPO_ROOT"

echo "============================================================"
echo "  HTMLレポート生成"
echo "============================================================"

cd "$REPO_ROOT"
echo ""
echo "  ⚠ 注意: このHTMLレポートは論文の主実験（30テーブル・150クエリ）"
echo "  論文の150クエリ実験検証には Step 7 (07_verify_results.py) を使用してください。"
echo ""

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
