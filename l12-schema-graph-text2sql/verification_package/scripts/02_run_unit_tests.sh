#!/bin/bash
# ============================================================
# Step 2: 単体テスト実行
# 80テスト全件パスを確認
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# verification_packageがリポジトリ内にあるか判定
if [ -d "$PKG_ROOT/../tests" ]; then
    REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"
else
    echo "エラー: verification_packageをリポジトリ内に配置してください。"
    echo "  例: cd machine-learning/l12-schema-graph-text2sql"
    echo "      cp -r /path/to/verification_package ."
    exit 1
fi

echo "============================================================"
echo "  単体テスト実行（80テスト）"
echo "============================================================"
echo "  パッケージ: $PKG_ROOT"
echo "  リポジトリ: $REPO_ROOT"
echo ""

cd "$REPO_ROOT"
python3 -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/test_results.txt

echo ""
echo "============================================================"

# 結果判定
PASSED=$(grep -c "PASSED" /tmp/test_results.txt 2>/dev/null || echo "0")
FAILED=$(grep -c "FAILED" /tmp/test_results.txt 2>/dev/null || echo "0")

echo "  結果: ${PASSED} passed / ${FAILED} failed"

if [ "$FAILED" -eq 0 ]; then
    echo "  判定: OK — 全テストパス"
else
    echo "  判定: NG — 失敗テストあり"
    echo "  失敗テストの詳細:"
    grep "FAILED" /tmp/test_results.txt
fi

echo "============================================================"
echo ""
echo "次のステップ: python3 scripts/03_verify_schema_graph.py"
