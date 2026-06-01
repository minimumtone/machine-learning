#!/bin/bash
# ============================================================
# Step 2: 単体テスト実行
# 80テスト全件パスを確認
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "  単体テスト実行（80テスト）"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT"
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
