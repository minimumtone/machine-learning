#!/bin/bash
# ============================================================
# Step 1: 環境構築スクリプト
# PostgreSQL起動 → 7テーブルスキーマ + seed → 30テーブル拡張
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# verification_packageがリポジトリ内にあるか判定
if [ -f "$PKG_ROOT/../experiments/setup_extended_schema.py" ]; then
    REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"
else
    echo "エラー: verification_packageをリポジトリ内に配置してください。"
    echo "  例: cd machine-learning/l12-schema-graph-text2sql"
    echo "      cp -r /path/to/verification_package ."
    exit 1
fi

echo "============================================================"
echo "  Schema-Graph Text-to-SQL 検証環境構築"
echo "============================================================"
echo "  パッケージ: $PKG_ROOT"
echo "  リポジトリ: $REPO_ROOT"

# --- 1. Python依存パッケージ ---
echo ""
echo "[1/5] Python依存パッケージのインストール..."
pip install psycopg2-binary psycopg "openai>=1.0" networkx matplotlib numpy 2>&1 | tail -3
echo "  → 完了"

# --- 2. .envファイルの確認 ---
echo ""
echo "[2/5] .envファイルの確認..."
if [ ! -f "$REPO_ROOT/.env" ]; then
    if [ -f "$PKG_ROOT/config/.env.example" ]; then
        cp "$PKG_ROOT/config/.env.example" "$REPO_ROOT/.env"
    elif [ -f "$REPO_ROOT/.env.example" ]; then
        cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    fi
    echo "  → .env を作成しました。"
    echo "  ※ OPENAI_API_KEY を設定してください（LLM実験を行う場合）"
    echo "    vim $REPO_ROOT/.env"
else
    echo "  → .env は既に存在します"
fi

# --- 3. Docker PostgreSQL起動 ---
echo ""
echo "[3/5] PostgreSQL (Docker) 起動..."
if docker ps | grep -q l12_postgres; then
    echo "  → l12_postgres は既に起動中"
else
    # 既存コンテナを削除（ボリュームが古い場合を考慮）
    docker rm -f l12_postgres 2>/dev/null || true
    docker compose -f "$PKG_ROOT/config/docker-compose.yml" up -d 2>&1
    echo "  → 起動中... 10秒待機"
    sleep 10
fi

# --- 4. DB接続確認 ---
echo ""
echo "[4/5] DB接続確認..."
python3 -c "
import psycopg2
try:
    conn = psycopg2.connect(
        dbname='l12_materials', user='l12_user',
        password='l12_password', host='localhost', port=5432
    )
    cur = conn.cursor()
    cur.execute(\"SELECT count(*) FROM information_schema.tables WHERE table_schema='public'\")
    n = cur.fetchone()[0]
    cur.execute('SELECT count(*) FROM material_entry')
    m = cur.fetchone()[0]
    print(f'  テーブル数: {n}')
    print(f'  material_entry件数: {m}')
    conn.close()
except Exception as e:
    print(f'  エラー: {e}')
    exit(1)
"

# --- 5. 30テーブル確認 ---
echo ""
echo "[5/5] 30テーブルスキーマの確認..."
echo "  ※ extended_schema.sqlによりDocker起動時に30テーブルが一括作成されます"

# --- 最終確認 ---
echo ""
echo "============================================================"
echo "  環境構築完了"
echo "============================================================"
python3 -c "
import psycopg2
conn = psycopg2.connect(dbname='l12_materials', user='l12_user', password='l12_password', host='localhost', port=5432)
cur = conn.cursor()
cur.execute(\"SELECT count(*) FROM information_schema.tables WHERE table_schema='public'\")
tables = cur.fetchone()[0]
cur.execute('SELECT count(*) FROM material_entry')
entries = cur.fetchone()[0]
cur.execute(\"\"\"
    SELECT count(*) FROM information_schema.table_constraints
    WHERE constraint_type = 'FOREIGN KEY' AND table_schema = 'public'
\"\"\")
fks = cur.fetchone()[0]
print(f'  テーブル数:         {tables}')
print(f'  material_entry件数: {entries}')
print(f'  FK関係数:           {fks}')
conn.close()
"
echo ""
echo "次のステップ: bash verification_package/scripts/02_run_unit_tests.sh"
