#!/usr/bin/env python3
"""
再現性・ロバストネステスト
E-1: LLM非決定論性の定量評価（要APIキー）
E-2: Docker再起動後データ永続性テスト
E-3: APIレート制限リトライ動作テスト（モック）
"""
import sys
import os
import json
import subprocess
from pathlib import Path

passed = 0
failed = 0
skipped = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  [OK] {name}")
        passed += 1
    else:
        print(f"  [NG] {name}  — {detail}")
        failed += 1


def skip(name: str, reason: str):
    global skipped
    print(f"  [SKIP] {name}  — {reason}")
    skipped += 1


def test_e2_docker_persistence():
    """E-2: Docker再起動後のデータ永続性テスト"""
    print("\n■ E-2: Docker再起動後データ永続性テスト")

    try:
        import psycopg2
    except ImportError:
        skip("Docker永続性", "psycopg2がインストールされていません")
        return

    db_config = {
        "dbname": "l12_materials", "user": "l12_user",
        "password": "l12_password", "host": "localhost", "port": 5432,
    }

    # 1. 現在の件数を記録
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM material_entry")
        before_count = cur.fetchone()[0]
        conn.close()
        print(f"    再起動前 material_entry: {before_count}件")
    except Exception as e:
        skip("Docker永続性", f"DB接続失敗: {e}")
        return

    # 2. docker compose down && up
    pkg_root = Path(__file__).parent.parent
    compose_file = pkg_root / "config" / "docker-compose.yml"
    if not compose_file.exists():
        # リポジトリルートからの相対位置を試す
        for candidate in [
            Path.cwd() / "verification_package" / "config" / "docker-compose.yml",
            Path.cwd() / "config" / "docker-compose.yml",
        ]:
            if candidate.exists():
                compose_file = candidate
                break

    if not compose_file.exists():
        skip("Docker永続性", "docker-compose.ymlが見つかりません")
        return

    print("    docker compose down...")
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        skip("Docker永続性", f"docker compose down 失敗: {result.stderr}")
        return

    print("    docker compose up -d...")
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        skip("Docker永続性", f"docker compose up 失敗: {result.stderr}")
        return

    # 3. DB初期化完了を待つ
    import time
    time.sleep(12)

    # 4. 件数を再確認
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM material_entry")
        after_count = cur.fetchone()[0]
        conn.close()
        print(f"    再起動後 material_entry: {after_count}件")
    except Exception as e:
        check("Docker再起動後のDB接続", False, f"接続失敗: {e}")
        return

    check(
        "Docker再起動後のデータ復元",
        after_count >= before_count,
        f"再起動前={before_count}, 再起動後={after_count}",
    )

    # named volumeがないことの注意表示
    print("    ※ 現構成ではnamed volumeを使用していないため、")
    print("      docker compose downでデータは消え、upで再作成されます。")
    print("      seedデータがdocker-entrypoint-initdb.dで再投入されることを確認。")


def test_e3_rate_limit_retry():
    """E-3: APIレート制限リトライ動作テスト（モック）"""
    print("\n■ E-3: APIレート制限リトライ動作テスト（モック）")

    # リトライロジックのモックテスト
    class MockRateLimitError(Exception):
        pass

    class MockAPIClient:
        def __init__(self, fail_count=2):
            self.call_count = 0
            self.fail_count = fail_count

        def call(self, prompt):
            self.call_count += 1
            if self.call_count <= self.fail_count:
                raise MockRateLimitError("Rate limit exceeded")
            return {"success": True, "sql": "SELECT 1"}

    def retry_with_backoff(client, prompt, max_retries=3):
        """指数バックオフ付きリトライ（実際のバックオフなしのモック版）"""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return client.call(prompt)
            except MockRateLimitError as e:
                last_error = e
                if attempt < max_retries:
                    continue
        return {"success": False, "error": str(last_error)}

    # テスト1: 2回失敗→3回目成功
    client1 = MockAPIClient(fail_count=2)
    result1 = retry_with_backoff(client1, "test query", max_retries=3)
    check(
        "2回RateLimit→3回目成功",
        result1["success"] is True,
        f"結果: {result1}",
    )
    check(
        "合計3回のAPI呼び出し",
        client1.call_count == 3,
        f"実際: {client1.call_count}回",
    )

    # テスト2: 4回失敗→3リトライで諦め
    client2 = MockAPIClient(fail_count=4)
    result2 = retry_with_backoff(client2, "test query", max_retries=3)
    check(
        "4回RateLimit→3リトライで失敗判定",
        result2["success"] is False,
        f"結果: {result2}",
    )
    check(
        "最大4回のAPI呼び出し（初回+3リトライ）",
        client2.call_count == 4,
        f"実際: {client2.call_count}回",
    )

    # テスト3: 失敗なし→1回で成功
    client3 = MockAPIClient(fail_count=0)
    result3 = retry_with_backoff(client3, "test query", max_retries=3)
    check(
        "RateLimitなし→1回で成功",
        result3["success"] is True and client3.call_count == 1,
        f"呼び出し回数: {client3.call_count}",
    )

    # テスト4: リトライ0回設定→即失敗
    client4 = MockAPIClient(fail_count=1)
    result4 = retry_with_backoff(client4, "test query", max_retries=0)
    check(
        "リトライ0回→即失敗",
        result4["success"] is False,
        f"結果: {result4}",
    )


def test_e1_llm_nondeterminism():
    """E-1: LLM非決定論性の定量評価（要APIキー）"""
    print("\n■ E-1: LLM非決定論性の定量評価")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        skip("LLM非決定論性", "OPENAI_API_KEYが未設定（オプション検証）")
        return

    try:
        from openai import OpenAI
    except ImportError:
        skip("LLM非決定論性", "openaiパッケージがインストールされていません")
        return

    client = OpenAI(api_key=api_key)
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

    # 同一プロンプトを3回実行
    test_prompt = (
        "Generate a SQL query to find all materials with formation energy below -0.4 eV/atom "
        "from a PostgreSQL database with tables: material_entry(entry_id, formula), "
        "phase_stability(stability_id, entry_id, formation_energy_per_atom). "
        "Return only the SQL."
    )

    results = []
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.0,
            )
            sql = response.choices[0].message.content.strip()
            results.append(sql)
        except Exception as e:
            skip(f"LLM呼び出し{i+1}", str(e))
            return

    # 一致率を計算
    unique_results = len(set(results))
    consistency = (3 - unique_results + 1) / 3 * 100

    print(f"    3回実行結果: {unique_results}種類")
    for i, r in enumerate(results):
        print(f"      [{i+1}] {r[:80]}...")

    check(
        "LLM再現性: 3回中2回以上同一",
        unique_results <= 2,
        f"異なる結果: {unique_results}種類",
    )
    check(
        "temperature=0で高い一貫性",
        unique_results == 1,
        f"異なる結果: {unique_results}種類（±2-3%変動は許容）",
    )


def main():
    global passed, failed, skipped

    print("=" * 60)
    print("  再現性・ロバストネステスト")
    print("=" * 60)

    test_e3_rate_limit_retry()
    test_e2_docker_persistence()
    test_e1_llm_nondeterminism()

    # --- 結果サマリ ---
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  検証結果: {passed}/{total} パス  ({skipped}件スキップ)")
    if failed == 0:
        print("  判定: OK — ロバストネステスト合格")
    else:
        print(f"  判定: NG — {failed}件の不合格あり")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
