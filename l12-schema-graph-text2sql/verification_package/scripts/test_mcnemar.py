#!/usr/bin/env python3
"""
D-1: McNemar検定ロジックの単体テスト
既知のモックデータで χ² 計算と有意性判定の正確性を検証する。
C-2: unnecessary_join_rate計算ロジックの検証も含む。
"""
import sys

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  [OK] {name}")
        passed += 1
    else:
        print(f"  [NG] {name}  — {detail}")
        failed += 1


def mcnemar_chi2(b: int, c: int) -> float:
    """Yates補正付きMcNemar検定の χ² 値を計算"""
    if b + c == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / (b + c)


def main():
    global passed, failed

    print("=" * 60)
    print("  D-1: McNemar検定ロジック単体テスト")
    print("=" * 60)

    # --- テスト1: 既知のb,c値 ---
    print("\n■ 1. 既知値での χ² 計算")

    # b=10, c=2: χ² = (|10-2|-1)² / (10+2) = 49/12 = 4.083
    chi2 = mcnemar_chi2(10, 2)
    check("b=10, c=2 → χ²=4.08", abs(chi2 - 4.083) < 0.01, f"実際: {chi2:.3f}")
    check("b=10, c=2 → p<0.05（有意）", chi2 > 3.84, f"χ²={chi2:.3f}")

    # b=5, c=3: χ² = (|5-3|-1)² / (5+3) = 1/8 = 0.125
    chi2 = mcnemar_chi2(5, 3)
    check("b=5, c=3 → χ²=0.125", abs(chi2 - 0.125) < 0.01, f"実際: {chi2:.3f}")
    check("b=5, c=3 → p≥0.05（非有意）", chi2 <= 3.84, f"χ²={chi2:.3f}")

    # b=0, c=0: χ² = 0
    chi2 = mcnemar_chi2(0, 0)
    check("b=0, c=0 → χ²=0.0", chi2 == 0.0, f"実際: {chi2:.3f}")

    # b=20, c=5: χ² = (|20-5|-1)² / (20+5) = 196/25 = 7.84
    chi2 = mcnemar_chi2(20, 5)
    check("b=20, c=5 → χ²=7.84", abs(chi2 - 7.84) < 0.01, f"実際: {chi2:.3f}")
    check("b=20, c=5 → p<0.05（有意）", chi2 > 3.84, f"χ²={chi2:.3f}")

    # 対称: b=5, c=20 → 同じ χ²
    chi2_sym = mcnemar_chi2(5, 20)
    check("対称性: b=5,c=20 ≈ b=20,c=5", abs(chi2 - chi2_sym) < 0.001, f"{chi2:.3f} vs {chi2_sym:.3f}")

    # --- テスト2: ペアリングロジック ---
    print("\n■ 2. クエリIDペアリングロジック")

    from collections import defaultdict

    # モックデータ: 5クエリ × 3条件
    mock_detail = [
        {"query_id": "Q1", "condition": "full_schema", "success": True},
        {"query_id": "Q1", "condition": "traversed", "success": True},
        {"query_id": "Q1", "condition": "no_schema", "success": False},
        {"query_id": "Q2", "condition": "full_schema", "success": False},
        {"query_id": "Q2", "condition": "traversed", "success": True},
        {"query_id": "Q2", "condition": "no_schema", "success": False},
        {"query_id": "Q3", "condition": "full_schema", "success": True},
        {"query_id": "Q3", "condition": "traversed", "success": False},
        {"query_id": "Q3", "condition": "no_schema", "success": False},
        {"query_id": "Q4", "condition": "full_schema", "success": False},
        {"query_id": "Q4", "condition": "traversed", "success": False},
        {"query_id": "Q4", "condition": "no_schema", "success": False},
        {"query_id": "Q5", "condition": "full_schema", "success": True},
        {"query_id": "Q5", "condition": "traversed", "success": True},
        {"query_id": "Q5", "condition": "no_schema", "success": True},
    ]

    by_query = defaultdict(dict)
    for d in mock_detail:
        qid = d["query_id"]
        cond = d["condition"]
        by_query[qid][cond] = d["success"]

    b = 0  # Full NG, Trav OK
    c = 0  # Full OK, Trav NG
    for qid, results in by_query.items():
        f_ok = results.get("full_schema", False)
        t_ok = results.get("traversed", False)
        if not f_ok and t_ok:
            b += 1
        if f_ok and not t_ok:
            c += 1

    # Q2: Full=F, Trav=T → b++. Q3: Full=T, Trav=F → c++.
    check("モック: b=1（Q2のみ）", b == 1, f"実際: b={b}")
    check("モック: c=1（Q3のみ）", c == 1, f"実際: c={c}")

    # --- テスト3: 条件欠損時の安全性 ---
    print("\n■ 3. 条件欠損時の安全性")

    incomplete_detail = [
        {"query_id": "Q1", "condition": "full_schema", "success": True},
        # Q1のtraversedが欠損
        {"query_id": "Q2", "condition": "traversed", "success": True},
        # Q2のfull_schemaが欠損
    ]
    by_query2 = defaultdict(dict)
    for d in incomplete_detail:
        by_query2[d["query_id"]][d["condition"]] = d["success"]

    b2, c2 = 0, 0
    for qid, results in by_query2.items():
        f_ok = results.get("full_schema", False)
        t_ok = results.get("traversed", False)
        if not f_ok and t_ok:
            b2 += 1
        if f_ok and not t_ok:
            c2 += 1

    # Q1: full=T, trav=missing→False → c++. Q2: full=missing→False, trav=T → b++.
    check("欠損時: b=1, c=1（デフォルトFalse適用）", b2 == 1 and c2 == 1,
          f"実際: b={b2}, c={c2}")

    # --- テスト4: ネスト構造のMcNemar（F: 実データ形式） ---
    print("\n■ 4. ネスト構造でのMcNemar（実データ形式F）")

    nested_detail = [
        {
            "query": {"id": "E001"},
            "llm_full_schema": {"success": True, "rows": 10},
            "llm_traversed": {"success": True, "rows": 10},
            "llm_no_schema": {"success": False, "rows": 0},
        },
        {
            "query": {"id": "E002"},
            "llm_full_schema": {"success": False, "rows": 0},
            "llm_traversed": {"success": True, "rows": 5},
            "llm_no_schema": {"success": False, "rows": 0},
        },
        {
            "query": {"id": "E003"},
            "llm_full_schema": {"success": True, "rows": 3},
            "llm_traversed": {"success": False, "rows": 0},
            "llm_no_schema": {"success": False, "rows": 0},
        },
    ]

    # ネスト構造の判定
    first = nested_detail[0]
    is_nested = "llm_full_schema" in first
    check("ネスト構造判定: llm_full_schema検出", is_nested, f"キー: {list(first.keys())}")

    b_nested, c_nested = 0, 0
    for d in nested_detail:
        f_ok = d.get("llm_full_schema", {}).get("success", False)
        t_ok = d.get("llm_traversed", {}).get("success", False)
        if not f_ok and t_ok:
            b_nested += 1
        if f_ok and not t_ok:
            c_nested += 1

    check("ネスト: b=1 (E002)", b_nested == 1, f"実際: b={b_nested}")
    check("ネスト: c=1 (E003)", c_nested == 1, f"実際: c={c_nested}")

    # 旧コードの誤り再現: detail[i].get("success") → Noneが返る
    old_b, old_c = 0, 0
    for i in range(0, len(nested_detail), 3):
        if i + 1 < len(nested_detail):
            f_ok_old = nested_detail[i].get("success", False)
            t_ok_old = nested_detail[i + 1].get("success", False)
            if not f_ok_old and t_ok_old:
                old_b += 1
            if f_ok_old and not t_ok_old:
                old_c += 1
    check("旧コード: b=0,c=0（バグ: successキーが存在しない）",
          old_b == 0 and old_c == 0,
          f"旧: b={old_b}, c={old_c}")

    # --- テスト5: J — material_entryバイアスの検出 ---
    print("\n■ 5. material_entryバイアス検出（J）")

    def count_uj_raw(tables_used, expected_tables):
        return len(set(tables_used) - set(expected_tables))

    def count_uj_corrected(tables_used, expected_tables):
        extra = set(tables_used) - set(expected_tables)
        extra.discard("material_entry")
        return len(extra)

    # E001: expected=["structure"], used=["material_entry", "structure"]
    raw = count_uj_raw(["material_entry", "structure"], ["structure"])
    corrected = count_uj_corrected(["material_entry", "structure"], ["structure"])
    check("J: 生カウント=1 (material_entry)", raw == 1, f"実際: {raw}")
    check("J: 補正カウント=0 (material_entry除外)", corrected == 0, f"実際: {corrected}")

    # 本当の不要JOIN: expected=["structure"], used=["material_entry", "structure", "band_structure"]
    raw2 = count_uj_raw(["material_entry", "structure", "band_structure"], ["structure"])
    corrected2 = count_uj_corrected(["material_entry", "structure", "band_structure"], ["structure"])
    check("J: 生=2, 補正=1 (band_structureのみ不要)", raw2 == 2 and corrected2 == 1,
          f"生={raw2}, 補正={corrected2}")

    # --- C-2: unnecessary_join_rate計算ロジック ---
    print("\n■ 6. unnecessary_join_rate計算ロジック（C-2）")

    def count_unnecessary_joins(generated_sql: str, expected_tables: list) -> int:
        """生成SQLから不要JOINを数える簡易ロジック"""
        import re
        expected_set = {t.lower() for t in expected_tables}
        expected_set.add("material_entry")
        joins = re.findall(r'\bjoin\s+(\w+)', generated_sql.lower())
        unnecessary = [j for j in joins if j not in expected_set]
        return len(unnecessary)

    # テスト: 不要JOINなし
    sql1 = "SELECT * FROM material_entry m JOIN structure s ON s.entry_id = m.entry_id"
    check("不要JOIN 0件", count_unnecessary_joins(sql1, ["structure"]) == 0,
          f"実際: {count_unnecessary_joins(sql1, ['structure'])}")

    # テスト: 不要JOIN 1件
    sql2 = ("SELECT * FROM material_entry m "
            "JOIN structure s ON s.entry_id = m.entry_id "
            "JOIN band_structure bs ON bs.entry_id = m.entry_id")
    check("不要JOIN 1件（band_structure）",
          count_unnecessary_joins(sql2, ["structure"]) == 1,
          f"実際: {count_unnecessary_joins(sql2, ['structure'])}")

    # テスト: 複数不要JOIN
    sql3 = ("SELECT * FROM material_entry m "
            "JOIN structure s ON s.entry_id = m.entry_id "
            "JOIN elastic_tensor et ON et.entry_id = m.entry_id "
            "JOIN thermal_property tp ON tp.entry_id = m.entry_id")
    check("不要JOIN 2件（elastic_tensor, thermal_property）",
          count_unnecessary_joins(sql3, ["structure"]) == 2,
          f"実際: {count_unnecessary_joins(sql3, ['structure'])}")

    # --- テスト7: 失敗原因帰属分類 ---
    print("\n■ 7. 失敗原因帰属分類テスト")

    def classify_failure(d):
        """Traversal逆効果クエリの失敗原因を分類する。
        Returns: "table_deletion" | "type_mismatch" | "sql_gen_error" | "other"
        """
        t = d.get("llm_traversed", {})
        t_err = t.get("error", "")
        q = d.get("query", {})
        expected = set(q.get("expected_tables", []))
        t_used = set(t.get("tables_used", []))

        if "type" in t_err.lower() or "cast" in t_err.lower() or "integer" in t_err.lower():
            return "type_mismatch"
        if "GROUP BY" in t_err or "aggregate" in t_err.lower():
            return "sql_gen_error"
        missing = expected - t_used - {"material_entry"}
        if missing:
            return "table_deletion"
        if "does not exist" in t_err:
            return "table_deletion"
        return "other"

    # テストケース: テーブル削除
    d1 = {
        "query": {"id": "V_T1", "expected_tables": ["material_entry", "structure", "band_structure"]},
        "llm_full_schema": {"success": True},
        "llm_traversed": {"success": False, "error": "relation does not exist", "tables_used": ["material_entry", "structure"]},
    }
    check("帰属: テーブル削除（band_structure欠落）",
          classify_failure(d1) == "table_deletion", f"実際: {classify_failure(d1)}")

    # テストケース: 型不一致
    d2 = {
        "query": {"id": "V_T2", "expected_tables": ["material_entry", "structure", "space_group"]},
        "llm_full_schema": {"success": True},
        "llm_traversed": {"success": False, "error": "invalid input syntax for type integer", "tables_used": ["material_entry", "structure", "space_group"]},
    }
    check("帰属: 型不一致（integer型エラー）",
          classify_failure(d2) == "type_mismatch", f"実際: {classify_failure(d2)}")

    # テストケース: SQL生成エラー
    d3 = {
        "query": {"id": "V_T3", "expected_tables": ["material_entry", "phase_stability"]},
        "llm_full_schema": {"success": True},
        "llm_traversed": {"success": False, "error": "column must appear in GROUP BY clause", "tables_used": ["material_entry", "phase_stability"]},
    }
    check("帰属: SQL生成エラー（GROUP BY違反）",
          classify_failure(d3) == "sql_gen_error", f"実際: {classify_failure(d3)}")

    # テストケース: テーブル全揃いだがその他のエラー
    d4 = {
        "query": {"id": "V_T4", "expected_tables": ["material_entry", "composition"]},
        "llm_full_schema": {"success": True},
        "llm_traversed": {"success": False, "error": "column reference is ambiguous", "tables_used": ["material_entry", "composition"]},
    }
    check("帰属: その他（カラム曖昧エラー）",
          classify_failure(d4) == "other", f"実際: {classify_failure(d4)}")

    # --- 結果サマリ ---
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  検証結果: {passed}/{total} パス")
    if failed == 0:
        print("  判定: OK — McNemar検定・JOINカウント・帰属分類ロジックは正確")
    else:
        print(f"  判定: NG — {failed}件の不合格あり")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
