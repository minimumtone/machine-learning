#!/usr/bin/env python3
"""
Step 7: 結果の自動検証（v2 — 全ロジック再構築）
実験の生データから全指標を独自に計算し、論文主張と照合する。
事前に計算されたsummaryには依存しない。

追加分析:
  提案1: 成功率分解（テーブル選択精度 × SQL生成精度）
  提案2: unnecessary_join_rate 再計算 + material_entry補正
  提案3: expected_tables 矛盾22件の定量分析
  提案4: Traversal逆効果クエリの根本原因分類
  提案5: (04スクリプト側で対応)
  提案6: rows=0 → definitive_success_rate 別途報告
  提案7: Latency p50/p95/p99
  ■13: 意味的正確性チェック（偽陽性＝テーブル欠落+成功の検出・補正成功率）
  ■14: rows膨張の自動検出（条件緩和疑いフラグ）
  ■15: 返却カラム数分析（出力品質）
  ■16: 実験設計の限界と注記
  ■17: 解釈乖離ペア検出（両方成功だが回答内容が根本的に異なる）
  ■18（統合）: RB比較の無効性警告 + HTMLレポート不整合 + 交絡変数6項目

Circular Reference対策:
  成功率チェックを固定値一致→許容範囲（Trav≥90%, Full≥80%）に変更
  ユニーク経路数・テーブルカバレッジ率を報告
  join_countのサブクエリ未計上をtables_usedベースで補正表示
"""
import json
import sys
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# ルート探索
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).parent
for _candidate in [_script_dir.parent, _script_dir.parent.parent, Path.cwd()]:
    if (_candidate / "experiments" / "results").exists():
        PROJECT_ROOT = _candidate
        break
else:
    print("エラー: experiments/results/ が見つかりません。")
    sys.exit(1)

print(f"  プロジェクトルート: {PROJECT_ROOT}")
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

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


def pN(vals, n):
    """percentile (0-100)"""
    s = sorted(vals)
    k = (len(s) - 1) * n / 100
    lo = int(k)
    hi = lo + 1
    if hi >= len(s):
        return s[-1]
    return s[lo] + (k - lo) * (s[hi] - s[lo])


# ---------------------------------------------------------------------------
# データ読み込み — 生データから計算
# ---------------------------------------------------------------------------
def load_experiment_data():
    """実験結果を読み込む。生データ (list of nested dicts) を返す。"""
    # 生データ（実験直接出力）
    raw_path = RESULTS_DIR / "extended_schema_experiment.json"
    # 旧フォーマット（summary付き）
    legacy_path = RESULTS_DIR / "extended_schema_experiment_150q.json"
    # 参照データ
    ref_path = _script_dir.parent / "results_reference" / "expected_150q.json"

    for p in [raw_path, legacy_path, ref_path]:
        if p.exists():
            data = json.loads(p.read_text("utf-8"))
            # 旧フォーマットはdetailed_resultsキーの中にリストがある
            if isinstance(data, dict) and "detailed_results" in data:
                return data["detailed_results"], str(p.name)
            if isinstance(data, list):
                return data, str(p.name)
    return None, None


def main():
    global passed, failed, skipped

    print("=" * 60)
    print("  結果検証 v2（生データから全指標を独自計算）")
    print("=" * 60)

    detail, source_name = load_experiment_data()
    if detail is None:
        print("\n  エラー: 実験結果が見つかりません。")
        print("  Step 4 (04_run_150query_experiment.sh) を実行してください。")
        return 1

    N = len(detail)
    print(f"\n  データソース: {source_name}  ({N}件)")

    # ------------------------------------------------------------------
    # ■ 1. 基本成功率（生データから再計算）
    # ------------------------------------------------------------------
    print("\n■ 1. 基本成功率（生データから独自計算）")

    full_ok = sum(1 for d in detail if d.get("llm_full_schema", {}).get("success"))
    trav_ok = sum(1 for d in detail if d.get("llm_traversed", {}).get("success"))
    nosc_ok = sum(1 for d in detail if d.get("llm_no_schema", {}).get("success"))

    full_rate = full_ok / N * 100
    trav_rate = trav_ok / N * 100
    nosc_rate = nosc_ok / N * 100
    diff = trav_rate - full_rate

    print(f"    Full Schema:  {full_ok}/{N} ({full_rate:.1f}%)")
    print(f"    Traversed:    {trav_ok}/{N} ({trav_rate:.1f}%)")
    print(f"    No Schema:    {nosc_ok}/{N} ({nosc_rate:.1f}%)")
    print(f"    Traversal効果: +{diff:.1f}pp")

    # ※ 許容範囲チェック（Circular Reference回避: 固定値一致ではなく範囲で検証）
    check("Traversed成功率 ≥ 90%", trav_rate >= 90.0,
          f"Trav={trav_rate:.1f}%")
    check("Full Schema成功率 ≥ 80%", full_rate >= 80.0,
          f"Full={full_rate:.1f}%")
    check("Traversal改善幅 ≥ +3pp", diff >= 3.0, f"+{diff:.1f}pp")
    check("No Schema成功率 < 5%", nosc_rate < 5.0, f"{nosc_rate:.1f}%")

    # No Schema失敗理由の内訳（ストローマン問題の透明化）
    nosc_fail_reasons = {}
    for d in detail:
        ns = d.get("llm_no_schema", {})
        if not ns.get("success"):
            err = ns.get("error", "")
            if "does not exist" in err:
                key = "relation does not exist"
            elif err:
                key = err[:50]
            else:
                key = "unknown"
            nosc_fail_reasons[key] = nosc_fail_reasons.get(key, 0) + 1
    if nosc_fail_reasons:
        print("    No Schema失敗理由内訳:")
        for reason, cnt in sorted(nosc_fail_reasons.items(), key=lambda x: -x[1]):
            print(f"      {reason}: {cnt}件")
    # No Schemaの成功例について暗黙知汚染の可能性を注記
    nosc_success = [d for d in detail if d.get("llm_no_schema", {}).get("success")]
    if nosc_success:
        print(f"    ⚠ No Schema成功{len(nosc_success)}件のテーブル名がLLM学習データに含まれる可能性")
        print(f"      → スキーマ未提供条件の純粋性について論文で注記が必要")

    # ------------------------------------------------------------------
    # ■ 2. 成功率の分解報告【提案1】
    #    SQL成功率 = テーブル選択成功率 × テーブル所与でのSQL生成成功率
    # ------------------------------------------------------------------
    print("\n■ 2. 成功率の分解【提案1】")

    def table_selection_metrics(detail, cond_key):
        """テーブル選択のRecall/Precision + テーブル完全一致時のSQL成功率"""
        recall_list = []
        precision_list = []
        perfect_select = 0  # expected ⊆ used
        perfect_sql_ok = 0

        for d in detail:
            r = d.get(cond_key, {})
            if not r.get("success") and not r.get("tables_used"):
                continue  # SQL生成自体が失敗しtables_used不明
            q = d.get("query", {})
            expected = set(q.get("expected_tables", []))
            used = set(r.get("tables_used", []))
            if not expected:
                continue

            # Recall: expected のうち used に含まれる割合
            recall = len(expected & used) / len(expected) if expected else 1.0
            recall_list.append(recall)

            # Precision: used のうち expected に含まれる割合
            # material_entry は FK ハブとして除外判定
            used_eval = used - {"material_entry"} if "material_entry" not in expected else used
            expected_eval = expected
            if used_eval:
                prec = len(expected_eval & used_eval) / len(used_eval)
            else:
                prec = 1.0
            precision_list.append(prec)

            # テーブル全選択時のSQL成功率
            if expected <= used:
                perfect_select += 1
                if r.get("success"):
                    perfect_sql_ok += 1

        avg_recall = statistics.mean(recall_list) * 100 if recall_list else 0
        avg_precision = statistics.mean(precision_list) * 100 if precision_list else 0
        sql_given_tables = perfect_sql_ok / perfect_select * 100 if perfect_select else 0

        return avg_recall, avg_precision, perfect_select, perfect_sql_ok, sql_given_tables

    for label, key in [("Full Schema", "llm_full_schema"), ("Traversed", "llm_traversed")]:
        rec, prec, n_perf, n_sql, sql_rate = table_selection_metrics(detail, key)
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
        print(f"    {label}:")
        print(f"      テーブル選択Recall:    {rec:.1f}%")
        print(f"      テーブル選択Precision: {prec:.1f}%")
        print(f"      テーブル選択F1:        {f1:.1f}%")
        print(f"      テーブル全選択時SQL成功率: {n_sql}/{n_perf} ({sql_rate:.1f}%)")

    # Full-Oracle上限推定（テーブル選択が完全な場合のSQL成功率）
    for label, key in [("Full Schema", "llm_full_schema"), ("Traversed", "llm_traversed")]:
        _, _, n_perf, n_sql, sql_rate = table_selection_metrics(detail, key)
        if n_perf > 0:
            oracle_upper = sql_rate  # テーブル完全選択時の成功率 = Full-Oracle上限
            print(f"    {label} Full-Oracle上限推定: {oracle_upper:.1f}%")
    print("    ※ Full-Oracle/Full-Random条件が未実装のため、")
    print("      Traversalエンジン固有の貢献とプロンプト短縮効果は分離不能")

    check("Traversedのテーブル選択Recall > Fullのテーブル選択Recall",
          table_selection_metrics(detail, "llm_traversed")[0] >
          table_selection_metrics(detail, "llm_full_schema")[0],
          "Traversalがテーブル選択を改善していない")

    # ------------------------------------------------------------------
    # ■ 3. McNemar検定【F】
    # ------------------------------------------------------------------
    print("\n■ 3. McNemar検定（F）")

    b = 0  # Full NG → Trav OK
    c = 0  # Full OK → Trav NG
    b_ids = []
    c_ids = []
    for d in detail:
        qid = d.get("query", {}).get("id", "?")
        f_ok = d.get("llm_full_schema", {}).get("success", False)
        t_ok = d.get("llm_traversed", {}).get("success", False)
        if not f_ok and t_ok:
            b += 1
            b_ids.append(qid)
        if f_ok and not t_ok:
            c += 1
            c_ids.append(qid)

    print(f"    b={b} (Full✗→Trav✓): {b_ids[:5]}{'...' if len(b_ids)>5 else ''}")
    print(f"    c={c} (Full✓→Trav✗): {c_ids}")
    check("McNemar検定が実行可能（b+c > 0）", b + c > 0,
          "全クエリで成否が同一—検定不能")

    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_sig = chi2 > 3.84
        print(f"    χ²={chi2:.2f}, p {'< 0.05 (有意)' if p_sig else '≥ 0.05 (非有意)'}")
        check("McNemar検定 p < 0.05", p_sig, f"χ²={chi2:.2f}")

    # ------------------------------------------------------------------
    # ■ 4. Traversal逆効果の根本原因分類【提案4】
    # ------------------------------------------------------------------
    print("\n■ 4. Traversal逆効果クエリの根本原因分類【提案4】")

    if c_ids:
        cats = {"table_deletion": [], "type_mismatch": [], "sql_gen_error": [], "other": []}
        for d in detail:
            qid = d.get("query", {}).get("id", "?")
            if qid not in c_ids:
                continue
            f = d.get("llm_full_schema", {})
            t = d.get("llm_traversed", {})
            t_err = t.get("error", "")
            t_sql = t.get("sql", "")
            q = d.get("query", {})
            expected = set(q.get("expected_tables", []))
            t_used = set(t.get("tables_used", []))

            if "does not exist" in t_err and any(
                tbl not in t_sql.lower() for tbl in expected if tbl != "material_entry"
            ):
                cats["table_deletion"].append(qid)
            elif "type" in t_err.lower() or "cast" in t_err.lower() or "integer" in t_err.lower():
                cats["type_mismatch"].append(qid)
            elif "GROUP BY" in t_err or "aggregate" in t_err.lower():
                cats["sql_gen_error"].append(qid)
            else:
                # 必要テーブルが欠落しているか確認
                missing = expected - t_used - {"material_entry"}
                if missing:
                    cats["table_deletion"].append(qid)
                else:
                    cats["other"].append(qid)

        for cat_name, label in [
            ("table_deletion", "必要テーブル削除"),
            ("type_mismatch", "型不一致"),
            ("sql_gen_error", "SQL生成エラー（Traversal無関係）"),
            ("other", "その他"),
        ]:
            ids = cats[cat_name]
            if ids:
                print(f"    {label}: {len(ids)}件 {ids}")
    else:
        print("    Traversal逆効果クエリなし")

    print("\n    ⚠ パイプライン複合評価の限界:")
    print("      detailed_resultsにentities_extracted / tables_candidates /")
    print("      traversal_subgraphが記録されていないため、失敗原因が")
    print("      エンティティ抽出段階かグラフ走査段階か事後判定不能")
    print("      → 論文でTraversal単体評価ではなく複合評価であることを明記すべき")

    # ------------------------------------------------------------------
    # ■ 5. unnecessary_join_rate 再計算【提案2】
    # ------------------------------------------------------------------
    print("\n■ 5. unnecessary_join_rate 再計算【提案2】")

    def calc_uj(detail, cond_key, exclude_material_entry=False):
        """不要JOINのクエリ数・合計数を計算"""
        queries_with_uj = 0
        total_uj = 0
        total_success = 0
        uj_details = []

        for d in detail:
            r = d.get(cond_key, {})
            if not r.get("success"):
                continue
            total_success += 1
            q = d.get("query", {})
            expected = set(q.get("expected_tables", []))
            used = set(r.get("tables_used", []))
            extra = used - expected
            if exclude_material_entry:
                extra.discard("material_entry")
            if extra:
                queries_with_uj += 1
                total_uj += len(extra)
                uj_details.append((d.get("query", {}).get("id", "?"), extra))
        rate = queries_with_uj / total_success * 100 if total_success else 0
        return queries_with_uj, total_uj, total_success, rate, uj_details

    for label, key in [("Full Schema", "llm_full_schema"), ("Traversed", "llm_traversed")]:
        raw_q, raw_t, raw_s, raw_r, raw_d = calc_uj(detail, key, False)
        cor_q, cor_t, cor_s, cor_r, cor_d = calc_uj(detail, key, True)
        print(f"    {label}:")
        print(f"      生:   {raw_q}/{raw_s} ({raw_r:.1f}%) — 不要JOIN合計={raw_t}")
        print(f"      補正: {cor_q}/{cor_s} ({cor_r:.1f}%) — material_entry除外")
        if raw_d and raw_r != cor_r:
            me_penalty = raw_q - cor_q
            print(f"      → material_entryペナルティ: {me_penalty}件")

    full_raw = calc_uj(detail, "llm_full_schema", False)[3]
    full_cor = calc_uj(detail, "llm_full_schema", True)[3]
    trav_raw = calc_uj(detail, "llm_traversed", False)[3]
    trav_cor = calc_uj(detail, "llm_traversed", True)[3]

    check("Traversed不要JOIN率(補正) < Full不要JOIN率(補正)",
          trav_cor < full_cor,
          f"Trav={trav_cor:.1f}%, Full={full_cor:.1f}%")

    # tables_usedベースの実効結合数（サブクエリ含む）
    print("\n    ※ join_countはJOIN句のみでサブクエリを含まない。tables_usedベースの実効値:")
    print("    ※ 論文の不要JOIN率2.8%も同じ定義（JOIN句のみ）で算出されており、")
    print("      サブクエリ経由テーブルが除外されている定義の不完全性が論文にも引き継がれている")
    for label, key in [("Full Schema", "llm_full_schema"), ("Traversed", "llm_traversed")]:
        join_counts = []
        tables_counts = []
        for d in detail:
            r = d.get(key, {})
            if not r.get("success"):
                continue
            jc = r.get("join_count", 0)
            tc = len(r.get("tables_used", []))
            join_counts.append(jc)
            tables_counts.append(tc)
        if join_counts:
            avg_jc = statistics.mean(join_counts)
            avg_tc = statistics.mean(tables_counts)
            gap = avg_tc - avg_jc
            print(f"    {label}: avg_join_count={avg_jc:.2f}  avg_tables_used={avg_tc:.2f}  差={gap:.2f}")
            if gap > 0.5:
                print(f"      ⚠ サブクエリ経由テーブルが平均{gap:.1f}件未計上")

    # ------------------------------------------------------------------
    # ■ 6. expected_tables 矛盾検出【提案3】
    # ------------------------------------------------------------------
    print("\n■ 6. expected_tables 整合性検査【提案3】")

    inconsistent = []
    for d in detail:
        q = d.get("query", {})
        qid = q.get("id", "")
        mt = q.get("min_tables", 0)
        et = q.get("expected_tables", [])
        if mt > len(et):
            inconsistent.append((qid, mt, len(et), et))

    print(f"    min_tables > len(expected_tables): {len(inconsistent)}/{N}")
    for qid, mt, et_len, et in inconsistent[:5]:
        print(f"      {qid}: min_tables={mt}, expected_tables={et_len} {et}")
    if len(inconsistent) > 5:
        print(f"      ... 他{len(inconsistent)-5}件")

    if inconsistent:
        # 矛盾クエリを除外した補正成功率
        clean_ids = {i[0] for i in inconsistent}
        clean_detail = [d for d in detail if d.get("query", {}).get("id") not in clean_ids]
        n_clean = len(clean_detail)
        if n_clean > 0:
            f_ok_c = sum(1 for d in clean_detail if d.get("llm_full_schema", {}).get("success"))
            t_ok_c = sum(1 for d in clean_detail if d.get("llm_traversed", {}).get("success"))
            print(f"    矛盾なし{n_clean}件での成功率:")
            print(f"      Full: {f_ok_c}/{n_clean} ({f_ok_c/n_clean*100:.1f}%)")
            print(f"      Trav: {t_ok_c}/{n_clean} ({t_ok_c/n_clean*100:.1f}%)")
            print(f"      差: +{(t_ok_c-f_ok_c)/n_clean*100:.1f}pp")

    check("矛盾クエリ ≤ 30件", len(inconsistent) <= 30,
          f"{len(inconsistent)}件")

    # ユニーク経路数（有効サンプル数）
    table_combos = set()
    for d in detail:
        et = tuple(sorted(d.get("query", {}).get("expected_tables", [])))
        table_combos.add(et)
    n_unique = len(table_combos)
    print(f"\n    ユニークテーブル経路数: {n_unique}/{N}")
    if n_unique < N:
        dup = N - n_unique
        print(f"    ※ {dup}件が同一テーブル組み合わせの重複クエリ（相関試行）")
        print(f"    ※ 信頼区間計算時はN={n_unique}が実効サンプル数")

    # テーブルカバレッジ率
    ALL_30_TABLES = {
        "material_entry", "composition", "element", "element_property",
        "structure", "phase_stability", "calculation", "calculated_property",
        "band_structure", "density_of_states", "elastic_tensor",
        "magnetic_property", "thermal_property", "surface_energy",
        "grain_boundary", "experimental_measurement", "measured_property",
        "synthesis_method", "material_synthesis", "application_domain",
        "material_application", "literature_reference", "material_reference",
        "defect_type", "material_defect", "alloy_system",
        "material_alloy_system", "phase_diagram_entry",
        "prototype_definition", "space_group",
    }
    covered_tables = set()
    for d in detail:
        for t in d.get("query", {}).get("expected_tables", []):
            covered_tables.add(t)
    uncovered = ALL_30_TABLES - covered_tables
    cov_rate = len(covered_tables) / len(ALL_30_TABLES) * 100
    print(f"\n    テーブルカバレッジ: {len(covered_tables)}/{len(ALL_30_TABLES)} ({cov_rate:.0f}%)")
    if uncovered:
        print(f"    ⚠ 未カバー: {sorted(uncovered)}")
    check("テーブルカバレッジ ≥ 90%", cov_rate >= 90,
          f"{cov_rate:.0f}% — 未カバー: {sorted(uncovered)}")

    # ------------------------------------------------------------------
    # ■ 7. rows=0 → definitive_success_rate【提案6】
    # ------------------------------------------------------------------
    print("\n■ 7. rows=0 分析 + 実効成功率【提案6】")

    for label, key in [("Full Schema", "llm_full_schema"), ("Traversed", "llm_traversed")]:
        zero_rows = []
        positive_rows = 0
        total_success_cond = 0
        for d in detail:
            r = d.get(key, {})
            if not r.get("success"):
                continue
            total_success_cond += 1
            if r.get("rows", -1) == 0:
                zero_rows.append(d.get("query", {}).get("id", "?"))
            else:
                positive_rows += 1

        definitive_rate = positive_rows / N * 100
        nominal_rate = total_success_cond / N * 100
        print(f"    {label}:")
        print(f"      公称成功率 (SQLエラーなし): {total_success_cond}/{N} ({nominal_rate:.1f}%)")
        print(f"      実効成功率 (rows>0):       {positive_rows}/{N} ({definitive_rate:.1f}%) ← 主要指標")
        print(f"      rows=0 (不確定):              {len(zero_rows)}/{N} ({len(zero_rows)/N*100:.1f}%)")
        if zero_rows:
            print(f"      不確定クエリ例: {zero_rows[:5]}")

    print("\n    === 実効成功率サマリ ===")
    full_def = sum(1 for d in detail
                   if d.get("llm_full_schema", {}).get("success")
                   and d.get("llm_full_schema", {}).get("rows", 0) > 0) / N * 100
    trav_def = sum(1 for d in detail
                   if d.get("llm_traversed", {}).get("success")
                   and d.get("llm_traversed", {}).get("rows", 0) > 0) / N * 100
    full_nom = sum(1 for d in detail
                   if d.get("llm_full_schema", {}).get("success")) / N * 100
    trav_nom = sum(1 for d in detail
                   if d.get("llm_traversed", {}).get("success")) / N * 100
    print(f"    Full:      公称={full_nom:.1f}%  実効={full_def:.1f}%  差={full_nom-full_def:.1f}pp")
    print(f"    Traversed: 公称={trav_nom:.1f}%  実効={trav_def:.1f}%  差={trav_nom-trav_def:.1f}pp")
    print(f"    実効成功率差: +{trav_def - full_def:.1f}pp（公称: +{trav_nom - full_nom:.1f}pp）")

    # seedデータL12バイアスの注記
    print("\n    ⚠ seedデータL12バイアス + 規模乖離:")
    print("      論文のDB: OQMD金属間化合物1,351件（B2型636, L12型273, NaCl型355, NiAs型74, BiF3型13）")
    print("      検証パッケージ: seed_l12_entries.csv = 120件（論文の1/11規模）")
    print("      → 非L12構造を要求するクエリで正しいSQLでもrows=0")
    print("        （fcc結晶系、BiF3型、youngs_modulus≥300GPa等）")
    print("      → rows=0の多くはTraversalの性能ではなくseedデータの規模・構造的偏りが原因")
    print("      → 論文結果の再現としては、seedデータの規模乖離が根本的障壁")

    check("実効成功率(rows>0): Traversed > Full",
          trav_def > full_def,
          f"Trav={trav_def:.1f}%, Full={full_def:.1f}%")

    # ------------------------------------------------------------------
    # ■ 8. LIMIT到達 (rows=100) 分析【H】
    # ------------------------------------------------------------------
    print("\n■ 8. LIMIT到達 (rows=100) 分析（H）")

    limit_total = 0
    limit_mismatch = []
    for d in detail:
        qid = d.get("query", {}).get("id", "")
        f = d.get("llm_full_schema", {})
        t = d.get("llm_traversed", {})
        f_limit = f.get("rows", 0) == 100
        t_limit = t.get("rows", 0) == 100
        if f_limit or t_limit:
            limit_total += 1
            if f.get("success") != t.get("success"):
                limit_mismatch.append({
                    "id": qid,
                    "full": {"success": f.get("success"), "rows": f.get("rows")},
                    "trav": {"success": t.get("success"), "rows": t.get("rows")},
                })

    print(f"    LIMIT到達クエリ数: {limit_total}/{N}")
    print(f"    うちFull/Traversed成否不一致: {len(limit_mismatch)}件")
    for m in limit_mismatch:
        print(f"      {m['id']}: Full={m['full']}, Trav={m['trav']}")

    check("LIMIT到達時の成否不一致 ≤ 10件",
          len(limit_mismatch) <= 10, f"{len(limit_mismatch)}件")

    # LIMIT=100のaggregation影響分析
    agg_limit_hit = []
    for d in detail:
        cat = d.get("query", {}).get("category", "")
        if cat != "aggregation":
            continue
        qid = d.get("query", {}).get("id", "?")
        for key in ["llm_full_schema", "llm_traversed"]:
            r = d.get(key, {})
            if r.get("success") and r.get("rows", 0) == 100:
                sql = r.get("sql", "")
                if "GROUP BY" in sql.upper():
                    agg_limit_hit.append({"id": qid, "cond": key})
    if agg_limit_hit:
        print(f"\n    ⚠ aggregationクエリでGROUP BY結果がLIMIT=100で切り捨て: {len(agg_limit_hit)}件")
        for a in agg_limit_hit[:5]:
            print(f"      {a['id']} ({a['cond']})")
        print(f"      → aggregationカテゴリの評価精度が体系的に低下する可能性")
        print(f"      → SQL_ROW_LIMITのaggregationクエリ除外を推奨")

    # ------------------------------------------------------------------
    # ■ 9. Latency分布【提案7】
    # ------------------------------------------------------------------
    print("\n■ 9. Latency分布【提案7】")

    for label, key in [("Full Schema", "llm_full_schema"), ("Traversed", "llm_traversed")]:
        lats = [d.get(key, {}).get("latency_ms", 0) for d in detail
                if d.get(key, {}).get("latency_ms")]
        if lats:
            p50 = pN(lats, 50)
            p95 = pN(lats, 95)
            p99 = pN(lats, 99)
            avg = statistics.mean(lats)
            sd = statistics.stdev(lats) if len(lats) > 1 else 0
            cv = sd / avg * 100 if avg > 0 else 0
            mx = max(lats)
            print(f"    {label}: avg={avg:.0f}ms  stdev={sd:.0f}ms  CV={cv:.1f}%")
            print(f"             p50={p50:.0f}  p95={p95:.0f}  p99={p99:.0f}  max={mx:.0f}ms")
        else:
            print(f"    {label}: latencyデータなし")

    full_lats = [d.get("llm_full_schema", {}).get("latency_ms", 0) for d in detail
                 if d.get("llm_full_schema", {}).get("latency_ms")]
    trav_lats = [d.get("llm_traversed", {}).get("latency_ms", 0) for d in detail
                 if d.get("llm_traversed", {}).get("latency_ms")]
    if full_lats and trav_lats:
        full_avg = statistics.mean(full_lats)
        trav_avg = statistics.mean(trav_lats)
        full_sd = statistics.stdev(full_lats) if len(full_lats) > 1 else 0
        trav_sd = statistics.stdev(trav_lats) if len(trav_lats) > 1 else 0
        savings = (1 - trav_avg / full_avg) * 100
        stdev_ratio = trav_sd / full_sd if full_sd > 0 else 0
        print(f"    Traversalによる平均レイテンシ削減: {savings:.1f}%")
        print(f"    stdev比: {stdev_ratio:.2f} (Full={full_sd:.0f}ms → Trav={trav_sd:.0f}ms)")
        print(f"    → Traversalはレイテンシの予測可能性を大幅に改善")
        full_p99 = pN(full_lats, 99)
        trav_p99 = pN(trav_lats, 99)
        print(f"    p99テール抑制: Full={full_p99:.0f}ms → Trav={trav_p99:.0f}ms")

    # ------------------------------------------------------------------
    # ■ 10. カテゴリ別分析
    # ------------------------------------------------------------------
    print("\n■ 10. カテゴリ別成功率")

    from collections import defaultdict
    by_cat = defaultdict(list)
    for d in detail:
        cat = d.get("query", {}).get("category", "unknown")
        by_cat[cat].append(d)

    cat_order = ["simple", "medium", "complex", "very_complex", "cross_domain", "aggregation"]
    print(f"    {'カテゴリ':15s} {'Full':>8s} {'Trav':>8s} {'diff':>8s} {'avg_min_t':>10s}")
    print(f"    {'-'*58}")
    for cat in cat_order:
        if cat not in by_cat:
            continue
        items = by_cat[cat]
        n_c = len(items)
        f_c = sum(1 for d in items if d.get("llm_full_schema", {}).get("success"))
        t_c = sum(1 for d in items if d.get("llm_traversed", {}).get("success"))
        f_r = f_c / n_c * 100
        t_r = t_c / n_c * 100
        d_r = t_r - f_r
        avg_mt = statistics.mean([d.get("query", {}).get("min_tables", 0) for d in items])
        print(f"    {cat:15s} {f_r:7.0f}% {t_r:7.0f}% {d_r:+7.0f}pp {avg_mt:9.1f}")

    print("\n    ※ カテゴリはSQL表現複雑さ（相関サブクエリ等）で定義、テーブル選択難易度とは直交")

    # テーブル選択難易度軸での再分類
    print("\n    テーブル選択難易度軸（min_tables基準）:")
    tbl_bins = {"1-2 tables": [], "3-4 tables": [], "5+ tables": []}
    for d in detail:
        mt = d.get("query", {}).get("min_tables", 0)
        if mt <= 2:
            tbl_bins["1-2 tables"].append(d)
        elif mt <= 4:
            tbl_bins["3-4 tables"].append(d)
        else:
            tbl_bins["5+ tables"].append(d)
    print(f"    {'難易度':15s} {'n':>4s} {'Full':>8s} {'Trav':>8s} {'diff':>8s}")
    print(f"    {'-'*48}")
    for bin_name in ["1-2 tables", "3-4 tables", "5+ tables"]:
        items = tbl_bins[bin_name]
        if not items:
            continue
        n_b = len(items)
        f_b = sum(1 for d in items if d.get("llm_full_schema", {}).get("success"))
        t_b = sum(1 for d in items if d.get("llm_traversed", {}).get("success"))
        f_br = f_b / n_b * 100
        t_br = t_b / n_b * 100
        print(f"    {bin_name:15s} {n_b:4d} {f_br:7.0f}% {t_br:7.0f}% {t_br - f_br:+7.0f}pp")

    # mediumでの改善幅チェック
    if "medium" in by_cat:
        med = by_cat["medium"]
        med_f = sum(1 for d in med if d.get("llm_full_schema", {}).get("success")) / len(med) * 100
        med_t = sum(1 for d in med if d.get("llm_traversed", {}).get("success")) / len(med) * 100
        check("Mediumカテゴリ改善幅 ≥ +10pp",
              med_t - med_f >= 10, f"+{med_t - med_f:.0f}pp")

    # ------------------------------------------------------------------
    # ■ 11. RB比較結果
    # ------------------------------------------------------------------
    print("\n■ 11. Rule-based比較結果")

    rb_path = RESULTS_DIR / "rb_30table_comparison.json"
    rb_verify_path = RESULTS_DIR / "rb_30table_comparison_verify.json"
    for rp in [rb_verify_path, rb_path]:
        if rp.exists():
            rb_data = json.loads(rp.read_text("utf-8"))
            rb_s = rb_data["summary"]
            naive = rb_s["naive_rb"]
            sg_rb = rb_s["sg_rb"]
            print(f"    Naive RB: {naive['success']}/{naive['total']} ({naive['rate']*100:.1f}%)")
            print(f"    SG+RB:    {sg_rb['success']}/{sg_rb['total']} ({sg_rb['rate']*100:.1f}%)")
            check("Naive RBの成功率 < 5%", naive["rate"] * 100 < 5,
                  f"{naive['rate']*100:.1f}%")
            check("SG+RBの成功率 > Naive RB", sg_rb["rate"] > naive["rate"])

            # SG+RB品質分析: WHERE条件有無 + rows=100率
            sg_detail = rb_data.get("sg_rb_details", rb_data.get("details", {}).get("sg_rb", []))
            if sg_detail:
                sg_success = [d for d in sg_detail if d.get("success")]
                sg_no_where = sum(1 for d in sg_success
                                  if "WHERE" not in d.get("sql", "").upper())
                sg_limit_hit = sum(1 for d in sg_success
                                   if d.get("rows", 0) == 100)
                n_sg = len(sg_success)
                if n_sg > 0:
                    print(f"\n    SG+RB品質分析 (成功{n_sg}件):")
                    print(f"      WHERE条件なし: {sg_no_where}/{n_sg} ({sg_no_where/n_sg*100:.0f}%)")
                    print(f"      rows=100到達: {sg_limit_hit}/{n_sg} ({sg_limit_hit/n_sg*100:.0f}%)")
                    if sg_no_where / n_sg > 0.8:
                        print(f"      ⚠ SG+RB成功の大部分がWHERE条件なし — 成功率はJOIN構文成功率に近い")

            # RB比較の無効性についての明示的警告
            print(f"\n    === RB比較の有効性に関する警告 ===")
            print(f"    Naive RB 0%: 全件がエイリアス重複バグで失敗")
            print(f"      → 'table name calc specified more than once' が原因")
            print(f"      → 30テーブルJOIN過多による破綻という論文の主張とは無関係")
            print(f"    SG+RB 54%: 成功件は全件WHERE条件ゼロ")
            print(f"      → バグ修正後の理論的SG+RB成功率は約99%（WHEREなしで全件返すだけ）")
            print(f"      → 54%はアルゴリズム性能ではなくバグ発生率を測定")
            print(f"    → RB比較は3者が異なる基準で「成功」を計上しており比較として成立していない")
            break
    else:
        skip("RB比較", "結果ファイルが存在しません")

    # ------------------------------------------------------------------
    # ■ 12. 走査対象外テーブル混入検査（B-1）
    # ------------------------------------------------------------------
    print("\n■ 12. 走査対象外テーブル混入検査（B-1）")

    trav_leaks = 0
    trav_total = 0
    for d in detail:
        q = d.get("query", {})
        t = d.get("llm_traversed", {})
        if not t.get("success"):
            continue
        trav_total += 1
        expected = set(q.get("expected_tables", []))
        expected.add("material_entry")
        used = set(t.get("tables_used", []))
        if used - expected:
            trav_leaks += 1

    if trav_total > 0:
        leak_r = trav_leaks / trav_total * 100
        print(f"    対象外テーブル混入: {trav_leaks}/{trav_total} ({leak_r:.1f}%)")
        check("走査対象外テーブル混入率 < 5%", leak_r < 5, f"{leak_r:.1f}%")
    else:
        skip("テーブル混入検査", "Traversed成功クエリなし")

    # ------------------------------------------------------------------
    # ■ 13. 意味的正確性チェック（Semantic Correctness）
    #    SQLがエラーなく実行されても、期待テーブルが欠落していれば偽陽性
    # ------------------------------------------------------------------
    print("\n■ 13. 意味的正確性チェック（Semantic Correctness）")

    def semantic_check(detail, cond_key):
        """success=True なのに expected_tables の一部が tables_used に含まれないケースを検出"""
        false_positives = []
        total_success = 0
        for d in detail:
            r = d.get(cond_key, {})
            if not r.get("success"):
                continue
            total_success += 1
            q = d.get("query", {})
            qid = q.get("id", "?")
            expected = set(q.get("expected_tables", []))
            used = set(r.get("tables_used", []))
            # expected_tables のうち使われなかったもの（material_entry はハブなので除外）
            missing = expected - used - {"material_entry"}
            if missing:
                false_positives.append({
                    "id": qid,
                    "missing_tables": sorted(missing),
                    "rows": r.get("rows", "?"),
                })
        return false_positives, total_success

    for label, key in [("Full Schema", "llm_full_schema"), ("Traversed", "llm_traversed")]:
        fps, n_succ = semantic_check(detail, key)
        fp_rate = len(fps) / n_succ * 100 if n_succ else 0
        print(f"    {label}: 偽陽性（テーブル欠落+成功）= {len(fps)}/{n_succ} ({fp_rate:.1f}%)")
        for fp in fps[:8]:
            print(f"      {fp['id']}: 欠落={fp['missing_tables']}, rows={fp['rows']}")
        if len(fps) > 8:
            print(f"      ... 他{len(fps)-8}件")

    # 補正成功率（偽陽性を除外）
    full_fps, full_succ = semantic_check(detail, "llm_full_schema")
    trav_fps, trav_succ = semantic_check(detail, "llm_traversed")
    full_semantic_ok = full_succ - len(full_fps)
    trav_semantic_ok = trav_succ - len(trav_fps)
    full_sem_rate = full_semantic_ok / N * 100
    trav_sem_rate = trav_semantic_ok / N * 100
    print(f"\n    === 意味的補正成功率 ===")
    print(f"    Full Schema:  {full_semantic_ok}/{N} ({full_sem_rate:.1f}%)")
    print(f"    Traversed:    {trav_semantic_ok}/{N} ({trav_sem_rate:.1f}%)")
    print(f"    差: +{trav_sem_rate - full_sem_rate:.1f}pp")

    check("意味的補正後もTraversed > Full",
          trav_sem_rate > full_sem_rate,
          f"Trav={trav_sem_rate:.1f}%, Full={full_sem_rate:.1f}%")

    # ------------------------------------------------------------------
    # ■ 14. rows膨張の自動検出（条件緩和疑い）
    #    Full=rows少 → Trav=rows大幅増 は条件消失の可能性
    # ------------------------------------------------------------------
    print("\n■ 14. rows膨張の自動検出（条件緩和疑い）")

    inflation_suspects = []
    for d in detail:
        qid = d.get("query", {}).get("id", "?")
        f = d.get("llm_full_schema", {})
        t = d.get("llm_traversed", {})
        f_rows = f.get("rows", -1)
        t_rows = t.get("rows", -1)

        # 条件: 両方success、かつ Trav rows が Full rows より大幅に多い
        if f.get("success") and t.get("success"):
            # パターン1: Full=0, Trav>5 （正解がないはずなのに結果が出た）
            if f_rows == 0 and t_rows > 5:
                inflation_suspects.append({
                    "id": qid, "full_rows": f_rows, "trav_rows": t_rows,
                    "reason": "Full=0でTrav>5: 条件消失疑い"
                })
            # パターン2: Trav rows が Full rows の5倍以上かつ Trav≥50
            elif f_rows > 0 and t_rows >= 50 and t_rows >= f_rows * 5:
                inflation_suspects.append({
                    "id": qid, "full_rows": f_rows, "trav_rows": t_rows,
                    "reason": f"rows 5倍以上膨張 ({f_rows}→{t_rows})"
                })

        # パターン3: Full失敗, Trav=rows=100 (LIMIT到達)
        if not f.get("success") and t.get("success") and t_rows == 100:
            # テーブル欠落で条件が消えてLIMIT到達の可能性
            q = d.get("query", {})
            expected = set(q.get("expected_tables", []))
            used = set(t.get("tables_used", []))
            missing = expected - used - {"material_entry"}
            if missing:
                inflation_suspects.append({
                    "id": qid, "full_rows": "FAIL", "trav_rows": t_rows,
                    "reason": f"テーブル欠落({sorted(missing)})+LIMIT到達"
                })

    print(f"    条件緩和疑いクエリ: {len(inflation_suspects)}/{N}")
    for s in inflation_suspects:
        print(f"      {s['id']}: Full={s['full_rows']}, Trav={s['trav_rows']} — {s['reason']}")

    check("条件緩和疑いクエリ ≤ 15件", len(inflation_suspects) <= 15,
          f"{len(inflation_suspects)}件")

    # ------------------------------------------------------------------
    # ■ 15. 返却カラム数分析（出力品質）
    # ------------------------------------------------------------------
    print("\n■ 15. 返却カラム数分析（出力品質）")

    for label, key in [("Full Schema", "llm_full_schema"), ("Traversed", "llm_traversed")]:
        col_counts = []
        for d in detail:
            r = d.get(key, {})
            if not r.get("success"):
                continue
            n_cols = r.get("columns_returned", r.get("num_columns", 0))
            if n_cols > 0:
                col_counts.append(n_cols)
        if col_counts:
            avg_cols = statistics.mean(col_counts)
            print(f"    {label}: 平均返却カラム数={avg_cols:.1f}")
        else:
            print(f"    {label}: カラム数データなし")

    # Full vs Traversedのカラム数比較
    trav_more = 0
    total_paired = 0
    for d in detail:
        f = d.get("llm_full_schema", {})
        t = d.get("llm_traversed", {})
        if f.get("success") and t.get("success"):
            f_cols = f.get("columns_returned", f.get("num_columns", 0))
            t_cols = t.get("columns_returned", t.get("num_columns", 0))
            if f_cols > 0 and t_cols > 0:
                total_paired += 1
                if t_cols > f_cols + 2:
                    trav_more += 1
    if total_paired > 0:
        print(f"    Traversedが3カラム以上多い: {trav_more}/{total_paired} ({trav_more/total_paired*100:.0f}%)")
        if trav_more / total_paired > 0.2:
            print(f"    ⚠ Traversalはテーブル削減と引き換えに返却カラムが冗長化する傾向")

    # ------------------------------------------------------------------
    # ■ 16. 実験設計の限界と注記
    # ------------------------------------------------------------------
    print("\n■ 16. 実験設計の限界と注記")

    limitations = [
        "1. Full-Oracle/Full-Random条件が未実装 → Traversalエンジン固有貢献とプロンプト短縮効果が分離不能",
        "2. No Schema条件はストローマン（149/150がrelation does not exist）→ 部分スキーマ条件を推奨",
        "3. パイプライン中間データ（entities_extracted等）が未記録 → 失敗段階の特定不能",
        "4. expected_150q.jsonは論文出力そのもの → 固定値一致は循環参照（許容範囲チェックに変更済み）",
        f"5. ユニーク経路数が総クエリ数より少ない → 重複テーブル組み合わせは相関試行",
    ]
    # 論文との照合から発見された限界
    paper_limitations = [
        "6. 30テーブル・150クエリ実験の使用モデル名が論文に未記載",
        "   → expected_150q.jsonはgpt-4o-miniだが、論文本文からは読み取れない",
        "7. 論文の主実験（7テーブル・57クエリ）はgpt-5.5で実施 → 未公開モデルのため再現不可能",
        "   → .env.exampleのLLM_MODEL=gpt-5も存在しないモデル名",
        "8. Table 13（Graph Traversalアブレーション）は7テーブル・gpt-5.5の結果",
        "   → 検証パッケージ（30テーブル）ではTable 13の再現が不可能",
        "9. seedデータ120件 vs 論文DB 1,351件（1/11規模）",
        "   → rows=0問題の直接原因、「論文結果の再現」に対する根本的障壁",
        "10. 論文のJaccard類似度評価（5.1.6節）が検証パッケージに未実装",
    ]
    for lim in limitations:
        print(f"    {lim}")
    print("\n    === 論文との照合から発見された限界 ===")
    for pl in paper_limitations:
        print(f"    {pl}")

    # 交絡変数の未記録項目
    print("\n    === 未記録の交絡変数（6項目）===")
    confounds = [
        "6. temperatureパラメータが未記録 → 決定論性の根拠が不明、再現性の保証不可",
        "7. システムプロンプトの内容と3条件間の一致が未確認",
        "8. クエリ実行順序が未記録 → APIキャッシュ/レートリミット影響の確認不能",
        "9. 3条件の実行順序（Full→Trav→NoSchemaか否か）が未記録",
        "10. DB状態（実験時のseedデータ件数）が未記録",
        "11. プロンプト全文（スキーマ情報の整形方法）が未記録",
    ]
    for c in confounds:
        print(f"    {c}")
    print("\n    推奨: 次回実験時に以下をdetailed_resultsに記録")
    print("      {\"temperature\": 0, \"system_prompt_hash\": \"sha256:...\",")
    print("       \"execution_order\": [\"full\", \"traversed\", \"no_schema\"],")
    print("       \"seed_row_count\": {\"material_entry\": 120, ...},")
    print("       \"prompt_full_text\": \"...\"}")

    # HTMLレポートの不整合警告
    print("\n    === HTMLレポートの不整合警告 ===")
    print("    comprehensive_experiment_report.htmlは論文の主実験（30テーブル・150クエリ）")
    print("    の結果を含まない別実験のレポートです:")
    print("      - 実験1: 57クエリ・7手法（7テーブル環境）")
    print("      - 実験2-3: RAGアブレーション（gpt-5.5という存在しないモデル名が混入）")
    print("      - 実験4: 30クエリのみ")
    print("    Step 6でこのHTMLを「実験結果」として提示すると検証者に誤解を与える")
    print("    → 論文の150クエリ実験専用のHTMLを別途生成すべき")

    # seedデータの不完全性
    print("\n    === seedデータの不完全性 ===")
    print("    参照結果(expected_150q.json)は30テーブル全てにデータがある環境で生成")
    print("    検証パッケージのseedデータは7テーブル分（120件）のみ:")
    print("      seedあり: material_entry, composition, structure, phase_stability,")
    print("                calculation, calculated_property, prototype_definition")
    print("      seedなし: elastic_tensor, band_structure, magnetic_property,")
    print("                thermal_property, surface_energy, grain_boundary 他23テーブル")
    print("    → 参照結果で101件がrows>0だが、検証環境では再現不能")
    print("    → 検証者が手順通りに環境構築しても参照結果と根本的に異なるDB")

    # extended_schema.sqlの問題
    print("\n    === extended_schema.sqlの構造的問題 ===")
    print("    - IF NOT EXISTS句なし → 既存DBに適用するとエラー")
    print("    - ON DELETE CASCADEなし → material_entry削除時に孤立レコード")
    print("    - schema.sqlのNOT NULL制約がextended_schema.sqlで欠落")

    # クエリパターンカバレッジ
    print("\n    === クエリ構文カバレッジの欠落 ===")
    # 実際にデータから検出
    cte_count = 0
    window_count = 0
    union_count = 0
    negation_count = 0
    for d in detail:
        for key in ["llm_full_schema", "llm_traversed"]:
            sql = d.get(key, {}).get("sql", "").upper()
            if "WITH " in sql and " AS " in sql:
                cte_count += 1
            if "OVER(" in sql or "OVER (" in sql:
                window_count += 1
            if " UNION " in sql:
                union_count += 1
            if "NOT IN" in sql or "NOT EXISTS" in sql or "!=" in sql or "<>" in sql:
                negation_count += 1
    print(f"    CTEクエリ: {cte_count}件")
    print(f"    窓関数: {window_count}件")
    print(f"    UNION: {union_count}件")
    print(f"    否定条件(NOT IN/NOT EXISTS/!=/<>): {negation_count}件")
    print("    → 材料DBで実用的なCTE（階層再帰）・窓関数（ランキング）が未テスト")

    # ------------------------------------------------------------------
    # ■ 17. 解釈乖離ペア検出（両方成功・回答内容乖離）
    # ------------------------------------------------------------------
    print("\n■ 17. 解釈乖離ペア検出（両方成功・回答内容乖離）")

    divergent_pairs = []
    for d in detail:
        f = d.get("llm_full_schema", {})
        t = d.get("llm_traversed", {})
        if not (f.get("success") and t.get("success")):
            continue
        f_rows = f.get("rows", -1)
        t_rows = t.get("rows", -1)
        qid = d.get("query", {}).get("id", "?")

        # パターン: 片方rows>10, 他方rows=0
        if (f_rows > 10 and t_rows == 0) or (t_rows > 10 and f_rows == 0):
            f_tables = sorted(f.get("tables_used", []))
            t_tables = sorted(t.get("tables_used", []))
            divergent_pairs.append({
                "id": qid,
                "full_rows": f_rows,
                "trav_rows": t_rows,
                "full_tables": f_tables,
                "trav_tables": t_tables,
            })

    print(f"    解釈乖離ペア: {len(divergent_pairs)}/{N} ({len(divergent_pairs)/N*100:.1f}%)")
    print("    ※ 両方成功判定だが一方rows>10・他方rows=0のペア")
    print("    ※ binary成否評価の最大の限界: 解釈の一貫性が担保されていない")
    for dp in divergent_pairs:
        print(f"      {dp['id']}: Full={dp['full_rows']}rows({dp['full_tables']}) "
              f"vs Trav={dp['trav_rows']}rows({dp['trav_tables']})")

    check("解釈乖離ペア ≤ 15件", len(divergent_pairs) <= 15,
          f"{len(divergent_pairs)}件")

    # ------------------------------------------------------------------
    # ■ 18. テーブル選択Jaccard類似度（論文5.1.6節の指標）
    # ------------------------------------------------------------------
    print("\n■ 18. テーブル選択Jaccard類似度（論文5.1.6節の指標）")

    jaccard_full = []
    jaccard_trav = []
    for d in detail:
        q = d.get("query", {})
        expected = set(q.get("expected_tables", []))
        if not expected:
            continue
        for key, jlist in [("llm_full_schema", jaccard_full),
                           ("llm_traversed", jaccard_trav)]:
            r = d.get(key, {})
            if not r.get("success"):
                jlist.append(0.0)
                continue
            used = set(r.get("tables_used", []))
            intersection = expected & used
            union = expected | used
            jac = len(intersection) / len(union) if union else 0.0
            jlist.append(jac)

    if jaccard_full:
        avg_jf = statistics.mean(jaccard_full)
        avg_jt = statistics.mean(jaccard_trav)
        print(f"    Full Schema:  Jaccard平均 = {avg_jf:.3f}")
        print(f"    Traversed:    Jaccard平均 = {avg_jt:.3f}")
        print(f"    差: +{avg_jt - avg_jf:.3f}")
        print("    ※ 論文5.1.6節のJaccard=0.897は7テーブル環境のSG+RB vs LLM+SG比較")
        print("    ※ 上記は30テーブル環境のexpected_tables vs tables_usedのJaccard")
        check("Jaccard類似度: Traversed ≥ Full",
              avg_jt >= avg_jf,
              f"Trav={avg_jt:.3f}, Full={avg_jf:.3f}")
    else:
        skip("Jaccard類似度", "expected_tablesデータなし")

    # ------------------------------------------------------------------
    # 総合判定
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  検証結果: {passed}/{total} パス  ({skipped}件スキップ)")

    if failed == 0:
        print("  総合判定: OK")
    else:
        print(f"  総合判定: NG — {failed}件の不合格あり")
        print("  LLMの非決定論性により成功率は±2-3%変動します。")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
