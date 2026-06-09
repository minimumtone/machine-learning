#!/usr/bin/env python3
"""
Step 3: Schema Graph走査エンジンの検証
APIキー不要 — DBに対してFK関係を取得し、グラフ構築・走査を検証する。
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
# scripts/ から実行する場合と、リポジトリルートから実行する場合の両方に対応
_script_dir = Path(__file__).parent
if (_script_dir.parent / "graph").exists():
    PROJECT_ROOT = _script_dir.parent
elif (_script_dir.parent.parent / "graph").exists():
    PROJECT_ROOT = _script_dir.parent.parent
elif Path.cwd().joinpath("graph").exists():
    PROJECT_ROOT = Path.cwd()
else:
    print("エラー: リポジトリルート（graph/ ディレクトリを含む場所）から実行してください。")
    sys.exit(1)
print(f"  プロジェクトルート: {PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg
import networkx as nx

from graph.schema_parser import get_tables, get_columns, get_foreign_keys, introspect_schema
from graph.graph_builder import build_schema_graph, build_table_graph, schema_graph_summary
from graph.traversal_engine import find_shortest_table_path, find_join_subgraph, extract_join_edges

DB_CONFIG = {
    "dbname": "l12_materials",
    "user": "l12_user",
    "password": "l12_password",
    "host": "localhost",
    "port": 5432,
}

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


def main():
    global passed, failed

    print("=" * 60)
    print("  Schema Graph 走査エンジン検証")
    print("=" * 60)

    # --- 1. DB接続 & イントロスペクション ---
    print("\n■ 1. PostgreSQLイントロスペクション")

    conninfo = (
        f"dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} "
        f"password={DB_CONFIG['password']} host={DB_CONFIG['host']} "
        f"port={DB_CONFIG['port']}"
    )
    conn = psycopg.connect(conninfo)

    tables = get_tables(conn)
    check("テーブル数 = 30", len(tables) == 30, f"実際: {len(tables)}")

    fks = get_foreign_keys(conn)
    # DB上のFK数は34（自己参照やcalculation経由の間接FKを含む）
    # 論文では主要31FK関係として記述
    check("FK関係数 ≥ 30", len(fks) >= 30, f"実際: {len(fks)}")
    print(f"    → FK関係数（実値）: {len(fks)}")

    # material_entryへのFK数を確認（ハブテーブル）
    fk_to_entry = [fk for fk in fks if fk.target_table == "material_entry"]
    check(
        "material_entryへのFK ≥ 15（ハブテーブル）",
        len(fk_to_entry) >= 15,
        f"実際: {len(fk_to_entry)}",
    )

    # --- 2. Schema Graph構築 ---
    print("\n■ 2. Schema Graph構築")

    schema_info = introspect_schema(conn)
    columns = schema_info["columns"]

    schema_graph = build_schema_graph(tables, columns, fks)
    summary = schema_graph_summary(schema_graph)
    check("グラフノード数 > 200", summary["num_tables"] + summary["num_columns"] > 200,
          f"tables={summary['num_tables']}, columns={summary['num_columns']}")
    check("テーブルノード = 30", summary["num_tables"] == 30,
          f"実際: {summary['num_tables']}")

    table_graph = build_table_graph(fks)
    n_nodes = table_graph.number_of_nodes()
    n_edges = table_graph.number_of_edges()
    # FK接続のないテーブル（prototype_definition, space_group）はグラフに含まれない
    check("テーブルグラフノード数 ≥ 25", n_nodes >= 25,
          f"実際: {n_nodes}")
    print(f"    → テーブルグラフ: {n_nodes}ノード, {n_edges}エッジ")

    # FK接続のないテーブルを表示
    tables_in_graph = set(table_graph.nodes())
    isolated = set(tables) - tables_in_graph
    if isolated:
        print(f"    → FK接続なしテーブル（グラフ外）: {isolated}")

    if n_nodes == 0:
        check("テーブルグラフは連結", False, "グラフが空です")
    else:
        check("テーブルグラフは連結", nx.is_connected(table_graph),
              "非連結 — FK関係が不足している可能性")

    # --- 3. 最短経路探索 ---
    print("\n■ 3. 最短JOINパス探索")

    # テスト1: composition → phase_stability (material_entry経由)
    path1 = find_shortest_table_path(table_graph, "composition", "phase_stability")
    check(
        "composition → phase_stability パス",
        len(path1) == 3 and "material_entry" in path1,
        f"実際: {path1}",
    )

    # テスト2: measured_property → structure (2ホップ以上)
    path2 = find_shortest_table_path(table_graph, "measured_property", "structure")
    check(
        "measured_property → structure パス存在",
        len(path2) >= 3,
        f"実際: {path2}",
    )

    # テスト3: calculated_property → element (3ホップ)
    path3 = find_shortest_table_path(table_graph, "calculated_property", "element")
    check(
        "calculated_property → element パス存在",
        len(path3) >= 3,
        f"実際: {path3}",
    )

    # テスト4: 同一テーブル → 自身
    path4 = find_shortest_table_path(table_graph, "material_entry", "material_entry")
    check(
        "同一テーブル → パス長 1",
        len(path4) == 1,
        f"実際: {path4}",
    )

    # --- 4. Steiner木近似（サブグラフ選択） ---
    print("\n■ 4. Steiner木近似（最小JOINサブグラフ）")

    # テスト1: 2テーブル
    sub1 = find_join_subgraph(table_graph, ["composition", "phase_stability"])
    tables_in_sub1 = set()
    for p in sub1["paths"]:
        tables_in_sub1.update(p)
    check(
        "2テーブル → material_entryが中継に含まれる",
        "material_entry" in tables_in_sub1,
        f"テーブルセット: {tables_in_sub1}",
    )
    check(
        "2テーブル → サブグラフサイズ ≤ 4",
        len(tables_in_sub1) <= 4,
        f"実際: {len(tables_in_sub1)}",
    )

    # テスト2: 3テーブル（composition + structure + phase_stability）
    sub2 = find_join_subgraph(
        table_graph, ["composition", "structure", "phase_stability"]
    )
    tables_in_sub2 = set()
    for p in sub2["paths"]:
        tables_in_sub2.update(p)
    check(
        "3テーブル → 全必要テーブルが含まれる",
        {"composition", "structure", "phase_stability"}.issubset(tables_in_sub2),
        f"テーブルセット: {tables_in_sub2}",
    )
    check(
        "3テーブル → material_entryが中継に含まれる",
        "material_entry" in tables_in_sub2,
        f"テーブルセット: {tables_in_sub2}",
    )

    # テスト3: 5テーブルクロスドメインクエリ
    cross_domain = [
        "material_entry", "composition", "elastic_tensor",
        "band_structure", "thermal_property",
    ]
    sub3 = find_join_subgraph(table_graph, cross_domain)
    tables_in_sub3 = set()
    for p in sub3["paths"]:
        tables_in_sub3.update(p)
    check(
        "5テーブル → 全必要テーブルが含まれる",
        set(cross_domain).issubset(tables_in_sub3),
        f"不足: {set(cross_domain) - tables_in_sub3}",
    )
    check(
        "5テーブル → サブグラフサイズ ≤ 8（最小に近い）",
        len(tables_in_sub3) <= 8,
        f"実際: {len(tables_in_sub3)}",
    )

    # --- 5. JOINエッジ抽出 ---
    print("\n■ 5. JOINエッジ抽出")

    # table_graph（無向グラフ）を使ってパスのJOINエッジを抽出
    path_for_edges = find_shortest_table_path(table_graph, "composition", "phase_stability")
    if len(path_for_edges) >= 2:
        edges = extract_join_edges(table_graph, path_for_edges)
        check(
            "composition→phase_stability間のJOINエッジが取得できる",
            len(edges) >= 1,
            f"エッジ数: {len(edges)}",
        )
        # エッジにJOIN条件が含まれることを確認
        has_entry_id = any("entry_id" in str(e) for e in edges)
        check(
            "JOINエッジにentry_idカラムが含まれる",
            has_entry_id,
            f"エッジ: {edges}",
        )
        print(f"    → JOINパス: {' → '.join(path_for_edges)}")
        for e in edges:
            print(f"      {e['source_table']}.{e['source_column']} = {e['target_table']}.{e['target_column']}")
    else:
        check("JOINエッジ抽出", False, "パスが見つかりません")

    # --- 6. スキーマ規模の検証 ---
    print("\n■ 6. スキーマ規模の検証（論文の記述と一致するか）")

    # 各テーブルのカラム数
    total_columns = sum(len(cols) for cols in columns.values())
    check(
        "総カラム数 > 150",
        total_columns > 150,
        f"実際: {total_columns}",
    )
    print(f"    → 総カラム数: {total_columns}")

    # material_entryのカラム確認
    entry_cols = [c.column_name for c in columns.get("material_entry", [])]
    check(
        "material_entryにformula列がある",
        "formula" in entry_cols,
        f"カラム: {entry_cols}",
    )

    # compositionテーブル確認
    comp_cols = [c.column_name for c in columns.get("composition", [])]
    check(
        "compositionにelement列がある",
        "element" in comp_cols,
        f"カラム: {comp_cols}",
    )

    cur = conn.cursor()
        "material_entry": 100,
        "composition": 100,
        "structure": 100,
        "phase_stability": 100,
        "calculation": 50,
        "calculated_property": 50,
        "prototype_definition": 1,
    }
    print("\n    Seedデータ件数:")
        cur.execute(f"SELECT count(*) FROM {tbl}")
        cnt = cur.fetchone()[0]
        check(
            f"{tbl}件数 ≥ {min_count}",
            cnt >= min_count,
            f"実際: {cnt}",
        )
        print(f"      {tbl}: {cnt}件")

    # --- A-1: band_gap / space_group NULLデータ偽陽性検出 ---
    print("\n■ 7. NULLデータ偽陽性検出（A-1）")

    cur.execute("SELECT count(*) FROM structure WHERE space_group IS NOT NULL")
    sg_notnull = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM structure")
    sg_total = cur.fetchone()[0]
    sg_ratio = sg_notnull / sg_total if sg_total > 0 else 0
    check(
        "structure.space_groupの非NULL率 > 50%",
        sg_ratio > 0.5,
        f"非NULL: {sg_notnull}/{sg_total} ({sg_ratio*100:.1f}%)",
    )

    cur.execute("SELECT count(*) FROM phase_stability WHERE band_gap IS NOT NULL")
    bg_notnull = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM phase_stability")
    bg_total = cur.fetchone()[0]
    bg_ratio = bg_notnull / bg_total if bg_total > 0 else 0
    check(
        "phase_stability.band_gapの非NULL率 > 50%",
        bg_ratio > 0.5,
        f"非NULL: {bg_notnull}/{bg_total} ({bg_ratio*100:.1f}%)",
    )

    # band_gapを使うクエリが意味のある結果を返すか確認
    cur.execute("SELECT avg(band_gap) FROM phase_stability WHERE band_gap IS NOT NULL")
    avg_bg = cur.fetchone()[0]
    check(
        "band_gap平均値が物理的に妥当（0-10 eV）",
        avg_bg is not None and 0 <= avg_bg <= 10,
        f"平均: {avg_bg}",
    )

    # --- B-2: 非連結テーブルへのクエリ処理 ---
    print("\n■ 8. 非連結テーブルのエラーハンドリング（B-2）")

    # FK接続のないテーブルを特定
    tables_in_graph = set(table_graph.nodes())
    isolated_tables = set(tables) - tables_in_graph
    print(f"    非連結テーブル: {isolated_tables if isolated_tables else 'なし'}")

    for iso_tbl in isolated_tables:
        try:
            path = find_shortest_table_path(table_graph, "material_entry", iso_tbl)
            if len(path) == 0:
                check(
                    f"非連結テーブル{iso_tbl}への走査 → 空パスを返す",
                    True,
                    "",
                )
            else:
                # 非連結テーブルにパスが見つかった場合は警告（FK接続ありの可能性）
                check(
                    f"非連結テーブル{iso_tbl}への走査 → 空パスを期待",
                    False,
                    f"パスが見つかりました: {path}（FK接続がある可能性）",
                )
        except Exception as e:
            # エラーが発生する場合、例外型を記録
            err_type = type(e).__name__
            check(
                f"非連結テーブル{iso_tbl}への走査 → 例外処理される",
                isinstance(e, (nx.NetworkXError, nx.NodeNotFound, KeyError)),
                f"予期しない例外: {err_type}: {e}",
            )

    # --- B-3: Steiner木べき等性テスト ---
    print("\n■ 9. Steiner木べき等性テスト（B-3）")

    idempotent_targets = [
        ["composition", "phase_stability"],
        ["composition", "structure", "elastic_tensor"],
        ["material_entry", "band_structure", "thermal_property", "surface_energy"],
    ]
    for targets in idempotent_targets:
        results = []
        for _ in range(5):
            sub = find_join_subgraph(table_graph, targets)
            tbl_set = set()
            for p in sub["paths"]:
                tbl_set.update(p)
            results.append(frozenset(tbl_set))
        all_same = len(set(results)) == 1
        check(
            f"Steiner木べき等: {targets} → 5回同一",
            all_same,
            f"異なるサブグラフ: {len(set(results))}種類",
        )

    conn.close()

    # --- 結果サマリ ---
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  検証結果: {passed}/{total} パス")
    if failed == 0:
        print("  判定: OK — 全チェック合格")
    else:
        print(f"  判定: NG — {failed}件の不合格あり")
    print("=" * 60)

    print("\n次のステップ: bash scripts/04_run_150query_experiment.sh")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
