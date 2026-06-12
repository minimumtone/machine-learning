#!/usr/bin/env python3
"""
論文掲載計算の唯一の出力源 (Single Source of Truth)
====================================================

【方針】
  論文 (paper/t2sql_materials_paper.tex) に掲載する数値は、
  すべてこのスクリプトの出力ファイル (paper/paper_figures.json) から
  参照しなければならない。手動で数値を埋め込むことは禁止する。

  ・paper_figures.json に存在しない数値を論文に記載してはならない
  ・数値を更新する場合は本スクリプトの入力データを更新し再実行する
  ・出力の各キーには paper_ref (論文中の参照先) を付与する

【入力】
  evaluation/proposed_result.csv        … Proposed 手法の 100 クエリ結果
  evaluation/baseline_result.csv        … 全 5 手法 × 100 クエリ結果
  evaluation/gold_sql/*.sql             … gold SQL（n_tables 算出に使用）
  evaluation/expert_evaluation_results.json … 独立設計 100 クエリ結果
  evaluation/known_l12_recovery.csv     … 既知 L1_2 再発見結果
  evaluation/stable_l12_candidates.csv  … 安定候補一覧
  evaluation/gamma_prime_candidate_ranking.csv … γ' ランキング

【出力】
  paper/paper_figures.json              … 論文掲載用数値の JSON
"""

import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"
PAPER = ROOT / "paper"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line)
    reader = csv.DictReader(rows)
    return list(reader)


def count_tables_in_sql(sql_text: str) -> int:
    """gold SQL の FROM / JOIN 句から参照テーブル数を数える。"""
    sql_upper = sql_text.upper()
    tables = set()
    # FROM table
    for m in re.finditer(r'\bFROM\s+(\w+)', sql_upper):
        tables.add(m.group(1))
    # JOIN table
    for m in re.finditer(r'\bJOIN\s+(\w+)', sql_upper):
        tables.add(m.group(1))
    # remove common aliases / keywords that aren't tables
    tables -= {"SELECT", "WHERE", "AND", "OR", "ON", "AS", "LATERAL"}
    return len(tables)


def n_tables_difficulty(n: int) -> str:
    if n <= 2:
        return "easy"
    elif n == 3:
        return "medium"
    elif n == 4:
        return "hard"
    else:
        return "very_hard"


def pct(val, digits=1):
    return round(val * 100, digits)


# ---------------------------------------------------------------------------
# 1. n_tables マッピング（gold SQL から算出）
# ---------------------------------------------------------------------------

gold_sql_dir = EVAL / "gold_sql"
n_tables_map = {}
for sql_file in sorted(gold_sql_dir.glob("*.sql")):
    qid = sql_file.stem
    sql_text = sql_file.read_text(encoding="utf-8")
    n_tables_map[qid] = count_tables_in_sql(sql_text)

# author queries only (q_easy/q_medium/q_hard/q_vhard)
author_qids = [q for q in n_tables_map if not q.startswith("q_expert")]
ntables_difficulty_map = {qid: n_tables_difficulty(n_tables_map[qid]) for qid in author_qids}

# difficulty distribution (n_tables-based)
diff_counts = Counter(ntables_difficulty_map.values())

# ---------------------------------------------------------------------------
# 2. Proposed 手法結果
# ---------------------------------------------------------------------------

proposed = read_csv(EVAL / "proposed_result.csv")
proposed_by_qid = {r["query_id"]: r for r in proposed}

# assign n_tables difficulty
for r in proposed:
    r["ntables_difficulty"] = ntables_difficulty_map.get(r["query_id"], r["difficulty"])

# overall metrics
n_total = len(proposed)
avg_exec_acc = sum(float(r["execution_accuracy"]) for r in proposed) / n_total
syntax_valid_rate = sum(1 for r in proposed if r["syntax_valid"].lower() == "true") / n_total
exec_valid_rate = sum(1 for r in proposed if r["execution_valid"].lower() == "true") / n_total
avg_table_halluc = sum(float(r["hallucinated_table_rate"]) for r in proposed) / n_total
avg_join_halluc = sum(float(r["hallucinated_join_rate"]) for r in proposed) / n_total
avg_latency = sum(float(r["latency_ms"]) for r in proposed) / n_total
avg_tokens = sum(float(r["token_usage"]) for r in proposed) / n_total
repair_count = sum(1 for r in proposed if int(r.get("repair_attempts") or 0) > 0)
repair_total_attempts = sum(int(r.get("repair_attempts") or 0) for r in proposed)

# per-difficulty (n_tables-based)
diff_metrics = {}
for diff in ["easy", "medium", "hard", "very_hard"]:
    subset = [r for r in proposed if r["ntables_difficulty"] == diff]
    n = len(subset)
    acc = sum(float(r["execution_accuracy"]) for r in subset) / n if n else 0
    ev = sum(1 for r in subset if r["execution_valid"].lower() == "true") / n if n else 0
    diff_metrics[diff] = {"n": n, "exec_accuracy": pct(acc), "exec_validity": pct(ev)}

# multi-hop (n_tables >= 3)
multihop_queries = [r for r in proposed if n_tables_map.get(r["query_id"], 0) >= 3]
multihop_success = sum(1 for r in multihop_queries if float(r["execution_accuracy"]) >= 0.8)
multihop_rate = multihop_success / len(multihop_queries) if multihop_queries else 0

# per n_tables group for multi-hop table
ntables_groups = defaultdict(list)
for r in proposed:
    nt = n_tables_map.get(r["query_id"], 0)
    if nt >= 3:
        if nt <= 4:
            key = str(nt)
        else:
            key = "5-6"
        ntables_groups[key].append(float(r["execution_accuracy"]))

# ---------------------------------------------------------------------------
# 3. ベースライン結果
# ---------------------------------------------------------------------------

baseline_all = read_csv(EVAL / "baseline_result.csv")

methods = ["baseline1_llm_only", "baseline2_full_schema", "baseline3_rule_based",
           "baseline4_fk_list", "proposed"]
method_labels = {"baseline1_llm_only": "B1", "baseline2_full_schema": "B2",
                 "baseline3_rule_based": "B3", "baseline4_fk_list": "B4",
                 "proposed": "P"}

baseline_metrics = {}
for method in methods:
    if method == "proposed":
        rows = proposed
    else:
        rows = [r for r in baseline_all if r["method"] == method]
    n = len(rows)
    if n == 0:
        continue
    acc = sum(float(r["execution_accuracy"]) for r in rows) / n
    syn = sum(1 for r in rows if r["syntax_valid"].lower() == "true") / n
    exe = sum(1 for r in rows if r["execution_valid"].lower() == "true") / n
    th = sum(float(r["hallucinated_table_rate"]) for r in rows) / n
    jh = sum(float(r["hallucinated_join_rate"]) for r in rows) / n
    lat = sum(float(r["latency_ms"]) for r in rows) / n
    tok = sum(float(r["token_usage"]) for r in rows) / n

    # join hallucination count (queries with rate > 0)
    join_halluc_count = sum(1 for r in rows if float(r["hallucinated_join_rate"]) > 0)
    # execution error count
    exec_error_count = sum(1 for r in rows if r["execution_valid"].lower() != "true")
    syntax_error_count = sum(1 for r in rows if r["syntax_valid"].lower() != "true")

    # n_tables difficulty breakdown
    diff_breakdown = {}
    for diff in ["easy", "medium", "hard", "very_hard"]:
        subset = [r for r in rows if ntables_difficulty_map.get(r["query_id"], r["difficulty"]) == diff]
        sn = len(subset)
        sacc = sum(float(r["execution_accuracy"]) for r in subset) / sn if sn else 0
        diff_breakdown[diff] = {"n": sn, "exec_accuracy": pct(sacc)}

    baseline_metrics[method_labels[method]] = {
        "method": method,
        "n": n,
        "exec_accuracy": pct(acc),
        "syntax_validity": pct(syn),
        "exec_validity": pct(exe),
        "table_halluc_rate_pct": pct(th),
        "join_halluc_count": join_halluc_count,
        "exec_error_count": exec_error_count,
        "syntax_error_count": syntax_error_count,
        "avg_latency_ms": round(lat, 2),
        "avg_token_usage": round(tok, 2),
        "difficulty_breakdown": diff_breakdown,
    }

# multi-hop comparison (Full Schema vs Proposed)
multihop_comparison = {}
for group_key in ["3", "4", "5-6"]:
    row = {}
    for method in ["baseline2_full_schema", "proposed"]:
        label = method_labels[method]
        if method == "proposed":
            rows = proposed
        else:
            rows = [r for r in baseline_all if r["method"] == method]
        subset = []
        for r in rows:
            nt = n_tables_map.get(r["query_id"], 0)
            if group_key == "5-6" and nt >= 5:
                subset.append(float(r["execution_accuracy"]))
            elif group_key != "5-6" and nt == int(group_key):
                subset.append(float(r["execution_accuracy"]))
        row[label] = pct(sum(subset) / len(subset)) if subset else 0
        row[f"{label}_n"] = len(subset)
    multihop_comparison[group_key] = row

# ---------------------------------------------------------------------------
# 4. 独立設計クエリ（専門家評価）
# ---------------------------------------------------------------------------

with open(EVAL / "expert_evaluation_results.json", encoding="utf-8") as f:
    expert_data = json.load(f)

expert_summary = expert_data["summary"]
expert_results = expert_data["results"]

expert_out = {
    "total": expert_summary["total"],
    "syntax_valid": expert_summary["syntax_valid"],
    "execution_success": expert_summary["execution_success"],
    "binary_correct": expert_summary["correct"],
    "binary_correct_rate": expert_summary["accuracy"],
    "mean_exec_accuracy": expert_data["comparison"]["expert_designed"]["mean_execution_accuracy"],
    "by_difficulty": expert_summary["by_difficulty"],
}

# ---------------------------------------------------------------------------
# 5. 材料工学的評価
# ---------------------------------------------------------------------------

# 5a. 既知 L1_2 回収
known = read_csv(EVAL / "known_l12_recovery.csv")
known_l12 = [r for r in known if r["is_known"].lower() == "true"]
recovered = [r for r in known_l12 if r["known_l12_recovered"].lower() == "true"]
total_l12 = sum(1 for r in known if r["prototype"] == "L12")

# 5b. 安定候補
stable_cands = read_csv(EVAL / "stable_l12_candidates.csv")
stable_count = sum(1 for r in stable_cands if r["stability_class"] == "stable")
metastable_count = sum(1 for r in stable_cands if r["stability_class"] == "metastable")

# 5c. γ' ランキング
gamma = read_csv(EVAL / "gamma_prime_candidate_ranking.csv")
top1 = gamma[0] if gamma else {}

# 5d. Ni3Al lattice matched
lattice_cands_path = EVAL / "ni3al_lattice_matched_candidates.csv"
if lattice_cands_path.exists():
    lattice_cands = read_csv(lattice_cands_path)
    lattice_matched_count = len(lattice_cands)
else:
    lattice_matched_count = 0

# ---------------------------------------------------------------------------
# 6. DB 規模
# ---------------------------------------------------------------------------

db_stats = {
    "n_tables": 30,
    "n_entries": 1470,
    "n_prototypes": 5,
    "prototype_distribution": {
        "L12": total_l12,
        "B2": 636,
        "NaCl": 355,
        "NiAs": 74,
        "BiF3": 13,
    },
}

# ---------------------------------------------------------------------------
# 7. 評価指標改善 (metric improvement)
# ---------------------------------------------------------------------------

# raw (tuple exact match) accuracy
raw_acc = sum(float(r["raw_execution_accuracy"]) for r in proposed) / n_total
metric_improvement = {
    "raw_accuracy": pct(raw_acc),
    "improved_accuracy": pct(avg_exec_acc),
    "improvement_pt": round(pct(avg_exec_acc) - pct(raw_acc), 1),
}

def _identify_representative_run(
    run_paths: list[Path], representative_path: Path
) -> str:
    """proposed_result.csv がどのランと一致するかをMD5で照合して返す。"""
    import hashlib
    rep_hash = hashlib.md5(representative_path.read_bytes()).hexdigest()
    for p in run_paths:
        if hashlib.md5(p.read_bytes()).hexdigest() == rep_hash:
            return f"{p.name} (= proposed_result.csv)"
    return "proposed_result.csv (ランファイルと不一致)"


# ---------------------------------------------------------------------------
# 7b. 3-run 再現性統計 (proposed_result_run{1,2,3}.csv から自動計算)
# ---------------------------------------------------------------------------

run_csv_paths = sorted(EVAL.glob("proposed_result_run*.csv"))
run_stats: dict = {}
if len(run_csv_paths) >= 2:
    run_accuracies = []
    run_by_difficulty: dict[str, list[float]] = defaultdict(list)
    for csv_path in run_csv_paths:
        rows = read_csv(csv_path)
        acc = sum(float(r["execution_accuracy"]) for r in rows) / len(rows)
        run_accuracies.append(pct(acc))
        for diff in ["easy", "medium", "hard", "very_hard"]:
            subset = [r for r in rows if ntables_difficulty_map.get(r["query_id"], r["difficulty"]) == diff]
            if subset:
                dacc = sum(float(r["execution_accuracy"]) for r in subset) / len(subset)
                run_by_difficulty[diff].append(pct(dacc))

    n_runs = len(run_accuracies)
    mean_acc = round(sum(run_accuracies) / n_runs, 1)
    if n_runs >= 2:
        variance = sum((x - mean_acc) ** 2 for x in run_accuracies) / (n_runs - 1)
        stdev = round(math.sqrt(variance), 1)
    else:
        stdev = 0.0

    diff_stats = {}
    for diff, vals in run_by_difficulty.items():
        dmean = round(sum(vals) / len(vals), 1)
        if len(vals) >= 2:
            dvar = sum((x - dmean) ** 2 for x in vals) / (len(vals) - 1)
            dstd = round(math.sqrt(dvar), 1)
        else:
            dstd = 0.0
        diff_stats[diff] = {"mean": dmean, "stdev": dstd, "values": vals}

    run_stats = {
        "paper_ref": "Abstract, Section 4.2 (再現性評価), Conclusion",
        "n_runs": n_runs,
        "run_files": [p.name for p in run_csv_paths],
        "run_accuracies": run_accuracies,
        "mean_accuracy_pct": mean_acc,
        "stdev_pp": stdev,
        "max_accuracy_pct": max(run_accuracies),
        "min_accuracy_pct": min(run_accuracies),
        "representative_run": _identify_representative_run(run_csv_paths, EVAL / "proposed_result.csv"),
        "by_difficulty": diff_stats,
        "note": "gpt-5.5は温度制御不可のため複数回実行し平均±標準偏差を報告",
    }
else:
    print("WARNING: proposed_result_run*.csv が2つ未満のため3-run統計をスキップ")

# ---------------------------------------------------------------------------
# 8. エラー分析 (error analysis)
# ---------------------------------------------------------------------------

# Very Hard failure breakdown: count queries with execution_accuracy < 0.45
vhard_queries = [r for r in proposed if r["ntables_difficulty"] == "very_hard"]
vhard_failure_count = sum(1 for r in vhard_queries if float(r["execution_accuracy"]) < 0.45)

# ---------------------------------------------------------------------------
# 9. テスト件数
# ---------------------------------------------------------------------------

test_counts = {
    "regression_tests": 80,
    "total_unit_tests": 125,
    "note": "pytest tests/ -q で確認"
}

# ---------------------------------------------------------------------------
# Assemble output
# ---------------------------------------------------------------------------

output = {
    "_meta": {
        "description": "論文掲載用数値 (Single Source of Truth)",
        "generated_by": "scripts/compute_paper_figures.py",
        "policy": "論文に掲載する数値はすべてこのファイルから参照すること",
    },

    "proposed_overall": {
        "paper_ref": "Abstract, Table (tab:baseline_results) P行, Table (tab:independent_eval) 著者設計列",
        "n_queries": n_total,
        "exec_accuracy_pct": pct(avg_exec_acc),
        "syntax_validity_pct": pct(syntax_valid_rate),
        "exec_validity_pct": pct(exec_valid_rate),
        "table_halluc_rate_pct": pct(avg_table_halluc),
        "join_halluc_rate_pct": pct(avg_join_halluc),
        "avg_latency_ms": round(avg_latency, 0),
        "avg_token_usage": round(avg_tokens, 0),
        "repair_query_count": repair_count,
        "repair_total_attempts": repair_total_attempts,
    },

    **({"proposed_3run_stats": run_stats} if run_stats else {}),

    "proposed_by_difficulty": {
        "paper_ref": "Table (tab:difficulty_breakdown), 難易度別の傾向",
        "classification": "gold SQLの参照テーブル数 (n_tables): 1-2=Easy, 3=Medium, 4=Hard, 5+=Very Hard",
        **diff_metrics,
    },

    "baseline_comparison": {
        "paper_ref": "Table (tab:baseline_results), 主要な発見",
        **baseline_metrics,
    },

    "multihop_comparison": {
        "paper_ref": "Table (tab:multihop)",
        "multihop_total": len(multihop_queries),
        "multihop_success_count": multihop_success,
        "multihop_success_rate_pct": pct(multihop_rate),
        "by_n_tables": multihop_comparison,
    },

    "error_analysis": {
        "paper_ref": "Table (tab:error_analysis), Very Hard失敗内訳 (tab:vhard_failure)",
        "by_method": {
            label: {
                "exec_error_count": baseline_metrics[label]["exec_error_count"],
                "syntax_error_count": baseline_metrics[label]["syntax_error_count"],
                "table_halluc_rate_pct": baseline_metrics[label]["table_halluc_rate_pct"],
                "join_halluc_count": baseline_metrics[label]["join_halluc_count"],
            }
            for label in ["B1", "B2", "B3", "B4", "P"]
        },
        "very_hard_failure_count": vhard_failure_count,
        "very_hard_total": len(vhard_queries),
        "very_hard_accuracy_pct": diff_metrics["very_hard"]["exec_accuracy"],
    },

    "latency": {
        "paper_ref": "Table (tab:latency)",
        "by_method": {
            label: {
                "avg_latency_ms": baseline_metrics[label]["avg_latency_ms"],
                "avg_token_usage": baseline_metrics[label]["avg_token_usage"],
            }
            for label in ["B1", "B2", "B3", "B4", "P"]
        },
    },

    "metric_improvement": {
        "paper_ref": "Table (tab:metric_improvement)",
        **metric_improvement,
    },

    "independent_eval": {
        "paper_ref": "Table (tab:independent_eval), 独立評価セクション",
        **expert_out,
    },

    "materials_engineering": {
        "paper_ref": "Table (tab:known_l12), Section 4.3.3, Table (tab:gamma_prime)",
        "known_l12_total": len(known_l12),
        "known_l12_recovered": len(recovered),
        "known_l12_recovery_rate_pct": pct(len(recovered) / len(known_l12)) if known_l12 else 0,
        "total_l12_in_db": total_l12,
        "stable_candidates": stable_count,
        "metastable_candidates": metastable_count,
        "gamma_prime_total": stable_count + metastable_count,
        "gamma_prime_top1_formula": top1.get("formula", ""),
        "gamma_prime_top1_score": float(top1.get("composite_score", 0)),
        "lattice_matched_candidates": lattice_matched_count,
    },

    "db_stats": {
        "paper_ref": "Abstract, Section 3.1",
        **db_stats,
    },

    "test_counts": {
        "paper_ref": "Section 4.1 (回帰テスト), 結論",
        **test_counts,
    },

    "difficulty_distribution": {
        "paper_ref": "Table (tab:difficulty_def)",
        "easy": diff_counts.get("easy", 0),
        "medium": diff_counts.get("medium", 0),
        "hard": diff_counts.get("hard", 0),
        "very_hard": diff_counts.get("very_hard", 0),
    },
}

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

out_path = PAPER / "paper_figures.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✓ {out_path} generated ({os.path.getsize(out_path):,} bytes)")
print()

# quick summary
print("=== 主要値サマリー ===")
print(f"Proposed 全体精度:    {output['proposed_overall']['exec_accuracy_pct']}%")
print(f"  Easy:               {diff_metrics['easy']['exec_accuracy']}% ({diff_metrics['easy']['n']}件)")
print(f"  Medium:             {diff_metrics['medium']['exec_accuracy']}% ({diff_metrics['medium']['n']}件)")
print(f"  Hard:               {diff_metrics['hard']['exec_accuracy']}% ({diff_metrics['hard']['n']}件)")
print(f"  Very Hard:          {diff_metrics['very_hard']['exec_accuracy']}% ({diff_metrics['very_hard']['n']}件)")
for label in ["B1", "B2", "B3", "B4"]:
    m = baseline_metrics[label]
    print(f"{label} ({m['method']}): {m['exec_accuracy']}%")
if run_stats:
    print(f"3-run統計:            {run_stats['mean_accuracy_pct']}% ± {run_stats['stdev_pp']}pp ({run_stats['run_accuracies']})")
print(f"独立評価: 二値正答率 {expert_out['binary_correct_rate']}%, 平均精度 {expert_out['mean_exec_accuracy']}%")
print(f"既知L12回収: {len(recovered)}/{len(known_l12)}")
print(f"γ'候補: {stable_count + metastable_count}件 (安定{stable_count} + 準安定{metastable_count})")
