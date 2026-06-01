#!/usr/bin/env python3
"""
Extended Schema Experiment:
Tests Schema Graph Text-to-SQL performance on 20-table complex schema.
Compares success rates with original 7-table schema experiments.
"""
import json
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from openai import OpenAI

DB_CONFIG = {
    'dbname': 'l12_materials',
    'user': 'l12_user',
    'password': 'l12_password',
    'host': 'localhost',
    'port': 5432
}

# Extended schema YAML for prompt injection (30 tables - matching OQMD real scale)
EXTENDED_SCHEMA_YAML = """
tables:
  material_entry:
    columns: [entry_id (PK, text), source_db, source_material_id, formula, reduced_formula, chemical_system, number_of_elements]
    
  composition:
    columns: [composition_id (PK), entry_id (FK→material_entry), element, atomic_fraction, site_label]
    
  structure:
    columns: [structure_id (PK), entry_id (FK→material_entry), prototype, strukturbericht, formula_type, space_group_number, crystal_system, lattice_a, lattice_b, lattice_c, volume_per_atom, space_group]
    
  phase_stability:
    columns: [stability_id (PK), entry_id (FK→material_entry), formation_energy_per_atom, energy_above_hull, is_stable, band_gap]
    
  calculation:
    columns: [calculation_id (PK, text), entry_id (FK→material_entry), method, functional, calculation_type]
    
  calculated_property:
    columns: [property_id (PK), calculation_id (FK→calculation), property_name, value, unit, tensor_component]
    
  prototype_definition:
    columns: [prototype_id (PK), prototype_name, strukturbericht, formula_type, description]
    
  element:
    columns: [element_id (PK), symbol (UNIQUE), name, atomic_number, atomic_mass, electronegativity, atomic_radius, group_number, period_number, block, category]
    
  element_property:
    columns: [element_property_id (PK), element_id (FK→element), property_name, value, unit, temperature_k, source]
    
  space_group:
    columns: [space_group_id (PK), space_group_number (UNIQUE), hermann_mauguin, crystal_system, point_group, laue_class, is_centrosymmetric]
    
  application_domain:
    columns: [domain_id (PK), domain_name, description, parent_domain_id (FK→application_domain, self-ref)]
    note: hierarchical structure, parent_domain_id references own table
    
  material_application:
    columns: [material_application_id (PK), entry_id (FK→material_entry), domain_id (FK→application_domain), relevance_score, notes]
    
  literature_reference:
    columns: [reference_id (PK), doi, title, authors, journal, year, volume, pages]
    
  material_reference:
    columns: [material_reference_id (PK), entry_id (FK→material_entry), reference_id (FK→literature_reference), context]
    
  experimental_measurement:
    columns: [measurement_id (PK), entry_id (FK→material_entry), reference_id (FK→literature_reference), method, temperature_k, pressure_gpa]
    
  measured_property:
    columns: [measured_property_id (PK), measurement_id (FK→experimental_measurement), property_name, value, uncertainty, unit]
    
  synthesis_method:
    columns: [synthesis_id (PK), method_name, category, description]
    
  material_synthesis:
    columns: [material_synthesis_id (PK), entry_id (FK→material_entry), synthesis_id (FK→synthesis_method), reference_id (FK→literature_reference), temperature_k, duration_hours, atmosphere, success]
    
  defect_type:
    columns: [defect_type_id (PK), defect_name, category, description]
    
  material_defect:
    columns: [material_defect_id (PK), entry_id (FK→material_entry), defect_type_id (FK→defect_type), formation_energy, concentration, site, dopant_element_id (FK→element)]

  band_structure:
    columns: [band_structure_id (PK), calculation_id (FK→calculation), entry_id (FK→material_entry), is_direct_gap, cbm_energy, vbm_energy, band_gap_type, num_bands, num_kpoints]

  density_of_states:
    columns: [dos_id (PK), calculation_id (FK→calculation), entry_id (FK→material_entry), total_dos_at_fermi, efermi, is_metallic, spin_polarized]

  elastic_tensor:
    columns: [elastic_id (PK), entry_id (FK→material_entry), calculation_id (FK→calculation), bulk_modulus_vrh, shear_modulus_vrh, youngs_modulus, poisson_ratio, is_stable]

  magnetic_property:
    columns: [magnetic_id (PK), entry_id (FK→material_entry), total_magnetization, magnetic_ordering, curie_temperature_k, magnetic_anisotropy_energy]

  thermal_property:
    columns: [thermal_id (PK), entry_id (FK→material_entry), calculation_id (FK→calculation), debye_temperature_k, thermal_conductivity, specific_heat_cv, gruneisen_parameter, temperature_k]

  surface_energy:
    columns: [surface_id (PK), entry_id (FK→material_entry), miller_index, surface_energy_j_m2, work_function, is_reconstructed]

  grain_boundary:
    columns: [grain_boundary_id (PK), entry_id (FK→material_entry), sigma_value, rotation_axis, tilt_angle, gb_energy_j_m2, excess_volume]

  phase_diagram_entry:
    columns: [phase_entry_id (PK), entry_id (FK→material_entry), chemical_system, is_on_hull, decomposition_products, hull_distance]

  alloy_system:
    columns: [alloy_system_id (PK), system_name, num_components, category, description]

  material_alloy_system:
    columns: [material_alloy_id (PK), entry_id (FK→material_entry), alloy_system_id (FK→alloy_system), phase, composition_type]

foreign_keys:
  - composition.entry_id → material_entry.entry_id
  - structure.entry_id → material_entry.entry_id
  - phase_stability.entry_id → material_entry.entry_id
  - calculation.entry_id → material_entry.entry_id
  - calculated_property.calculation_id → calculation.calculation_id
  - element_property.element_id → element.element_id
  - application_domain.parent_domain_id → application_domain.domain_id (self-ref)
  - material_application.entry_id → material_entry.entry_id
  - material_application.domain_id → application_domain.domain_id
  - material_reference.entry_id → material_entry.entry_id
  - material_reference.reference_id → literature_reference.reference_id
  - experimental_measurement.entry_id → material_entry.entry_id
  - experimental_measurement.reference_id → literature_reference.reference_id
  - measured_property.measurement_id → experimental_measurement.measurement_id
  - material_synthesis.entry_id → material_entry.entry_id
  - material_synthesis.synthesis_id → synthesis_method.synthesis_id
  - material_synthesis.reference_id → literature_reference.reference_id
  - material_defect.entry_id → material_entry.entry_id
  - material_defect.defect_type_id → defect_type.defect_type_id
  - material_defect.dopant_element_id → element.element_id
  - band_structure.calculation_id → calculation.calculation_id
  - band_structure.entry_id → material_entry.entry_id
  - density_of_states.calculation_id → calculation.calculation_id
  - density_of_states.entry_id → material_entry.entry_id
  - elastic_tensor.entry_id → material_entry.entry_id
  - elastic_tensor.calculation_id → calculation.calculation_id
  - magnetic_property.entry_id → material_entry.entry_id
  - thermal_property.entry_id → material_entry.entry_id
  - thermal_property.calculation_id → calculation.calculation_id
  - surface_energy.entry_id → material_entry.entry_id
  - grain_boundary.entry_id → material_entry.entry_id
  - phase_diagram_entry.entry_id → material_entry.entry_id
  - material_alloy_system.entry_id → material_entry.entry_id
  - material_alloy_system.alloy_system_id → alloy_system.alloy_system_id
"""

# Test queries categorized by JOIN complexity
EXTENDED_QUERIES = [
    # === Category 1: Simple (1-2 tables, same as original) ===
    {"id": "E01", "query": "B2プロトタイプの化合物を全て出して", "category": "simple", "min_tables": 1, "expected_tables": ["structure"]},
    {"id": "E02", "query": "Feを含む安定な化合物は？", "category": "simple", "min_tables": 3, "expected_tables": ["material_entry", "composition", "phase_stability"]},
    {"id": "E03", "query": "band_gapが2以上の化合物を出して", "category": "simple", "min_tables": 2, "expected_tables": ["material_entry", "phase_stability"]},
    {"id": "E04", "query": "NaCl型でband_gapが0の金属的な化合物は？", "category": "simple", "min_tables": 3, "expected_tables": ["material_entry", "structure", "phase_stability"]},
    {"id": "E05", "query": "L12型でenergy_above_hullが0.01未満のものを出して", "category": "simple", "min_tables": 3, "expected_tables": ["material_entry", "structure", "phase_stability"]},
    
    # === Category 2: Medium (3-4 tables, multi-hop) ===
    {"id": "E06", "query": "Aerospace Alloys用途に適した安定なB2化合物を探して", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "structure", "phase_stability", "material_application", "application_domain"]},
    {"id": "E07", "query": "Arc Meltingで合成されたNiを含む化合物の安定性は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "composition", "phase_stability", "material_synthesis", "synthesis_method"]},
    {"id": "E08", "query": "XRDで測定された安定なL12化合物のlattice_parameterを出して", "category": "medium", "min_tables": 5, "expected_tables": ["material_entry", "structure", "phase_stability", "experimental_measurement", "measured_property"]},
    {"id": "E09", "query": "遷移金属(dブロック)を含む化合物で電池用途のものは？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "material_application", "application_domain"]},
    {"id": "E10", "query": "2020年以降に出版された論文で報告されたB2化合物は？", "category": "medium", "min_tables": 4, "expected_tables": ["material_entry", "structure", "material_reference", "literature_reference"]},
    
    # === Category 3: Complex (5+ tables, multi-hop chains) ===
    {"id": "E11", "query": "Vacancy欠陥を持つ安定なNaCl型化合物で、その構成元素の電気陰性度が2.0以上のものは？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "structure", "phase_stability", "material_defect", "defect_type", "composition", "element"]},
    {"id": "E12", "query": "XRDで測定されたhardnessが10GPa以上の化合物のうち、Arc Meltingで合成されたものは？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "experimental_measurement", "measured_property", "material_synthesis", "synthesis_method"]},
    {"id": "E13", "query": "Structural MaterialsのサブカテゴリであるAerospace Alloysに適したB2化合物で、実験データがあるものの格子定数は？", "category": "complex", "min_tables": 6, "expected_tables": ["material_entry", "structure", "material_application", "application_domain", "experimental_measurement", "measured_property"]},
    {"id": "E14", "query": "Nature Materialsに掲載された化合物のうち、Vacancy形成エネルギーが1eV未満のものは？", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "material_reference", "literature_reference", "material_defect", "defect_type"]},
    {"id": "E15", "query": "4族元素(Ti,Zr,Hf)を含むB2化合物で、触媒用途があり、かつ安定なものを出して", "category": "complex", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "structure", "phase_stability", "material_application", "application_domain"]},
    
    # === Category 4: Very Complex (self-ref, subqueries, aggregation) ===
    {"id": "E16", "query": "親カテゴリがEnergy Materialsであるすべてのサブカテゴリに属する化合物数を数えて", "category": "very_complex", "min_tables": 3, "expected_tables": ["application_domain", "material_application"]},
    {"id": "E17", "query": "ドーパント元素として使われている元素で、その元素自体の電気陰性度が1.5以上のものを列挙して", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_defect", "element"]},
    {"id": "E18", "query": "3つ以上の異なるアプリケーションドメインに紐付けられた化合物のうち、安定なものだけ出して", "category": "very_complex", "min_tables": 3, "expected_tables": ["material_entry", "material_application", "phase_stability"]},
    {"id": "E19", "query": "同一化合物に対して実験値と計算値の両方が存在するものを出して", "category": "very_complex", "min_tables": 4, "expected_tables": ["material_entry", "calculated_property", "calculation", "experimental_measurement", "measured_property"]},
    {"id": "E20", "query": "vacuum雰囲気で1500K以上で合成された化合物のうち、band_gapが正で、かつ論文引用があるものは？", "category": "very_complex", "min_tables": 6, "expected_tables": ["material_entry", "material_synthesis", "phase_stability", "material_reference", "literature_reference"]},
    
    # === Category 5: Cross-domain (element properties + material properties) ===
    {"id": "E21", "query": "原子番号が26以上30以下の元素を含む化合物で、formation_energyが負のものは？", "category": "cross_domain", "min_tables": 4, "expected_tables": ["material_entry", "composition", "element", "phase_stability"]},
    {"id": "E22", "query": "alkali_metalカテゴリの元素を含む化合物で、Thermoelectrics用途に適したものは？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "composition", "element", "material_application", "application_domain"]},
    {"id": "E23", "query": "cubic結晶系の空間群に属する化合物のうち、DSCで測定されたものを出して", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "structure", "space_group", "experimental_measurement"]},
    {"id": "E24", "query": "Substitutional欠陥のドーパント元素がpブロックの化合物で、安定なものは？", "category": "cross_domain", "min_tables": 5, "expected_tables": ["material_entry", "material_defect", "defect_type", "element", "phase_stability"]},
    {"id": "E25", "query": "Ball Millingで合成され、かつnanoindentationで測定されたhardnessデータがある化合物は？", "category": "cross_domain", "min_tables": 6, "expected_tables": ["material_entry", "material_synthesis", "synthesis_method", "experimental_measurement", "measured_property"]},
    
    # === Category 6: Aggregation & Comparison ===
    {"id": "E26", "query": "プロトタイプ別の平均formation_energyを出して", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_entry", "structure", "phase_stability"]},
    {"id": "E27", "query": "合成方法ごとの成功率を計算して", "category": "aggregation", "min_tables": 2, "expected_tables": ["material_synthesis", "synthesis_method"]},
    {"id": "E28", "query": "アプリケーションドメインごとの安定化合物数を多い順に並べて", "category": "aggregation", "min_tables": 3, "expected_tables": ["material_application", "application_domain", "phase_stability"]},
    {"id": "E29", "query": "各元素が含まれる化合物数のランキングTop10を出して", "category": "aggregation", "min_tables": 1, "expected_tables": ["composition"]},
    {"id": "E30", "query": "出版年ごとの論文数と、それらに紐付く化合物数を出して", "category": "aggregation", "min_tables": 3, "expected_tables": ["literature_reference", "material_reference"]},
]


def build_schema_subset(tables: list) -> str:
    """Build a minimal schema YAML containing only the specified tables and their FK relations."""
    import re
    lines = EXTENDED_SCHEMA_YAML.strip().split('\n')
    
    # Parse table definitions
    table_defs = {}
    current_table = None
    for line in lines:
        m = re.match(r'^  (\w+):', line)
        if m and 'columns:' not in line and 'note:' not in line and 'foreign_keys:' not in line:
            current_table = m.group(1)
            table_defs[current_table] = []
        if current_table:
            table_defs[current_table].append(line)
        if line.strip() == '' and current_table:
            current_table = None
    
    # Build subset
    subset = "tables:\n"
    for t in tables:
        if t in table_defs:
            for l in table_defs[t]:
                subset += l + '\n'
            subset += '\n'
    
    # Add relevant FK lines
    subset += "foreign_keys:\n"
    fk_section = False
    for line in lines:
        if 'foreign_keys:' in line:
            fk_section = True
            continue
        if fk_section and line.strip().startswith('-'):
            # Check if both sides of FK reference tables in our subset
            parts = line.strip('- ').split('→')
            if len(parts) == 2:
                left_table = parts[0].strip().split('.')[0].strip('- ')
                right_table = parts[1].strip().split('.')[0].strip()
                # Remove trailing annotations like " (self-ref)"
                right_table = right_table.split(' ')[0].split('(')[0].strip()
                if left_table in tables and right_table in tables:
                    subset += line + '\n'
    
    return subset


def build_prompt(query: str, schema_mode: str = "full", relevant_tables: list = None) -> str:
    """Build the LLM prompt for SQL generation.
    
    schema_mode:
      - "full": Include all 30 tables (simulates: schema in prompt, no traversal)
      - "traversed": Include only relevant tables (simulates: Schema Graph traversal)
      - "none": No schema information (baseline)
    """
    base = f"""あなたは材料科学データベースのSQL生成エンジンです。
ユーザーの自然言語クエリを、PostgreSQL SQLに変換してください。

重要なルール:
- SELECT文のみ生成すること（INSERT/UPDATE/DELETE禁止）
- テーブル名・カラム名は正確に使用すること
- JOINは必要最小限のテーブルのみ使用すること
- 結果は最大100件に制限すること（LIMIT 100）
"""
    if schema_mode == "full":
        base += f"\n=== データベーススキーマ（全30テーブル） ===\n{EXTENDED_SCHEMA_YAML}\n"
    elif schema_mode == "traversed" and relevant_tables:
        subset = build_schema_subset(relevant_tables)
        base += f"\n=== データベーススキーマ（関連テーブルのみ） ===\n{subset}\n"
    # "none" mode: no schema
    
    base += f"\n=== ユーザークエリ ===\n{query}\n\n生成SQL:"
    return base


def execute_sql(sql: str) -> dict:
    """Execute SQL and return results or error."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description] if cur.description else []
        cur.close()
        return {"success": True, "rows": len(rows), "columns": cols}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if conn is not None:
            conn.close()


def count_joins_in_sql(sql: str) -> int:
    """Count the number of JOIN clauses in generated SQL."""
    import re
    return len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))


def count_tables_in_sql(sql: str) -> list:
    """Extract table names referenced in SQL (FROM and JOIN clauses)."""
    import re
    tables = set()
    # FROM clause
    for m in re.finditer(r'\bFROM\s+(\w+)', sql, re.IGNORECASE):
        tables.add(m.group(1).lower())
    # JOIN clause
    for m in re.finditer(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE):
        tables.add(m.group(1).lower())
    return sorted(tables)


def run_llm_query(query: str, model: str = "gpt-4o-mini", schema_mode: str = "full",
                  relevant_tables: list = None) -> dict:
    """Generate SQL via LLM and execute."""
    client = OpenAI()
    prompt = build_prompt(query, schema_mode, relevant_tables)
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
        )
        sql = response.choices[0].message.content.strip()
        # Clean SQL
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
            sql = sql.rsplit("```", 1)[0]
        sql = sql.strip()
    except Exception as e:
        return {"success": False, "error": f"LLM error: {e}", "latency_ms": int((time.time()-start)*1000)}
    
    latency = int((time.time() - start) * 1000)
    
    # Execute the generated SQL
    result = execute_sql(sql)
    result["sql"] = sql
    result["latency_ms"] = latency
    result["model"] = model
    result["schema_mode"] = schema_mode
    result["join_count"] = count_joins_in_sql(sql)
    result["tables_used"] = count_tables_in_sql(sql)
    
    return result


def run_experiment():
    """Run the full extended schema experiment with 3 conditions:
    1. LLM + Full Schema (30 tables in prompt, no traversal)
    2. LLM + Traversed Schema (only relevant tables, simulating SG traversal)
    3. LLM without Schema (baseline)
    """
    print("=" * 60)
    print("EXTENDED SCHEMA EXPERIMENT (30 tables)")
    print("=" * 60)
    print("Conditions: Full Schema (30t) | Traversed (subset) | No Schema")
    print("=" * 60)
    
    results = []
    
    for q in EXTENDED_QUERIES:
        print(f"\n[{q['id']}] {q['query'][:60]}...")
        
        # Condition 1: LLM + Full Schema (all 30 tables, no traversal)
        r1 = run_llm_query(q["query"], model="gpt-4o-mini", schema_mode="full")
        r1["condition"] = "llm_full_schema"
        r1["query_id"] = q["id"]
        r1["query_text"] = q["query"]
        r1["category"] = q["category"]
        r1["min_tables"] = q["min_tables"]
        r1["expected_tables"] = q["expected_tables"]
        print(f"  Full(30t):  {'✓' if r1['success'] else '✗'} JOINs={r1.get('join_count',0)} ({r1['latency_ms']}ms)")
        if not r1['success']:
            print(f"    Error: {r1.get('error', '')[:80]}")
        
        # Condition 2: LLM + Traversed Schema (only relevant tables)
        r2 = run_llm_query(q["query"], model="gpt-4o-mini", schema_mode="traversed",
                          relevant_tables=q["expected_tables"])
        r2["condition"] = "llm_traversed"
        r2["query_id"] = q["id"]
        r2["query_text"] = q["query"]
        r2["category"] = q["category"]
        r2["min_tables"] = q["min_tables"]
        r2["expected_tables"] = q["expected_tables"]
        print(f"  Traversed:  {'✓' if r2['success'] else '✗'} JOINs={r2.get('join_count',0)} ({r2['latency_ms']}ms)")
        if not r2['success']:
            print(f"    Error: {r2.get('error', '')[:80]}")
        
        # Condition 3: LLM without schema (baseline)
        r3 = run_llm_query(q["query"], model="gpt-4o-mini", schema_mode="none")
        r3["condition"] = "llm_no_schema"
        r3["query_id"] = q["id"]
        r3["query_text"] = q["query"]
        r3["category"] = q["category"]
        r3["min_tables"] = q["min_tables"]
        r3["expected_tables"] = q["expected_tables"]
        print(f"  No-Schema:  {'✓' if r3['success'] else '✗'} ({r3['latency_ms']}ms)")
        if not r3['success']:
            print(f"    Error: {r3.get('error', '')[:80]}")
        
        # Compute unnecessary JOINs for Full schema condition
        expected_join_count = max(0, len(q["expected_tables"]) - 1)
        r1["unnecessary_joins"] = max(0, r1.get("join_count", 0) - expected_join_count)
        r2["unnecessary_joins"] = max(0, r2.get("join_count", 0) - expected_join_count)
        
        results.append({
            "query": q,
            "llm_full_schema": r1,
            "llm_traversed": r2,
            "llm_no_schema": r3
        })
    
    # Save results
    output_path = Path(__file__).parent / "results" / "extended_schema_experiment.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Make JSON serializable
    for r in results:
        for key in ['llm_full_schema', 'llm_traversed', 'llm_no_schema']:
            if 'rows' not in r[key]:
                r[key]['rows'] = 0
            if 'columns' not in r[key]:
                r[key]['columns'] = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (30 tables)")
    print("=" * 60)
    
    categories = ['simple', 'medium', 'complex', 'very_complex', 'cross_domain', 'aggregation']
    for cat in categories:
        cat_results = [r for r in results if r['query']['category'] == cat]
        full_ok = sum(1 for r in cat_results if r['llm_full_schema']['success'])
        trav_ok = sum(1 for r in cat_results if r['llm_traversed']['success'])
        no_ok = sum(1 for r in cat_results if r['llm_no_schema']['success'])
        total = len(cat_results)
        print(f"  {cat:15s}: Full={full_ok}/{total} ({100*full_ok/total:.0f}%)  Traversed={trav_ok}/{total} ({100*trav_ok/total:.0f}%)  No-Schema={no_ok}/{total} ({100*no_ok/total:.0f}%)")
    
    total = len(results)
    full_total = sum(1 for r in results if r['llm_full_schema']['success'])
    trav_total = sum(1 for r in results if r['llm_traversed']['success'])
    no_total = sum(1 for r in results if r['llm_no_schema']['success'])
    print(f"\n  {'TOTAL':15s}: Full={full_total}/{total} ({100*full_total/total:.0f}%)  Traversed={trav_total}/{total} ({100*trav_total/total:.0f}%)  No-Schema={no_total}/{total} ({100*no_total/total:.0f}%)")
    
    # JOIN analysis
    print("\n" + "=" * 60)
    print("JOIN ANALYSIS (Traversal Effect)")
    print("=" * 60)
    full_joins = [r['llm_full_schema'].get('join_count', 0) for r in results if r['llm_full_schema']['success']]
    trav_joins = [r['llm_traversed'].get('join_count', 0) for r in results if r['llm_traversed']['success']]
    full_unnecessary = [r['llm_full_schema'].get('unnecessary_joins', 0) for r in results if r['llm_full_schema']['success']]
    trav_unnecessary = [r['llm_traversed'].get('unnecessary_joins', 0) for r in results if r['llm_traversed']['success']]
    
    if full_joins:
        print(f"  Full Schema avg JOINs:     {sum(full_joins)/len(full_joins):.1f}")
        print(f"  Traversed avg JOINs:       {sum(trav_joins)/len(trav_joins):.1f}" if trav_joins else "  Traversed: N/A")
        full_with_unnecessary = sum(1 for u in full_unnecessary if u > 0)
        trav_with_unnecessary = sum(1 for u in trav_unnecessary if u > 0)
        print(f"  Full: queries with unnecessary JOINs: {full_with_unnecessary}/{len(full_unnecessary)} ({100*full_with_unnecessary/len(full_unnecessary):.0f}%)")
        if trav_unnecessary:
            print(f"  Traversed: queries with unnecessary JOINs: {trav_with_unnecessary}/{len(trav_unnecessary)} ({100*trav_with_unnecessary/len(trav_unnecessary):.0f}%)")
        print(f"  Full avg unnecessary JOINs:    {sum(full_unnecessary)/len(full_unnecessary):.2f}")
        if trav_unnecessary:
            print(f"  Traversed avg unnecessary JOINs: {sum(trav_unnecessary)/len(trav_unnecessary):.2f}")
    
    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == '__main__':
    run_experiment()
