#!/usr/bin/env python3
"""Create gold SQL, expected results, and JSONL dataset for expert-designed 100 queries."""
import json
import os
import psycopg2

DB = dict(dbname='l12_materials', user='l12_user', password='l12_password', host='localhost', port=5432)

EXPERT_QUERIES = [
    # A. 基本検索 (1-10)
    {
        "id": "q_expert_001", "question": "データベースに登録されている化合物の総数を教えて。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry"],
        "sql": "SELECT COUNT(*) AS total_count FROM material_entry LIMIT 10000;"
    },
    {
        "id": "q_expert_002", "question": "BCC_B2構造を持つ化合物を全て表示して。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.prototype, s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'B2' OR s.strukturbericht = 'B2'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_003", "question": "3元素以上で構成される化合物はあるか？",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry"],
        "sql": "SELECT entry_id, formula, number_of_elements FROM material_entry WHERE number_of_elements >= 3 ORDER BY formula LIMIT 10000;"
    },
    {
        "id": "q_expert_004", "question": "化学式にFeが含まれる化合物を一覧にして。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry", "composition"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
WHERE c.element = 'Fe'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_005", "question": "NaCl型構造の化合物は何件あるか。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["structure"],
        "sql": "SELECT COUNT(DISTINCT entry_id) AS nacl_count FROM structure WHERE prototype = 'NaCl' OR strukturbericht = 'B1' LIMIT 10000;"
    },
    {
        "id": "q_expert_006", "question": "Ni3Alのエントリを表示して。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry"],
        "sql": "SELECT * FROM material_entry WHERE formula = 'Ni3Al' OR reduced_formula = 'Ni3Al' LIMIT 10000;"
    },
    {
        "id": "q_expert_007", "question": "空間群Pm-3mの化合物を一覧にして。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.space_group, s.space_group_number
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.space_group = 'Pm-3m' OR s.space_group_number = 221
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_008", "question": "OQMDのエントリ数を教えて。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry"],
        "sql": "SELECT COUNT(*) AS oqmd_count FROM material_entry WHERE source_db = 'OQMD' LIMIT 10000;"
    },
    {
        "id": "q_expert_009", "question": "化学系（chemical_system）にTiを含む化合物のformulaを表示して。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry"],
        "sql": "SELECT entry_id, formula, chemical_system FROM material_entry WHERE chemical_system LIKE '%Ti%' ORDER BY formula LIMIT 10000;"
    },
    {
        "id": "q_expert_010", "question": "BiF3型構造の化合物を全て出して。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.prototype, s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'BiF3' OR s.strukturbericht = 'D03'
ORDER BY m.formula
LIMIT 10000;"""
    },
    # B. 組成・元素に関する質問 (11-20)
    {
        "id": "q_expert_011", "question": "Ptを25%以上含むL12型化合物を表示して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "composition", "structure"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, c.element, c.atomic_fraction
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Pt' AND c.atomic_fraction >= 0.25
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_012", "question": "RhとAlの両方を含む化合物を教えて。まずは構造のリストが欲しい。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "composition", "structure"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, s.prototype, s.crystal_system
FROM material_entry m
JOIN composition c_rh ON c_rh.entry_id = m.entry_id AND c_rh.element = 'Rh'
JOIN composition c_al ON c_al.entry_id = m.entry_id AND c_al.element = 'Al'
JOIN structure s ON s.entry_id = m.entry_id
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_013", "question": "希土類元素を含むL12型化合物はあるか。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["material_entry", "composition", "structure"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, c.element
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element IN ('La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Sc','Y')
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_014", "question": "遷移金属のみで構成されるL12型化合物を抽出して。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["material_entry", "composition", "structure"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND NOT EXISTS (
    SELECT 1 FROM composition c
    JOIN element e ON e.symbol = c.element
    WHERE c.entry_id = m.entry_id AND e.category NOT IN ('transition_metal')
  )
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_015", "question": "Cuの原子分率が0.75であるL12型化合物を出して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "composition", "structure"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, c.atomic_fraction
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Cu' AND c.atomic_fraction = 0.75
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_016", "question": "4d遷移金属をBサイトに持つL12型化合物を表示して。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["material_entry", "composition", "structure"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, c.element, c.atomic_fraction
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element IN ('Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd')
  AND c.atomic_fraction <= 0.25
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_017", "question": "データベース中に登場する元素を全て一覧にして。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["composition"],
        "sql": "SELECT DISTINCT element FROM composition ORDER BY element LIMIT 10000;"
    },
    {
        "id": "q_expert_018", "question": "Mnを含む化合物はどのプロトタイプに多いか。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["composition", "structure"],
        "sql": """SELECT s.prototype, COUNT(DISTINCT m.entry_id) AS cnt
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Mn'
GROUP BY s.prototype
ORDER BY cnt DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_019", "question": "原子番号が40以上の元素を含むL12型化合物を教えて。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "composition", "structure", "element"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, c.element, e.atomic_number
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN element e ON e.symbol = c.element
WHERE e.atomic_number >= 40
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_020", "question": "VとAlの両方を含む安定な化合物を出して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "composition", "phase_stability"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, ps.energy_above_hull
FROM material_entry m
JOIN composition c_v ON c_v.entry_id = m.entry_id AND c_v.element = 'V'
JOIN composition c_al ON c_al.entry_id = m.entry_id AND c_al.element = 'Al'
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE ps.is_stable = TRUE
ORDER BY m.formula
LIMIT 10000;"""
    },
    # C. 構造・格子定数に関する質問 (21-30)
    {
        "id": "q_expert_021", "question": "格子定数aが3.50〜3.60Åの範囲にあるL12型化合物をリストして。",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.lattice_a BETWEEN 3.50 AND 3.60
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY s.lattice_a
LIMIT 10000;"""
    },
    {
        "id": "q_expert_022", "question": "B2構造でcubic以外の結晶系を持つものはあるか。",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.crystal_system
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2')
  AND s.crystal_system != 'cubic'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_023", "question": "体積が最小のL12化合物はどれか。",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.volume_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.volume_per_atom IS NOT NULL
ORDER BY s.volume_per_atom ASC
LIMIT 1;"""
    },
    {
        "id": "q_expert_024", "question": "格子定数aとcが異なる化合物を表示して。（正方晶歪み）",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.lattice_a, s.lattice_c, s.crystal_system
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.lattice_a IS NOT NULL AND s.lattice_c IS NOT NULL
  AND ABS(s.lattice_a - s.lattice_c) > 0.01
ORDER BY ABS(s.lattice_a - s.lattice_c) DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_025", "question": "Strukturbericht記号がL12の化合物をprototypeと共に表示して。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.prototype, s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.strukturbericht = 'L12'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_026", "question": "空間群番号225の化合物は何件か。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["structure"],
        "sql": "SELECT COUNT(DISTINCT entry_id) AS cnt FROM structure WHERE space_group_number = 225 LIMIT 10000;"
    },
    {
        "id": "q_expert_027", "question": "NiAs型構造でヘキサゴナルの結晶系のもののうち、格子定数cが最大のもの",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.lattice_c
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'NiAs' OR s.strukturbericht = 'B81')
  AND s.crystal_system = 'hexagonal'
ORDER BY s.lattice_c DESC
LIMIT 1;"""
    },
    {
        "id": "q_expert_028", "question": "格子定数aが4.0Å以上のB2化合物を表示して。",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2')
  AND s.lattice_a >= 4.0
ORDER BY s.lattice_a
LIMIT 10000;"""
    },
    {
        "id": "q_expert_029", "question": "L12型化合物の格子定数aの平均値と標準偏差を教えて。",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["structure"],
        "sql": """SELECT AVG(s.lattice_a) AS avg_a, STDDEV(s.lattice_a) AS std_a
FROM structure s
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
LIMIT 10000;"""
    },
    {
        "id": "q_expert_030", "question": "全プロトタイプについて、それぞれの登録件数を教えて。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["structure"],
        "sql": """SELECT prototype, COUNT(*) AS cnt
FROM structure
GROUP BY prototype
ORDER BY cnt DESC
LIMIT 10000;"""
    },
    # D. 安定性・形成エネルギー (31-40)
    {
        "id": "q_expert_031", "question": "convex_hullを構成するL12化合物（完全安定）を全て出して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "structure", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, ps.energy_above_hull, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
ORDER BY ps.formation_energy_per_atom
LIMIT 10000;"""
    },
    {
        "id": "q_expert_032", "question": "形成エネルギーが正の化合物はあるか。あればリストして。",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["material_entry", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, ps.formation_energy_per_atom
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE ps.formation_energy_per_atom > 0
ORDER BY ps.formation_energy_per_atom DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_033", "question": "B2構造で安定な化合物のうち、形成エネルギーが負のものを表示して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "structure", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, ps.formation_energy_per_atom, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2')
  AND ps.is_stable = TRUE
  AND ps.formation_energy_per_atom < 0
ORDER BY ps.formation_energy_per_atom
LIMIT 10000;"""
    },
    {
        "id": "q_expert_034", "question": "準安定（energy_above_hull > 0 かつ < 0.1 eV/atom）なL12化合物",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "structure", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull > 0 AND ps.energy_above_hull < 0.1
ORDER BY ps.energy_above_hull
LIMIT 10000;"""
    },
    {
        "id": "q_expert_035", "question": "NaCl型構造で不安定な化合物の割合を教えて。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["structure", "phase_stability"],
        "sql": """SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN ps.is_stable = FALSE THEN 1 ELSE 0 END) AS unstable,
  ROUND(100.0 * SUM(CASE WHEN ps.is_stable = FALSE THEN 1 ELSE 0 END) / COUNT(*), 2) AS unstable_pct
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'NaCl' OR s.strukturbericht = 'B1'
LIMIT 10000;"""
    },
    {
        "id": "q_expert_036", "question": "バンドギャップが0より大きいL12型化合物（半導体的なもの）はあるか。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "structure", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, ps.band_gap
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.band_gap > 0
ORDER BY ps.band_gap DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_037", "question": "安定なL12化合物のうち、バンドギャップが0のもの（金属的なもの）",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "structure", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, ps.band_gap
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND (ps.band_gap = 0 OR ps.band_gap IS NULL)
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_038", "question": "Ni-Al系の化合物を抽出してconvex_hullからの距離を全て表示して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE m.chemical_system = 'Al-Ni' OR m.chemical_system = 'Ni-Al'
ORDER BY ps.energy_above_hull
LIMIT 10000;"""
    },
    {
        "id": "q_expert_039", "question": "形成エネルギーが-0.5 eV/atom以下の極めて安定な化合物を表示して",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["material_entry", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, ps.formation_energy_per_atom
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE ps.formation_energy_per_atom <= -0.5
ORDER BY ps.formation_energy_per_atom
LIMIT 10000;"""
    },
    {
        "id": "q_expert_040", "question": "プロトタイプ別の平均energy_above_hullを比較して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["structure", "phase_stability"],
        "sql": """SELECT s.prototype, AVG(ps.energy_above_hull) AS avg_ehull, COUNT(*) AS cnt
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
GROUP BY s.prototype
ORDER BY avg_ehull
LIMIT 10000;"""
    },
    # E. DFT計算・物性 (41-50)
    {
        "id": "q_expert_041", "question": "GGA-PBE以外の汎関数で計算されたエントリはあるか。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry", "calculation"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, ca.functional
FROM material_entry m
JOIN calculation ca ON ca.entry_id = m.entry_id
WHERE ca.functional IS NOT NULL AND ca.functional != 'GGA-PBE' AND ca.functional != 'PBE'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_042", "question": "バルクモジュラスが200GPa以上のL12化合物を出して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "elastic_tensor"],
        "sql": """SELECT m.entry_id, m.formula, et.bulk_modulus_vrh
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND et.bulk_modulus_vrh >= 200
ORDER BY et.bulk_modulus_vrh DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_043", "question": "せん断弾性率が最も高いL12化合物TOP3を教えて。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "elastic_tensor"],
        "sql": """SELECT m.entry_id, m.formula, et.shear_modulus_vrh
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY et.shear_modulus_vrh DESC
LIMIT 3;"""
    },
    {
        "id": "q_expert_044", "question": "ヤング率が記録されている化合物を全て表示して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "elastic_tensor"],
        "sql": """SELECT m.entry_id, m.formula, et.youngs_modulus
FROM material_entry m
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE et.youngs_modulus IS NOT NULL
ORDER BY et.youngs_modulus DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_045", "question": "ポアソン比が0.3以上の化合物をリストして。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "elastic_tensor"],
        "sql": """SELECT m.entry_id, m.formula, et.poisson_ratio
FROM material_entry m
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE et.poisson_ratio >= 0.3
ORDER BY et.poisson_ratio DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_046", "question": "バルクモジュラスとせん断弾性率の比（B/G比）が2以上のL12化合物",
        "difficulty": "very_hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "elastic_tensor"],
        "sql": """SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, et.shear_modulus_vrh,
  ROUND((et.bulk_modulus_vrh / NULLIF(et.shear_modulus_vrh, 0))::numeric, 3) AS bg_ratio
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND et.shear_modulus_vrh > 0
  AND (et.bulk_modulus_vrh / et.shear_modulus_vrh) >= 2.0
ORDER BY bg_ratio DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_047", "question": "DFT計算で弾性的に不安定（elastic tensor is_stable=false）とされる化合物はあるか。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "elastic_tensor"],
        "sql": """SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, et.shear_modulus_vrh
FROM material_entry m
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE et.is_stable = FALSE
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_048", "question": "格子定数にc/aが大きいものを探してほしい。",
        "difficulty": "medium", "hop_count": 1,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.lattice_a, s.lattice_c,
  ROUND((s.lattice_c / NULLIF(s.lattice_a, 0))::numeric, 4) AS c_over_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.lattice_a > 0 AND s.lattice_c IS NOT NULL
ORDER BY (s.lattice_c / NULLIF(s.lattice_a, 0)) DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_049", "question": "Ni3Alのバルクモジュラスと形成エネルギーを教えて。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "elastic_tensor", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, ps.formation_energy_per_atom
FROM material_entry m
JOIN elastic_tensor et ON et.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE m.formula = 'Ni3Al'
LIMIT 10000;"""
    },
    {
        "id": "q_expert_050", "question": "磁気モーメントが0でないL12化合物を全て出して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "magnetic_property"],
        "sql": """SELECT m.entry_id, m.formula, mp.total_magnetization, mp.magnetic_ordering
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mp.total_magnetization != 0 AND mp.total_magnetization IS NOT NULL
ORDER BY mp.total_magnetization DESC
LIMIT 10000;"""
    },
    # F. 電子構造・磁性・熱特性 (51-60)
    {
        "id": "q_expert_051", "question": "フェルミ面でのDOSが最も高いL12化合物は。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "density_of_states"],
        "sql": """SELECT m.entry_id, m.formula, dos.total_dos_at_fermi
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN density_of_states dos ON dos.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY dos.total_dos_at_fermi DESC
LIMIT 1;"""
    },
    {
        "id": "q_expert_052", "question": "直接バンドギャップを持つ化合物はあるか。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "band_structure"],
        "sql": """SELECT m.entry_id, m.formula, bs.band_gap_type, bs.cbm_energy, bs.vbm_energy
FROM material_entry m
JOIN band_structure bs ON bs.entry_id = m.entry_id
WHERE bs.is_direct_gap = TRUE
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_053", "question": "スピン偏極計算がされている化合物を表示して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "density_of_states"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN density_of_states dos ON dos.entry_id = m.entry_id
WHERE dos.spin_polarized = TRUE
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_054", "question": "強磁性（magnetic_ordering = 'ferromagnetic'）のL12化合物を出して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "magnetic_property"],
        "sql": """SELECT m.entry_id, m.formula, mp.total_magnetization, mp.curie_temperature_k
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mp.magnetic_ordering = 'ferromagnetic'
ORDER BY mp.total_magnetization DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_055", "question": "キュリー温度が最も高い化合物はどれか。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "magnetic_property"],
        "sql": """SELECT m.entry_id, m.formula, mp.curie_temperature_k
FROM material_entry m
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE mp.curie_temperature_k IS NOT NULL
ORDER BY mp.curie_temperature_k DESC
LIMIT 1;"""
    },
    {
        "id": "q_expert_056", "question": "デバイ温度が500K以上の化合物を教えて。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "thermal_property"],
        "sql": """SELECT m.entry_id, m.formula, tp.debye_temperature_k
FROM material_entry m
JOIN thermal_property tp ON tp.entry_id = m.entry_id
WHERE tp.debye_temperature_k >= 500
ORDER BY tp.debye_temperature_k DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_057", "question": "熱伝導率が記録されているL12化合物を表示して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "thermal_property"],
        "sql": """SELECT m.entry_id, m.formula, tp.thermal_conductivity
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN thermal_property tp ON tp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND tp.thermal_conductivity IS NOT NULL
ORDER BY tp.thermal_conductivity DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_058", "question": "グリュナイゼン定数が2以上の化合物はあるか。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "thermal_property"],
        "sql": """SELECT m.entry_id, m.formula, tp.gruneisen_parameter
FROM material_entry m
JOIN thermal_property tp ON tp.entry_id = m.entry_id
WHERE tp.gruneisen_parameter >= 2.0
ORDER BY tp.gruneisen_parameter DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_059", "question": "磁気異方性エネルギーが最も大きいL12化合物を出して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "magnetic_property"],
        "sql": """SELECT m.entry_id, m.formula, mp.magnetic_anisotropy_energy
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mp.magnetic_anisotropy_energy IS NOT NULL
ORDER BY mp.magnetic_anisotropy_energy DESC
LIMIT 1;"""
    },
    {
        "id": "q_expert_060", "question": "金属的（is_metallic=true）かつ強磁性のL12化合物を探して。",
        "difficulty": "very_hard", "hop_count": 4,
        "expected_tables": ["material_entry", "structure", "density_of_states", "magnetic_property"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, dos.total_dos_at_fermi, mp.magnetic_ordering
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN density_of_states dos ON dos.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND dos.is_metallic = TRUE
  AND mp.magnetic_ordering = 'ferromagnetic'
ORDER BY m.formula
LIMIT 10000;"""
    },
    # G. 表面・粒界・欠陥 (61-70)
    {
        "id": "q_expert_061", "question": "(111)面の表面エネルギーが最も低いL12化合物を教えて。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "surface_energy"],
        "sql": """SELECT m.entry_id, m.formula, se.surface_energy_j_m2
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND se.miller_index = '111'
ORDER BY se.surface_energy_j_m2 ASC
LIMIT 1;"""
    },
    {
        "id": "q_expert_062", "question": "仕事関数が5 eV以上のL12化合物を出して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "surface_energy"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, se.work_function, se.miller_index
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND se.work_function >= 5.0
ORDER BY se.work_function DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_063", "question": "表面再構成が起こる化合物はあるか。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "surface_energy"],
        "sql": """SELECT m.entry_id, m.formula, se.miller_index, se.is_reconstructed
FROM material_entry m
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE se.is_reconstructed = TRUE
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_064", "question": "Σ5粒界エネルギーが記録されている化合物を表示して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "grain_boundary"],
        "sql": """SELECT m.entry_id, m.formula, gb.gb_energy_j_m2, gb.rotation_axis
FROM material_entry m
JOIN grain_boundary gb ON gb.entry_id = m.entry_id
WHERE gb.sigma_value = 5
ORDER BY gb.gb_energy_j_m2
LIMIT 10000;"""
    },
    {
        "id": "q_expert_065", "question": "空孔形成エネルギーが最も低いL12化合物は。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "material_defect", "defect_type"],
        "sql": """SELECT m.entry_id, m.formula, md.formation_energy
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_defect md ON md.entry_id = m.entry_id
JOIN defect_type dt ON dt.defect_type_id = md.defect_type_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND dt.category = 'vacancy'
ORDER BY md.formation_energy ASC
LIMIT 1;"""
    },
    {
        "id": "q_expert_066", "question": "アンチサイト欠陥の情報があるL12化合物を教えて。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "material_defect", "defect_type"],
        "sql": """SELECT m.entry_id, m.formula, md.formation_energy, md.site
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_defect md ON md.entry_id = m.entry_id
JOIN defect_type dt ON dt.defect_type_id = md.defect_type_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND dt.category = 'antisite'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_067", "question": "ドーパント元素としてBが使われている化合物はあるか。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "material_defect", "element"],
        "sql": """SELECT m.entry_id, m.formula, e.symbol AS dopant
FROM material_entry m
JOIN material_defect md ON md.entry_id = m.entry_id
JOIN element e ON e.element_id = md.dopant_element_id
WHERE e.symbol = 'B'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_068", "question": "格子間原子（interstitial）欠陥の情報がある化合物を表示して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "material_defect", "defect_type"],
        "sql": """SELECT m.entry_id, m.formula, md.formation_energy, md.site
FROM material_entry m
JOIN material_defect md ON md.entry_id = m.entry_id
JOIN defect_type dt ON dt.defect_type_id = md.defect_type_id
WHERE dt.category = 'interstitial'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_069", "question": "(100)面と(110)面の表面エネルギーを両方持つ化合物を出して。",
        "difficulty": "very_hard", "hop_count": 2,
        "expected_tables": ["material_entry", "surface_energy"],
        "sql": """SELECT m.entry_id, m.formula, se100.surface_energy_j_m2 AS se_100, se110.surface_energy_j_m2 AS se_110
FROM material_entry m
JOIN surface_energy se100 ON se100.entry_id = m.entry_id AND se100.miller_index = '100'
JOIN surface_energy se110 ON se110.entry_id = m.entry_id AND se110.miller_index = '110'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_070", "question": "Vegard則からの乖離が大きいL12、B2化合物の情報がしりたい。",
        "difficulty": "very_hard", "hop_count": 2,
        "expected_tables": ["material_entry", "structure"],
        "sql": """SELECT m.entry_id, m.formula, s.prototype, s.lattice_a, s.volume_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype IN ('L12', 'B2') OR s.strukturbericht IN ('L12', 'B2')
ORDER BY s.volume_per_atom DESC
LIMIT 10000;"""
    },
    # H. 文献・合成・応用 (71-80)
    {
        "id": "q_expert_071", "question": "実験で合成されたことのあるL12化合物はどれか。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "material_synthesis"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ms.success = TRUE
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_072", "question": "アーク溶解法で合成されたL12化合物を表示して。",
        "difficulty": "hard", "hop_count": 4,
        "expected_tables": ["material_entry", "structure", "material_synthesis", "synthesis_method"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
JOIN synthesis_method sm ON sm.synthesis_id = ms.synthesis_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND sm.method_name = 'Arc Melting'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_073", "question": "1000K以上の温度で合成されたL12化合物を出して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "material_synthesis"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, ms.temperature_k
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ms.temperature_k >= 1000
ORDER BY ms.temperature_k DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_074", "question": "文献参照が3件以上あるL12化合物を教えて。",
        "difficulty": "very_hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "material_reference"],
        "sql": """SELECT m.entry_id, m.formula, COUNT(mr.reference_id) AS ref_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_reference mr ON mr.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
GROUP BY m.entry_id, m.formula
HAVING COUNT(mr.reference_id) >= 3
ORDER BY ref_count DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_075", "question": "高温超合金の応用領域に分類されている化合物を全て表示して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "material_application", "application_domain"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, ad.domain_name
FROM material_entry m
JOIN material_application ma ON ma.entry_id = m.entry_id
JOIN application_domain ad ON ad.domain_id = ma.domain_id
WHERE ad.domain_name LIKE '%Superalloy%' OR ad.domain_name LIKE '%High-Temperature%'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_076", "question": "2020年以降の文献で報告されたL12化合物を出して。",
        "difficulty": "very_hard", "hop_count": 4,
        "expected_tables": ["material_entry", "structure", "material_reference", "literature_reference"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, lr.year, lr.title
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_reference mr ON mr.entry_id = m.entry_id
JOIN literature_reference lr ON lr.reference_id = mr.reference_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND lr.year >= 2020
ORDER BY lr.year DESC, m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_077", "question": "実験値と計算値の両方がある化合物を表示して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "calculation", "experimental_measurement"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN calculation ca ON ca.entry_id = m.entry_id
JOIN experimental_measurement em ON em.entry_id = m.entry_id
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_078", "question": "ボールミリングで合成された化合物はあるか。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "material_synthesis", "synthesis_method"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
JOIN synthesis_method sm ON sm.synthesis_id = ms.synthesis_id
WHERE sm.method_name = 'Ball Milling'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_079", "question": "耐熱材料として分類されている化合物を表示して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "material_application", "application_domain"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, ad.domain_name
FROM material_entry m
JOIN material_application ma ON ma.entry_id = m.entry_id
JOIN application_domain ad ON ad.domain_id = ma.domain_id
WHERE ad.domain_name LIKE '%Heat%' OR ad.domain_name LIKE '%Thermal%' OR ad.domain_name LIKE '%High-Temperature%'
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_080", "question": "DOIが記録されている文献の一覧を出して。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["literature_reference"],
        "sql": """SELECT reference_id, doi, title, authors, journal, year
FROM literature_reference
WHERE doi IS NOT NULL
ORDER BY year DESC
LIMIT 10000;"""
    },
    # I. 材料設計・スクリーニング（複合条件）(81-90)
    {
        "id": "q_expert_081", "question": "Ni3Alと格子定数が0.05Å以内で、かつ安定で、バルクモジュラスが150GPa以上の化合物を探して。",
        "difficulty": "very_hard", "hop_count": 4,
        "expected_tables": ["material_entry", "structure", "phase_stability", "elastic_tensor"],
        "sql": """SELECT m.entry_id, m.formula, s.lattice_a, ps.energy_above_hull, et.bulk_modulus_vrh
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE ABS(s.lattice_a - 3.572) < 0.05
  AND ps.is_stable = TRUE
  AND et.bulk_modulus_vrh >= 150
ORDER BY ABS(s.lattice_a - 3.572)
LIMIT 10000;"""
    },
    {
        "id": "q_expert_082", "question": "γ'相候補として有望な、安定で準安定なL12化合物を形成エネルギー順にランキングして。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["material_entry", "structure", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, ps.formation_energy_per_atom, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.1
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_083", "question": "Coを含む安定なL12化合物で、バルクモジュラスが180GPa以上かつデバイ温度400K以上のものを出して。",
        "difficulty": "very_hard", "hop_count": 5,
        "expected_tables": ["material_entry", "composition", "structure", "phase_stability", "elastic_tensor", "thermal_property"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, et.bulk_modulus_vrh, tp.debye_temperature_k
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
JOIN thermal_property tp ON tp.entry_id = m.entry_id
WHERE c.element = 'Co'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND et.bulk_modulus_vrh >= 180
  AND tp.debye_temperature_k >= 400
ORDER BY et.bulk_modulus_vrh DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_084", "question": "格子定数が3.56Å±0.03Åで、せん断弾性率が70GPa以上で、安定なL12化合物を探して。",
        "difficulty": "very_hard", "hop_count": 4,
        "expected_tables": ["material_entry", "structure", "phase_stability", "elastic_tensor"],
        "sql": """SELECT m.entry_id, m.formula, s.lattice_a, et.shear_modulus_vrh, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.lattice_a BETWEEN 3.53 AND 3.59
  AND et.shear_modulus_vrh >= 70
  AND ps.is_stable = TRUE
ORDER BY et.shear_modulus_vrh DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_085", "question": "強磁性かつ弾性的に安定なL12化合物を全て表示して。",
        "difficulty": "very_hard", "hop_count": 4,
        "expected_tables": ["material_entry", "structure", "magnetic_property", "elastic_tensor"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, mp.magnetic_ordering, et.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mp.magnetic_ordering = 'ferromagnetic'
  AND et.is_stable = TRUE
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_086", "question": "実験合成実績があり、かつDFT計算でも安定と判定されているL12化合物を出して。",
        "difficulty": "very_hard", "hop_count": 4,
        "expected_tables": ["material_entry", "structure", "phase_stability", "material_synthesis"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND ms.success = TRUE
ORDER BY m.formula
LIMIT 10000;"""
    },
    {
        "id": "q_expert_087", "question": "Ni3Alよりバルクモジュラスが高く、かつコンベックスハルからの距離が0.01以下の化合物を抽出して。",
        "difficulty": "very_hard", "hop_count": 3,
        "expected_tables": ["material_entry", "elastic_tensor", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, ps.energy_above_hull
FROM material_entry m
JOIN elastic_tensor et ON et.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE et.bulk_modulus_vrh > (
    SELECT et2.bulk_modulus_vrh FROM elastic_tensor et2
    JOIN material_entry m2 ON m2.entry_id = et2.entry_id
    WHERE m2.formula = 'Ni3Al' LIMIT 1
  )
  AND ps.energy_above_hull <= 0.01
ORDER BY et.bulk_modulus_vrh DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_088", "question": "元素数が2の安定な化合物を、プロトタイプ別に件数を集計して。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["material_entry", "structure", "phase_stability"],
        "sql": """SELECT s.prototype, COUNT(*) AS cnt
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE m.number_of_elements = 2
  AND ps.is_stable = TRUE
GROUP BY s.prototype
ORDER BY cnt DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_089", "question": "L12構造のうち、ポアソン比が低く（<0.25）延性が低いと予想される化合物を出して。",
        "difficulty": "hard", "hop_count": 3,
        "expected_tables": ["material_entry", "structure", "elastic_tensor"],
        "sql": """SELECT m.entry_id, m.formula, et.poisson_ratio, et.bulk_modulus_vrh, et.shear_modulus_vrh
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND et.poisson_ratio < 0.25
ORDER BY et.poisson_ratio
LIMIT 10000;"""
    },
    {
        "id": "q_expert_090", "question": "表面エネルギーが低い（<1.5 J/m²）かつ安定なL12化合物を全て表示して。",
        "difficulty": "very_hard", "hop_count": 4,
        "expected_tables": ["material_entry", "structure", "phase_stability", "surface_energy"],
        "sql": """SELECT DISTINCT m.entry_id, m.formula, se.miller_index, se.surface_energy_j_m2, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND se.surface_energy_j_m2 < 1.5
ORDER BY se.surface_energy_j_m2
LIMIT 10000;"""
    },
    # J. 比較・統計・俯瞰的な質問 (91-100)
    {
        "id": "q_expert_091", "question": "L12とB2で、安定な化合物の割合を比較して。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["structure", "phase_stability"],
        "sql": """SELECT s.prototype,
  COUNT(*) AS total,
  SUM(CASE WHEN ps.is_stable = TRUE THEN 1 ELSE 0 END) AS stable,
  ROUND(100.0 * SUM(CASE WHEN ps.is_stable = TRUE THEN 1 ELSE 0 END) / COUNT(*), 2) AS stable_pct
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
WHERE s.prototype IN ('L12', 'B2')
GROUP BY s.prototype
ORDER BY s.prototype
LIMIT 10000;"""
    },
    {
        "id": "q_expert_092", "question": "プロトタイプ別の平均バルクモジュラスを教えて。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["structure", "elastic_tensor"],
        "sql": """SELECT s.prototype, AVG(et.bulk_modulus_vrh) AS avg_bulk, COUNT(*) AS cnt
FROM structure s
JOIN elastic_tensor et ON et.entry_id = s.entry_id
GROUP BY s.prototype
ORDER BY avg_bulk DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_093", "question": "Aサイト元素ごとのL12化合物数を集計して。どの元素が最多か。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["material_entry", "composition", "structure"],
        "sql": """SELECT c.element, COUNT(DISTINCT m.entry_id) AS cnt
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.atomic_fraction >= 0.70
GROUP BY c.element
ORDER BY cnt DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_094", "question": "格子定数と形成エネルギーに相関はあるか？全L12化合物のデータを出して。",
        "difficulty": "medium", "hop_count": 2,
        "expected_tables": ["material_entry", "structure", "phase_stability"],
        "sql": """SELECT m.entry_id, m.formula, s.lattice_a, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.lattice_a IS NOT NULL
  AND ps.formation_energy_per_atom IS NOT NULL
ORDER BY s.lattice_a
LIMIT 10000;"""
    },
    {
        "id": "q_expert_095", "question": "安定なL12化合物と不安定なL12化合物で、平均バルクモジュラスに差はあるか。",
        "difficulty": "very_hard", "hop_count": 3,
        "expected_tables": ["structure", "phase_stability", "elastic_tensor"],
        "sql": """SELECT
  CASE WHEN ps.is_stable THEN 'stable' ELSE 'unstable' END AS stability,
  AVG(et.bulk_modulus_vrh) AS avg_bulk,
  COUNT(*) AS cnt
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
JOIN elastic_tensor et ON et.entry_id = s.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
GROUP BY ps.is_stable
LIMIT 10000;"""
    },
    {
        "id": "q_expert_096", "question": "化学系（chemical_system）ごとのエントリ数TOP10を教えて。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["material_entry"],
        "sql": """SELECT chemical_system, COUNT(*) AS cnt
FROM material_entry
GROUP BY chemical_system
ORDER BY cnt DESC
LIMIT 10;"""
    },
    {
        "id": "q_expert_097", "question": "DFT計算手法（method）ごとのエントリ数を出して。",
        "difficulty": "easy", "hop_count": 1,
        "expected_tables": ["calculation"],
        "sql": """SELECT method, COUNT(*) AS cnt
FROM calculation
GROUP BY method
ORDER BY cnt DESC
LIMIT 10000;"""
    },
    {
        "id": "q_expert_098", "question": "L12型化合物のバンドギャップの分布を教えて。0のものとそうでないものを分けてほしい。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["structure", "phase_stability"],
        "sql": """SELECT
  CASE WHEN ps.band_gap = 0 OR ps.band_gap IS NULL THEN 'metallic (gap=0)' ELSE 'non-zero gap' END AS gap_category,
  COUNT(*) AS cnt,
  AVG(ps.band_gap) AS avg_gap
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
GROUP BY gap_category
LIMIT 10000;"""
    },
    {
        "id": "q_expert_099", "question": "格子定数aの分布を0.1Å刻みでヒストグラム的に表示するデータが欲しい。",
        "difficulty": "hard", "hop_count": 1,
        "expected_tables": ["structure"],
        "sql": """SELECT
  FLOOR(lattice_a * 10) / 10.0 AS bin_start,
  COUNT(*) AS cnt
FROM structure
WHERE lattice_a IS NOT NULL
GROUP BY FLOOR(lattice_a * 10)
ORDER BY bin_start
LIMIT 10000;"""
    },
    {
        "id": "q_expert_100", "question": "全プロトタイプについて、安定/不安定/準安定の件数内訳を教えて。",
        "difficulty": "hard", "hop_count": 2,
        "expected_tables": ["structure", "phase_stability"],
        "sql": """SELECT s.prototype,
  SUM(CASE WHEN ps.is_stable = TRUE THEN 1 ELSE 0 END) AS stable,
  SUM(CASE WHEN ps.is_stable = FALSE AND ps.energy_above_hull < 0.1 THEN 1 ELSE 0 END) AS metastable,
  SUM(CASE WHEN ps.is_stable = FALSE AND ps.energy_above_hull >= 0.1 THEN 1 ELSE 0 END) AS unstable,
  COUNT(*) AS total
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
GROUP BY s.prototype
ORDER BY total DESC
LIMIT 10000;"""
    },
]


def main():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    gold_dir = os.path.join(os.path.dirname(__file__), "gold_sql")
    results_dir = os.path.join(os.path.dirname(__file__), "expected_results")
    os.makedirs(gold_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    dataset_lines = []
    success_count = 0
    fail_count = 0

    for q in EXPERT_QUERIES:
        qid = q["id"]
        sql = q["sql"]

        # Write gold SQL
        with open(os.path.join(gold_dir, f"{qid}.sql"), "w") as f:
            f.write(sql + "\n")

        # Execute gold SQL and save expected results
        try:
            cur.execute("SET statement_timeout = '10s'")
            cur.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            result = {
                "query_id": qid,
                "columns": columns,
                "row_count": len(rows),
                "rows": [[str(v) if v is not None else None for v in r] for r in rows],
            }
            with open(os.path.join(results_dir, f"{qid}.json"), "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            success_count += 1
            print(f"  {qid}: {len(rows)} rows")
        except Exception as e:
            conn.rollback()
            print(f"  {qid}: ERROR - {e}")
            # Write empty result
            with open(os.path.join(results_dir, f"{qid}.json"), "w") as f:
                json.dump({"query_id": qid, "columns": [], "row_count": 0, "rows": []}, f)
            fail_count += 1

        # Build JSONL entry
        entry = {
            "id": qid,
            "question": q["question"],
            "difficulty": q["difficulty"],
            "hop_count": q["hop_count"],
            "expected_tables": q["expected_tables"],
            "expected_columns": [],  # Will be populated from results
            "required_join_path": [],
            "gold_sql_path": f"gold_sql/{qid}.sql",
            "expected_result_path": f"expected_results/{qid}.json",
        }
        dataset_lines.append(json.dumps(entry, ensure_ascii=False))

    # Write JSONL dataset
    jsonl_path = os.path.join(os.path.dirname(__file__), "expert_evaluation_dataset.jsonl")
    with open(jsonl_path, "w") as f:
        f.write("\n".join(dataset_lines) + "\n")

    conn.close()
    print(f"\nDone: {success_count} success, {fail_count} failed")
    print(f"Dataset: {jsonl_path}")


if __name__ == "__main__":
    main()
