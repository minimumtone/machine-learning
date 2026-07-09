-- CTE I: 多段CTE + 2段集約
-- 「元素ごとに安定なL1₂化合物の数と平均生成エンタルピーを計算し、化合物数上位5元素を出して」
WITH l12_enthalpy AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.energy_per_atom) AS weighted_ref
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN composition comp ON comp.entry_id = m.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND ps.is_stable = true
    GROUP BY m.entry_id, m.formula, ps.formation_energy_per_atom
),
per_element AS (
    SELECT c.element,
           COUNT(DISTINCT le.entry_id) AS n_compounds,
           AVG(le.formation_energy_per_atom - le.weighted_ref) AS avg_delta_h_f
    FROM l12_enthalpy le
    JOIN composition c ON c.entry_id = le.entry_id
    GROUP BY c.element
)
SELECT element, n_compounds,
       ROUND(avg_delta_h_f::numeric, 4) AS avg_delta_h_f
FROM per_element
ORDER BY n_compounds DESC, element ASC
LIMIT 5;
