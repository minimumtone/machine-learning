-- CTE D: 多段CTE（2段以上）
-- 「安定なL1₂化合物の生成エンタルピーを計算して、Ni含有のもので最も安定な5件を出して」
WITH stable_l12 AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND ps.is_stable = true
),
enthalpy_calc AS (
    SELECT sl.entry_id, sl.formula, sl.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.energy_per_atom) AS weighted_ref
    FROM stable_l12 sl
    JOIN composition comp ON comp.entry_id = sl.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    GROUP BY sl.entry_id, sl.formula, sl.formation_energy_per_atom
),
ni_bearing AS (
    SELECT ec.entry_id, ec.formula, ec.formation_energy_per_atom,
           ec.formation_energy_per_atom - ec.weighted_ref AS delta_h_f
    FROM enthalpy_calc ec
    WHERE EXISTS (
        SELECT 1 FROM composition c WHERE c.entry_id = ec.entry_id AND c.element = 'Ni'
    )
)
SELECT formula, formation_energy_per_atom,
       ROUND(delta_h_f::numeric, 4) AS corrected_enthalpy
FROM ni_bearing
ORDER BY delta_h_f ASC
LIMIT 5;
