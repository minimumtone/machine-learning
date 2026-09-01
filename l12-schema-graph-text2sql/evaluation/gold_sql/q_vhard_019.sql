-- CTE D: 多段CTE（2段以上）
-- 「安定なL1₂化合物の純元素基底状態基準に再基準化した生成エネルギーを計算して、Ni含有のもので再基準化生成エネルギーが低い5件を出して」
WITH stable_l12 AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           ps.reference_set
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND ps.energy_above_hull <= 0.001
),
enthalpy_calc AS (
    SELECT sl.entry_id, sl.formula, sl.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM stable_l12 sl
    JOIN composition comp ON comp.entry_id = sl.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
        AND per.reference_set = sl.reference_set
    GROUP BY sl.entry_id, sl.formula, sl.formation_energy_per_atom
),
ni_bearing AS (
    SELECT ec.entry_id, ec.formula, ec.formation_energy_per_atom,
           ec.formation_energy_per_atom - ec.weighted_ref AS e_vs_gs
    FROM enthalpy_calc ec
    WHERE EXISTS (
        SELECT 1 FROM composition c WHERE c.entry_id = ec.entry_id AND c.element = 'Ni'
    )
)
SELECT formula, formation_energy_per_atom,
       ROUND(e_vs_gs::numeric, 4) AS enthalpy_vs_element_ground_states
FROM ni_bearing
ORDER BY e_vs_gs ASC
LIMIT 5;
