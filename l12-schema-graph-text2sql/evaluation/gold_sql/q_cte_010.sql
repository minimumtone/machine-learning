-- CTE J: CTE + 弾性物性テーブル到達
-- 「体積弾性率が150 GPa以上のL1₂化合物の純元素基底状態基準に再基準化した生成エネルギーを計算して出して」
WITH stiff_l12 AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           ps.reference_set,
           et.bulk_modulus_vrh
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN calculation cal_et ON cal_et.entry_id = m.entry_id
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND et.bulk_modulus_vrh >= 150
),
enthalpy AS (
    SELECT sl.entry_id, sl.formula, sl.formation_energy_per_atom,
           sl.bulk_modulus_vrh,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM stiff_l12 sl
    JOIN composition comp ON comp.entry_id = sl.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
        AND per.reference_set = sl.reference_set
    GROUP BY sl.entry_id, sl.formula, sl.formation_energy_per_atom,
             sl.bulk_modulus_vrh
)
SELECT formula,
       ROUND(bulk_modulus_vrh::numeric, 1) AS bulk_modulus_vrh,
       ROUND((formation_energy_per_atom - weighted_ref)::numeric, 4)
           AS enthalpy_vs_element_ground_states
FROM enthalpy
ORDER BY enthalpy_vs_element_ground_states ASC
LIMIT 20;
