-- CTE L: CTE + 磁性テーブル到達
-- 「強磁性のL1₂化合物の生成エンタルピーを計算して安定な順に出して」
WITH fm_l12 AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           mp.total_magnetization
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN magnetic_property mp ON mp.entry_id = m.entry_id
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND mp.magnetic_ordering = 'ferromagnetic'
),
enthalpy AS (
    SELECT fl.entry_id, fl.formula, fl.formation_energy_per_atom,
           fl.total_magnetization,
           SUM(comp.atomic_fraction * per.energy_per_atom) AS weighted_ref
    FROM fm_l12 fl
    JOIN composition comp ON comp.entry_id = fl.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    GROUP BY fl.entry_id, fl.formula, fl.formation_energy_per_atom,
             fl.total_magnetization
)
SELECT formula,
       ROUND(total_magnetization::numeric, 3) AS total_magnetization,
       ROUND((formation_energy_per_atom - weighted_ref)::numeric, 4) AS delta_h_f
FROM enthalpy
ORDER BY delta_h_f ASC
LIMIT 20;
