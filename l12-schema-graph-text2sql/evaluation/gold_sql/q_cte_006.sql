-- CTE F: CTE + ウィンドウ関数
-- 「A-site元素ごとに純元素基底状態基準に再基準化した生成エネルギーが最も低いL1₂化合物を1件ずつ出して」
WITH l12_enthalpy AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN composition comp ON comp.entry_id = m.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
        AND per.reference_set = ps.reference_set
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    GROUP BY m.entry_id, m.formula, ps.formation_energy_per_atom
),
ranked AS (
    SELECT le.formula,
           ca.element AS a_site,
           le.formation_energy_per_atom - le.weighted_ref AS e_vs_gs,
           ROW_NUMBER() OVER (
               PARTITION BY ca.element
               ORDER BY le.formation_energy_per_atom - le.weighted_ref ASC
           ) AS rn
    FROM l12_enthalpy le
    JOIN composition ca ON ca.entry_id = le.entry_id AND ca.site_label = 'A-site'
)
SELECT a_site, formula,
       ROUND(e_vs_gs::numeric, 4) AS enthalpy_vs_element_ground_states
FROM ranked
WHERE rn = 1
ORDER BY enthalpy_vs_element_ground_states ASC, a_site ASC;
