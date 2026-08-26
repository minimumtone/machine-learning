-- CTE C: CTE + 集約
-- 「安定なL1₂化合物のA-site元素ごとに平均生成エンタルピーを計算して」
WITH l12_enthalpy AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.energy_per_atom) AS weighted_ref
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN composition comp ON comp.entry_id = m.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND ps.energy_above_hull <= 0.001
    GROUP BY m.entry_id, m.formula, ps.formation_energy_per_atom
),
with_a_site AS (
    SELECT le.formula, le.entry_id,
           le.formation_energy_per_atom - le.weighted_ref AS delta_h_f,
           ca.element AS a_site
    FROM l12_enthalpy le
    JOIN composition ca ON ca.entry_id = le.entry_id AND ca.site_label = 'A-site'
)
SELECT a_site,
       COUNT(*) AS n_compounds,
       ROUND(AVG(delta_h_f)::numeric, 4) AS avg_delta_h_f,
       ROUND(MIN(delta_h_f)::numeric, 4) AS min_delta_h_f
FROM with_a_site
GROUP BY a_site
HAVING COUNT(*) >= 2
ORDER BY avg_delta_h_f ASC;
