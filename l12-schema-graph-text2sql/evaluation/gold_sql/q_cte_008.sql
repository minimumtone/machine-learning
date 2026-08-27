-- CTE H: CTE + HAVING（グループフィルタ）
-- 「B-site元素ごとの平均生成エンタルピーが-0.4 eV/atom未満のグループを出して」
WITH l12_enthalpy AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN composition comp ON comp.entry_id = m.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND ps.energy_above_hull <= 0.001
    GROUP BY m.entry_id, m.formula, ps.formation_energy_per_atom
)
SELECT cb.element AS b_site,
       COUNT(*) AS n_compounds,
       ROUND(AVG(le.formation_energy_per_atom - le.weighted_ref)::numeric, 4)
           AS avg_delta_h_f
FROM l12_enthalpy le
JOIN composition cb ON cb.entry_id = le.entry_id AND cb.site_label = 'B-site'
GROUP BY cb.element
HAVING AVG(le.formation_energy_per_atom - le.weighted_ref) < -0.4
ORDER BY avg_delta_h_f ASC;
