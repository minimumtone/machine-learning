-- CTE H: CTE + HAVING（グループフィルタ）
-- 「B-site元素ごとの純元素基底状態基準に再基準化した生成エネルギーの平均が-0.4 eV/atom未満のグループを出して」
WITH l12_enthalpy AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM material_entry m
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN composition comp ON comp.entry_id = m.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
        AND per.reference_set = ps.reference_set
    GROUP BY m.entry_id, m.formula, ps.formation_energy_per_atom
)
SELECT cb.element AS b_site,
       COUNT(*) AS n_compounds,
       ROUND(AVG(le.formation_energy_per_atom - le.weighted_ref)::numeric, 4)
           AS avg_enthalpy_vs_element_gs
FROM l12_enthalpy le
JOIN composition cb ON cb.entry_id = le.entry_id AND cb.site_label = 'B-site'
GROUP BY cb.element
HAVING AVG(le.formation_energy_per_atom - le.weighted_ref) < -0.4
ORDER BY avg_enthalpy_vs_element_gs ASC;
