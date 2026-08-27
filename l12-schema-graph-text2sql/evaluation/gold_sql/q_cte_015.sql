-- CTE O: 3段CTE（フィルタ → 計算 → 集約）
-- 「純物質の原子あたり体積が15 Å³以上のA-site元素を持つ安定L1₂化合物の純元素基底状態基準に再基準化した生成エネルギーの平均をA-site元素ごとに出して」
WITH large_a_site AS (
    SELECT c.entry_id, c.element AS a_site
    FROM composition c
    JOIN pure_element_reference per ON per.element_symbol = c.element
    WHERE c.site_label = 'A-site'
      AND per.reference_set = 'L12-FIXTURE-PBE-v1'
      AND per.volume_per_atom >= 15
),
enthalpy AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN composition comp ON comp.entry_id = m.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
        AND per.reference_set = ps.reference_set
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND ps.energy_above_hull <= 0.001
    GROUP BY m.entry_id, m.formula, ps.formation_energy_per_atom
)
SELECT la.a_site,
       COUNT(*) AS n_compounds,
       ROUND(AVG(e.formation_energy_per_atom - e.weighted_ref)::numeric, 4)
           AS avg_enthalpy_vs_element_gs
FROM enthalpy e
JOIN large_a_site la ON la.entry_id = e.entry_id
GROUP BY la.a_site
ORDER BY avg_enthalpy_vs_element_gs ASC;
