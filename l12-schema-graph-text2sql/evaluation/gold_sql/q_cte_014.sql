-- CTE N: CTE + 割合計算（条件付き集約）
-- 「元素ごとに、その元素を含むL1₂化合物のうち純元素基底状態基準に再基準化した生成エネルギーが負である割合を出して」
WITH enthalpy AS (
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
)
SELECT c.element,
       COUNT(DISTINCT e.entry_id) AS n_total,
       COUNT(DISTINCT e.entry_id) FILTER (
           WHERE e.formation_energy_per_atom - e.weighted_ref < 0
       ) AS n_negative,
       ROUND(
           COUNT(DISTINCT e.entry_id) FILTER (
               WHERE e.formation_energy_per_atom - e.weighted_ref < 0
           )::numeric / COUNT(DISTINCT e.entry_id), 3
       ) AS negative_ratio
FROM enthalpy e
JOIN composition c ON c.entry_id = e.entry_id
GROUP BY c.element
ORDER BY negative_ratio DESC, c.element ASC;
