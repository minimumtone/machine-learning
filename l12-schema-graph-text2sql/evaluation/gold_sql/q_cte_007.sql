-- CTE G: CTE + 純物質参照テーブル集約
-- 「多形数が多い純物質元素上位5つについて、その元素を含むL1₂化合物数と平均形成エネルギーを出して」
WITH top_polymorph AS (
    SELECT element_symbol, n_polymorphs
    FROM pure_element_reference
    ORDER BY n_polymorphs DESC, element_symbol ASC
    LIMIT 5
)
SELECT tp.element_symbol,
       tp.n_polymorphs,
       COUNT(DISTINCT m.entry_id) AS n_l12_compounds,
       ROUND(AVG(ps.formation_energy_per_atom)::numeric, 4) AS avg_e_form
FROM top_polymorph tp
LEFT JOIN composition c ON c.element = tp.element_symbol
LEFT JOIN material_entry m ON m.entry_id = c.entry_id
LEFT JOIN structure s ON s.entry_id = m.entry_id
    AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
LEFT JOIN phase_stability ps ON ps.entry_id = m.entry_id
GROUP BY tp.element_symbol, tp.n_polymorphs
ORDER BY tp.n_polymorphs DESC, tp.element_symbol ASC;
