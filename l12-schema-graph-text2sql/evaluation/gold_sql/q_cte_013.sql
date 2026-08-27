-- CTE M: CTE + 補正量ランキング
-- 「形成エネルギーと純物質補正後の生成エンタルピーの差が大きいL1₂化合物上位10件を出して」
WITH enthalpy AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN composition comp ON comp.entry_id = m.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    GROUP BY m.entry_id, m.formula, ps.formation_energy_per_atom
)
SELECT formula,
       ROUND(formation_energy_per_atom::numeric, 4) AS e_form,
       ROUND(weighted_ref::numeric, 4) AS correction,
       ROUND(ABS(weighted_ref)::numeric, 4) AS abs_correction
FROM enthalpy
ORDER BY ABS(weighted_ref) DESC
LIMIT 10;
