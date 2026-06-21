-- CTE E: CTE + カラム間比較
-- 「L1₂化合物の形成エネルギーが加重純物質基準より0.1 eV/atom以上低い化合物を出して」
WITH compound_ref AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           SUM(comp.atomic_fraction * per.energy_per_atom) AS weighted_ref
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN composition comp ON comp.entry_id = m.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    GROUP BY m.entry_id, m.formula, ps.formation_energy_per_atom
)
SELECT formula,
       ROUND(formation_energy_per_atom::numeric, 4) AS e_compound,
       ROUND(weighted_ref::numeric, 4) AS e_ref,
       ROUND((formation_energy_per_atom - weighted_ref)::numeric, 4) AS delta
FROM compound_ref
WHERE formation_energy_per_atom - weighted_ref < -0.1
ORDER BY formation_energy_per_atom - weighted_ref ASC
LIMIT 20;
