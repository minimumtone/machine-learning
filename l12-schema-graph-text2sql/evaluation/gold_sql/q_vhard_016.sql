-- CTE B: CTE + フィルタ
-- 「安定なL1₂化合物の生成エンタルピーを計算して、-0.3 eV/atom未満のものを出して」
WITH stable_l12 AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom, s.lattice_a
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND ps.energy_above_hull <= 0.001
),
enthalpy AS (
    SELECT sl.entry_id, sl.formula, sl.formation_energy_per_atom, sl.lattice_a,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM stable_l12 sl
    JOIN composition comp ON comp.entry_id = sl.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    GROUP BY sl.entry_id, sl.formula, sl.formation_energy_per_atom, sl.lattice_a
)
SELECT formula, lattice_a, formation_energy_per_atom,
       weighted_ref,
       formation_energy_per_atom - weighted_ref AS delta_h_f
FROM enthalpy
WHERE formation_energy_per_atom - weighted_ref < -0.3
ORDER BY formation_energy_per_atom - weighted_ref ASC
LIMIT 20;
