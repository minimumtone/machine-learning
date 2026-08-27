-- VH: 相図上でハル上にある(is_on_hull)エントリを化学系・化学式・ハル距離とともに一覧して
-- Tables: phase_diagram_entry, material_entry (2)
-- Exercises the phase_diagram_entry relation and its generated is_on_hull column.
SELECT pde.chemical_system, m.formula, pde.hull_distance
FROM phase_diagram_entry pde
JOIN material_entry m ON m.entry_id = pde.entry_id
WHERE pde.is_on_hull
ORDER BY pde.chemical_system, m.formula
LIMIT 10000;
