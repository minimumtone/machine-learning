SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  pde.hull_distance AS convex_hull_distance
FROM material_entry AS me
JOIN phase_diagram_entry AS pde
  ON pde.entry_id = me.entry_id
WHERE me.chemical_system = 'Al-Ni'
ORDER BY pde.hull_distance, me.reduced_formula, me.entry_id;
