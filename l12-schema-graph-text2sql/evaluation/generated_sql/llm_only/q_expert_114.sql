SELECT
  pde.entry_id,
  pde.chemical_system,
  me.formula,
  pde.hull_distance
FROM phase_diagram_entry AS pde
JOIN material_entry AS me
  ON me.entry_id = pde.entry_id
WHERE pde.is_on_hull = TRUE
ORDER BY pde.chemical_system, me.formula, pde.entry_id;
