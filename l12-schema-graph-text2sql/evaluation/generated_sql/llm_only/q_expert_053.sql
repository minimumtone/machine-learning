SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system
FROM material_entry AS me
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN density_of_states AS dos
    ON dos.calculation_id = c.calculation_id
WHERE dos.spin_polarized = TRUE
  AND me.number_of_elements > 1
ORDER BY me.chemical_system, me.reduced_formula;
