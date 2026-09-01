SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system
FROM material_entry AS me
WHERE me.number_of_elements > 1
  AND EXISTS (
      SELECT 1
      FROM experimental_measurement AS em
      JOIN measured_property AS mp
        ON mp.measurement_id = em.measurement_id
      WHERE em.entry_id = me.entry_id
  )
  AND EXISTS (
      SELECT 1
      FROM calculation AS c
      JOIN calculated_property AS cp
        ON cp.calculation_id = c.calculation_id
      WHERE c.entry_id = me.entry_id
  )
ORDER BY me.reduced_formula, me.entry_id;
