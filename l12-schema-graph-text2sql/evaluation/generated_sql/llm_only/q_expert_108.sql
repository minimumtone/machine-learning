SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    dt.defect_name AS defect_type,
    md.formation_energy AS defect_formation_energy,
    et.bulk_modulus_vrh
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
JOIN material_defect AS md
    ON md.entry_id = me.entry_id
JOIN defect_type AS dt
    ON dt.defect_type_id = md.defect_type_id
WHERE ps.is_stable = TRUE
  AND et.bulk_modulus_vrh >= 150
  AND (
      UPPER(REPLACE(REPLACE(s.strukturbericht, '_', ''), '₂', '2')) = 'L12'
      OR UPPER(REPLACE(REPLACE(s.prototype, '_', ''), '₂', '2')) = 'L12'
  )
ORDER BY me.entry_id, dt.defect_name;
