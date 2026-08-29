SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    et.bulk_modulus_vrh AS bulk_modulus_gpa,
    sm.method_name AS synthesis_method,
    ms.temperature_k AS synthesis_temperature_k
FROM material_entry AS me
JOIN structure AS st
    ON st.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = st.prototype
JOIN material_synthesis AS ms
    ON ms.entry_id = me.entry_id
   AND ms.success = TRUE
JOIN synthesis_method AS sm
    ON sm.synthesis_id = ms.synthesis_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE et.bulk_modulus_vrh >= 150
  AND EXISTS (
      SELECT 1
      FROM phase_stability AS ps
      WHERE ps.entry_id = me.entry_id
        AND ps.is_stable = TRUE
  )
  AND (
      regexp_replace(upper(translate(COALESCE(st.strukturbericht, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(translate(COALESCE(st.prototype, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(translate(COALESCE(pd.strukturbericht, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(translate(COALESCE(pd.prototype_name, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
  )
ORDER BY
    me.formula,
    sm.method_name,
    ms.temperature_k;
