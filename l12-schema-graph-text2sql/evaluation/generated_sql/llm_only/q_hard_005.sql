SELECT DISTINCT
    fh.entry_id,
    me.formula,
    fh.reduced_formula,
    fh.formation_enthalpy_ev_per_atom,
    fh.energy_above_hull,
    fh.prototype,
    fh.strukturbericht
FROM formation_enthalpy AS fh
JOIN material_entry AS me
  ON me.entry_id = fh.entry_id
WHERE fh.is_stable = TRUE
  AND (
      regexp_replace(upper(coalesce(fh.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(coalesce(fh.prototype, '')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
  )
  AND EXISTS (
      SELECT 1
      FROM composition AS c
      WHERE c.entry_id = fh.entry_id
        AND c.element IN ('Ni', 'Co')
  )
ORDER BY fh.formation_enthalpy_ev_per_atom ASC;
