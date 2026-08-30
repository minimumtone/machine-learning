SELECT
  me.entry_id,
  me.formula,
  dos.total_dos_at_fermi
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN density_of_states AS dos
  ON dos.calculation_id = c.calculation_id
WHERE dos.total_dos_at_fermi IS NOT NULL
  AND (
    regexp_replace(upper(coalesce(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(coalesce(pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(coalesce(s.prototype, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(coalesce(pd.prototype_name, '')), '[^A-Z0-9]', '', 'g') = 'L12'
  )
ORDER BY dos.total_dos_at_fermi DESC
LIMIT 1;
