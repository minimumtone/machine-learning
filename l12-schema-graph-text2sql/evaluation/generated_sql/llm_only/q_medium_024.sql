SELECT DISTINCT
  fh.entry_id,
  fh.formula,
  fh.reduced_formula,
  fh.formation_enthalpy_ev_per_atom AS formation_energy_ev_per_atom,
  COALESCE(fh.prototype, s.prototype) AS prototype,
  COALESCE(fh.strukturbericht, s.strukturbericht) AS strukturbericht
FROM formation_enthalpy fh
LEFT JOIN structure s
  ON s.entry_id = fh.entry_id
WHERE EXISTS (
  SELECT 1
  FROM composition c
  WHERE c.entry_id = fh.entry_id
    AND c.element = 'Pd'
)
AND (
  regexp_replace(lower(COALESCE(fh.strukturbericht, '')), '[^[:alnum:]]', '', 'g') = 'l12'
  OR regexp_replace(lower(COALESCE(s.strukturbericht, '')), '[^[:alnum:]]', '', 'g') = 'l12'
)
ORDER BY fh.formation_enthalpy_ev_per_atom ASC NULLS LAST;
