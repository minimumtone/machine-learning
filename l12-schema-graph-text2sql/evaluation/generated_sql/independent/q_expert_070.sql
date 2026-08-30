SELECT
  m.formula,
  s.prototype,
  s.strukturbericht,
  s.lattice_a,
  s.lattice_b,
  s.lattice_c,
  s.volume_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype IN ('L12', 'B2')
   OR s.strukturbericht IN ('L12', 'B2')
ORDER BY s.prototype, s.strukturbericht, m.formula
LIMIT 10000;
