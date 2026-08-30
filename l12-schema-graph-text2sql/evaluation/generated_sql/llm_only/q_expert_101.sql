SELECT DISTINCT
    fh.formula,
    fh.lattice_a AS lattice_constant_angstrom,
    et.bulk_modulus_vrh AS bulk_modulus_gpa,
    fh.formation_enthalpy_ev_per_atom AS formation_energy_ev_per_atom
FROM formation_enthalpy AS fh
JOIN calculation AS c
    ON c.entry_id = fh.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE fh.is_stable = TRUE
  AND lower(regexp_replace(replace(COALESCE(fh.strukturbericht, fh.prototype, ''), '₂', '2'), '[^a-z0-9]', '', 'g')) = 'l12'
  AND et.bulk_modulus_vrh >= 180
  AND fh.lattice_a <= 3.9
ORDER BY fh.formula;
