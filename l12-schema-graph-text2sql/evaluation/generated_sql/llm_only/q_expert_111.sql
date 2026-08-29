SELECT DISTINCT
    me.entry_id,
    me.formula,
    s.strukturbericht,
    s.lattice_a,
    et.bulk_modulus_vrh
FROM material_entry AS me
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE ps.is_stable = TRUE
  AND UPPER(REPLACE(REPLACE(s.strukturbericht, '₂', '2'), '-', '_')) IN ('L1_2', 'L12')
  AND et.bulk_modulus_vrh >= 200
  AND s.lattice_a BETWEEN 3.5 AND 4.0
ORDER BY et.bulk_modulus_vrh DESC;
