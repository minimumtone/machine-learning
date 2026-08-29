SELECT DISTINCT
    me.entry_id,
    me.formula,
    s.strukturbericht,
    s.lattice_a,
    ABS(s.lattice_a - 3.57) AS lattice_a_diff_from_ni3al,
    et.bulk_modulus_vrh
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE (
        s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
        OR s.prototype IN ('L1_2', 'L12', 'L1₂')
      )
  AND EXISTS (
        SELECT 1
        FROM composition AS cni
        WHERE cni.entry_id = me.entry_id
          AND cni.element = 'Ni'
      )
  AND EXISTS (
        SELECT 1
        FROM composition AS cal
        WHERE cal.entry_id = me.entry_id
          AND cal.element = 'Al'
      )
  AND ABS(s.lattice_a - 3.57) <= 0.1
  AND et.bulk_modulus_vrh >= 100
ORDER BY lattice_a_diff_from_ni3al, et.bulk_modulus_vrh DESC;
