WITH ni3al AS (
    SELECT AVG(s.lattice_a) AS lattice_a
    FROM material_entry me
    JOIN structure s ON s.entry_id = me.entry_id
    WHERE me.chemical_system = 'Al-Ni'
      AND s.lattice_a IS NOT NULL
      AND EXISTS (
          SELECT 1
          FROM composition c
          WHERE c.entry_id = me.entry_id
            AND c.element = 'Ni'
            AND ABS(c.atomic_fraction - 0.75) < 1e-6
      )
      AND EXISTS (
          SELECT 1
          FROM composition c
          WHERE c.entry_id = me.entry_id
            AND c.element = 'Al'
            AND ABS(c.atomic_fraction - 0.25) < 1e-6
      )
)
SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.lattice_a,
    et.bulk_modulus_vrh
FROM material_entry me
JOIN structure s ON s.entry_id = me.entry_id
JOIN calculation calc ON calc.entry_id = me.entry_id
JOIN elastic_tensor et ON et.calculation_id = calc.calculation_id
CROSS JOIN ni3al
WHERE ni3al.lattice_a IS NOT NULL
  AND ABS(s.lattice_a - ni3al.lattice_a) <= 0.05
  AND et.bulk_modulus_vrh >= 150
  AND EXISTS (
      SELECT 1
      FROM phase_stability ps
      WHERE ps.entry_id = me.entry_id
        AND ps.is_stable = TRUE
  )
ORDER BY ABS(s.lattice_a - ni3al.lattice_a), et.bulk_modulus_vrh DESC;
