WITH candidates AS (
  SELECT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    ps.energy_above_hull,
    s.prototype,
    s.strukturbericht,
    s.lattice_a,
    et.bulk_modulus_vrh AS bulk_modulus_gpa,
    (
      (1.0 - ps.energy_above_hull / 0.05) * 0.35
      + (1.0 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.35
      + (LEAST(et.bulk_modulus_vrh, 300.0) / 300.0) * 0.30
    ) AS score
  FROM material_entry me
  JOIN phase_stability ps
    ON ps.entry_id = me.entry_id
  JOIN structure s
    ON s.entry_id = me.entry_id
  JOIN calculation c
    ON c.entry_id = me.entry_id
  JOIN elastic_tensor et
    ON et.calculation_id = c.calculation_id
  WHERE ps.energy_above_hull <= 0.05
    AND s.lattice_a IS NOT NULL
    AND et.bulk_modulus_vrh IS NOT NULL
    AND (
      regexp_replace(upper(translate(COALESCE(s.strukturbericht, ''), '₁₂₃', '123')), '[^A-Z0-9]', '', 'g') = 'L12'
      OR regexp_replace(upper(translate(COALESCE(s.prototype, ''), '₁₂₃', '123')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
    )
)
SELECT
  entry_id,
  formula,
  reduced_formula,
  energy_above_hull,
  lattice_a,
  bulk_modulus_gpa,
  score,
  prototype,
  strukturbericht
FROM candidates
ORDER BY score DESC, energy_above_hull ASC
LIMIT 20;
