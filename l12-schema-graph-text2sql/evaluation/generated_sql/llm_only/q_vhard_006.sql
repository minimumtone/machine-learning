WITH l12_candidates AS (
    SELECT
        me.entry_id,
        me.formula,
        me.reduced_formula,
        MIN(ps.energy_above_hull) AS energy_above_hull,
        MIN(ABS(s.lattice_a - 3.57)) AS lattice_a_diff,
        (ARRAY_AGG(s.lattice_a ORDER BY ABS(s.lattice_a - 3.57)))[1] AS lattice_a,
        MAX(et.bulk_modulus_vrh) AS bulk_modulus_vrh
    FROM material_entry me
    JOIN phase_stability ps
        ON ps.entry_id = me.entry_id
    JOIN structure s
        ON s.entry_id = me.entry_id
    JOIN calculation c
        ON c.entry_id = me.entry_id
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE ps.is_stable = TRUE
      AND et.bulk_modulus_vrh IS NOT NULL
      AND s.lattice_a IS NOT NULL
      AND (
          s.strukturbericht = 'L1_2'
          OR s.prototype = 'L1_2'
          OR s.strukturbericht ILIKE 'L1₂'
          OR s.prototype ILIKE 'L1₂'
          OR s.strukturbericht ILIKE 'L12'
          OR s.prototype ILIKE 'L12'
      )
    GROUP BY
        me.entry_id,
        me.formula,
        me.reduced_formula
)
SELECT
    entry_id,
    formula,
    reduced_formula,
    energy_above_hull,
    lattice_a,
    bulk_modulus_vrh,
    ((1 - energy_above_hull / 0.05) * 0.35
     + (1 - ABS(lattice_a - 3.57) / 0.3) * 0.35
     + (bulk_modulus_vrh / 300.0) * 0.30) AS total_score
FROM l12_candidates
ORDER BY total_score DESC;
