WITH candidates AS (
    SELECT
        me.entry_id,
        me.formula,
        me.reduced_formula,
        ps.energy_above_hull,
        s.lattice_a AS a_angstrom,
        et.bulk_modulus_vrh AS bulk_modulus_gpa,
        (
            (1 - ps.energy_above_hull / 0.05) * 0.4
          + (1 - ABS(s.lattice_a - 3.57) / 0.3) * 0.3
          + (et.bulk_modulus_vrh / 300.0) * 0.3
        ) AS weighted_score
    FROM material_entry me
    JOIN phase_stability ps
        ON ps.entry_id = me.entry_id
    JOIN structure s
        ON s.entry_id = me.entry_id
    LEFT JOIN prototype_definition pd
        ON pd.prototype_id = s.prototype
    JOIN calculation c
        ON c.entry_id = me.entry_id
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE ps.energy_above_hull <= 0.05
      AND (
          s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
          OR pd.strukturbericht IN ('L1_2', 'L12', 'L1₂')
      )
      AND s.lattice_a IS NOT NULL
      AND et.bulk_modulus_vrh IS NOT NULL
)
SELECT
    RANK() OVER (ORDER BY weighted_score DESC) AS rank,
    entry_id,
    formula,
    reduced_formula,
    energy_above_hull,
    a_angstrom,
    bulk_modulus_gpa,
    weighted_score
FROM candidates
ORDER BY weighted_score DESC;
