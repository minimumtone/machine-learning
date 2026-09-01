WITH elastic_by_entry AS (
    SELECT
        c.entry_id,
        MAX(et.bulk_modulus_vrh) AS bulk_modulus_vrh
    FROM calculation c
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE et.bulk_modulus_vrh IS NOT NULL
    GROUP BY c.entry_id
),
candidates AS (
    SELECT
        me.entry_id,
        me.formula,
        me.reduced_formula,
        me.chemical_system,
        ps.energy_above_hull,
        ps.formation_energy_per_atom,
        s.lattice_a,
        ebe.bulk_modulus_vrh,
        (
            (1 - LEAST(ps.energy_above_hull, 0.05) / 0.05) * 0.30
          + (1 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.25
          + (LEAST(ebe.bulk_modulus_vrh, 300) / 300) * 0.25
          + CASE
                WHEN ps.formation_energy_per_atom < -0.3 THEN 0.20
                ELSE 0.10
            END
        ) AS design_score
    FROM material_entry me
    JOIN phase_stability ps
        ON ps.entry_id = me.entry_id
    JOIN structure s
        ON s.entry_id = me.entry_id
    LEFT JOIN prototype_definition pd
        ON pd.prototype_id = s.prototype
    JOIN elastic_by_entry ebe
        ON ebe.entry_id = me.entry_id
    WHERE ps.is_stable = TRUE
      AND me.chemical_system <> 'Al-Ni'
      AND s.lattice_a IS NOT NULL
      AND ps.energy_above_hull IS NOT NULL
      AND ps.formation_energy_per_atom IS NOT NULL
      AND (
            UPPER(REPLACE(COALESCE(s.strukturbericht, pd.strukturbericht), '_', '')) IN ('L12', 'L1₂')
            OR COALESCE(s.prototype, pd.prototype_name) ILIKE '%L12%'
            OR COALESCE(s.prototype, pd.prototype_name) ILIKE '%L1_2%'
          )
)
SELECT
    RANK() OVER (ORDER BY design_score DESC) AS rank,
    entry_id,
    formula,
    reduced_formula,
    chemical_system,
    energy_above_hull,
    lattice_a,
    bulk_modulus_vrh AS bulk_modulus,
    formation_energy_per_atom,
    design_score
FROM candidates
ORDER BY design_score DESC, energy_above_hull ASC;
