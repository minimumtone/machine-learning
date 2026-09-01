WITH candidates AS (
    SELECT
        me.entry_id,
        me.formula,
        me.reduced_formula,
        s.prototype,
        s.strukturbericht,
        s.lattice_a,
        et.bulk_modulus_vrh AS bulk_modulus_gpa,
        et.shear_modulus_vrh,
        et.youngs_modulus,
        et.poisson_ratio,
        c.calculation_id,
        c.method,
        c.functional,
        ROW_NUMBER() OVER (
            PARTITION BY me.entry_id
            ORDER BY et.bulk_modulus_vrh DESC NULLS LAST, ABS(s.lattice_a - 3.55)
        ) AS rn
    FROM material_entry me
    JOIN structure s
        ON s.entry_id = me.entry_id
    LEFT JOIN prototype_definition pd
        ON pd.prototype_id = s.prototype
    JOIN calculation c
        ON c.entry_id = me.entry_id
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE ABS(s.lattice_a - 3.55) <= 0.1
      AND et.bulk_modulus_vrh IS NOT NULL
      AND (
          regexp_replace(replace(upper(COALESCE(s.strukturbericht, '')), '₂', '2'), '[^A-Z0-9]', '', 'g') = 'L12'
          OR regexp_replace(replace(upper(COALESCE(pd.strukturbericht, '')), '₂', '2'), '[^A-Z0-9]', '', 'g') = 'L12'
          OR upper(COALESCE(s.prototype, '')) LIKE '%L12%'
          OR upper(COALESCE(pd.prototype_name, '')) LIKE '%L12%'
      )
)
SELECT
    entry_id,
    formula,
    reduced_formula,
    prototype,
    strukturbericht,
    lattice_a,
    bulk_modulus_gpa,
    shear_modulus_vrh,
    youngs_modulus,
    poisson_ratio,
    calculation_id,
    method,
    functional
FROM candidates
WHERE rn = 1
ORDER BY bulk_modulus_gpa DESC NULLS LAST, ABS(lattice_a - 3.55)
LIMIT 20;
