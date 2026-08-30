WITH bulk_moduli AS (
    SELECT
        c.entry_id,
        MAX(et.bulk_modulus_vrh) AS bulk_modulus_vrh_gpa
    FROM calculation c
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE et.bulk_modulus_vrh >= 150
    GROUP BY c.entry_id
),
l12_entries AS (
    SELECT entry_id
    FROM formation_enthalpy
    WHERE UPPER(REPLACE(REPLACE(COALESCE(strukturbericht, ''), '₂', '2'), '-', '_')) IN ('L1_2', 'L12')
    UNION
    SELECT entry_id
    FROM structure
    WHERE UPPER(REPLACE(REPLACE(COALESCE(strukturbericht, ''), '₂', '2'), '-', '_')) IN ('L1_2', 'L12')
)
SELECT
    f.entry_id,
    f.formula,
    f.reduced_formula,
    b.bulk_modulus_vrh_gpa,
    f.reference_set,
    COALESCE(
        f.enthalpy_vs_element_ground_states,
        f.formation_enthalpy_ev_per_atom + COALESCE(f.weighted_element_delta_e, 0.0)
    ) AS formation_energy_vs_pure_element_ground_states_ev_per_atom
FROM formation_enthalpy f
JOIN l12_entries l
    ON l.entry_id = f.entry_id
JOIN bulk_moduli b
    ON b.entry_id = f.entry_id
ORDER BY formation_energy_vs_pure_element_ground_states_ev_per_atom;
