WITH entry_bulk AS (
    SELECT
        c.entry_id,
        MAX(et.bulk_modulus_vrh) AS bulk_modulus_vrh
    FROM calculation c
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE et.bulk_modulus_vrh IS NOT NULL
    GROUP BY c.entry_id
),
stable_entries AS (
    SELECT
        ps.entry_id,
        MIN(ps.energy_above_hull) AS energy_above_hull
    FROM phase_stability ps
    WHERE ps.is_stable = TRUE
    GROUP BY ps.entry_id
),
l12_entries AS (
    SELECT DISTINCT
        s.entry_id
    FROM structure s
    WHERE s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
)
SELECT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    se.energy_above_hull,
    eb.bulk_modulus_vrh
FROM material_entry me
JOIN stable_entries se
    ON se.entry_id = me.entry_id
JOIN entry_bulk eb
    ON eb.entry_id = me.entry_id
JOIN l12_entries l12
    ON l12.entry_id = me.entry_id
WHERE EXISTS (
    SELECT 1
    FROM composition comp
    WHERE comp.entry_id = me.entry_id
      AND comp.element IN ('Ni', 'Co')
)
ORDER BY eb.bulk_modulus_vrh DESC
LIMIT 10;
