WITH l12_entries AS (
    SELECT DISTINCT s.entry_id
    FROM structure s
    WHERE s.strukturbericht = 'L12'
       OR s.prototype ILIKE '%L12%'
),
entry_bulk_modulus AS (
    SELECT
        c.entry_id,
        AVG(et.bulk_modulus_vrh) AS bulk_modulus_vrh
    FROM calculation c
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE et.bulk_modulus_vrh IS NOT NULL
    GROUP BY c.entry_id
),
entry_stability AS (
    SELECT
        ps.entry_id,
        MIN(ps.energy_above_hull) AS energy_above_hull
    FROM phase_stability ps
    WHERE ps.energy_above_hull IS NOT NULL
    GROUP BY ps.entry_id
),
classified AS (
    SELECT
        le.entry_id,
        ebm.bulk_modulus_vrh,
        CASE
            WHEN es.energy_above_hull <= 0.001 THEN 'stable'
            WHEN es.energy_above_hull <= 0.05 THEN 'metastable'
            ELSE 'unstable'
        END AS stability_class
    FROM l12_entries le
    JOIN entry_stability es
        ON es.entry_id = le.entry_id
    JOIN entry_bulk_modulus ebm
        ON ebm.entry_id = le.entry_id
)
SELECT
    stability_class,
    COUNT(*) AS n_compounds,
    AVG(bulk_modulus_vrh) AS avg_bulk_modulus_vrh,
    STDDEV_SAMP(bulk_modulus_vrh) AS stddev_bulk_modulus_vrh
FROM classified
GROUP BY stability_class
ORDER BY CASE stability_class
    WHEN 'stable' THEN 1
    WHEN 'metastable' THEN 2
    WHEN 'unstable' THEN 3
END;
