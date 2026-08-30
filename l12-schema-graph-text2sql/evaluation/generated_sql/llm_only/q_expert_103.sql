WITH l12_entries AS (
    SELECT DISTINCT
        me.entry_id,
        ca.element AS a_site_element,
        cb.element AS b_site_element
    FROM material_entry me
    JOIN structure s
        ON s.entry_id = me.entry_id
    JOIN composition ca
        ON ca.entry_id = me.entry_id
       AND ca.site_label = 'A-site'
    JOIN composition cb
        ON cb.entry_id = me.entry_id
       AND cb.site_label = 'B-site'
    WHERE s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
),
entry_bulk AS (
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
        BOOL_OR(ps.is_stable = TRUE) AS is_stable
    FROM phase_stability ps
    GROUP BY ps.entry_id
)
SELECT
    le.a_site_element,
    le.b_site_element,
    AVG(eb.bulk_modulus_vrh) AS avg_bulk_modulus_vrh,
    COUNT(DISTINCT le.entry_id) FILTER (WHERE es.is_stable = TRUE) AS stable_phase_count
FROM l12_entries le
LEFT JOIN entry_bulk eb
    ON eb.entry_id = le.entry_id
LEFT JOIN entry_stability es
    ON es.entry_id = le.entry_id
GROUP BY
    le.a_site_element,
    le.b_site_element
ORDER BY
    le.a_site_element,
    le.b_site_element;
