WITH bif3_entries AS (
    SELECT DISTINCT s.entry_id
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON s.prototype = pd.prototype_id
    WHERE s.prototype ILIKE '%BiF3%'
       OR pd.prototype_name ILIKE '%BiF3%'
       OR s.strukturbericht ILIKE '%BiF3%'

    UNION

    SELECT DISTINCT fe.entry_id
    FROM formation_enthalpy fe
    WHERE fe.prototype ILIKE '%BiF3%'
       OR fe.strukturbericht ILIKE '%BiF3%'
),
site_elements AS (
    SELECT
        c.entry_id,
        string_agg(DISTINCT c.element, '-' ORDER BY c.element)
            FILTER (WHERE c.site_label = 'A-site') AS a_site_elements,
        string_agg(DISTINCT c.element, '-' ORDER BY c.element)
            FILTER (WHERE c.site_label = 'B-site') AS b_site_elements
    FROM composition c
    JOIN bif3_entries b
        ON c.entry_id = b.entry_id
    GROUP BY c.entry_id
    HAVING COUNT(*) FILTER (WHERE c.site_label = 'A-site') > 0
       AND COUNT(*) FILTER (WHERE c.site_label = 'B-site') > 0
),
entry_stability AS (
    SELECT
        ps.entry_id,
        MIN(ps.energy_above_hull) AS energy_above_hull
    FROM phase_stability ps
    JOIN bif3_entries b
        ON ps.entry_id = b.entry_id
    WHERE ps.energy_above_hull IS NOT NULL
    GROUP BY ps.entry_id
)
SELECT
    se.a_site_elements AS a_site_elements,
    se.b_site_elements AS b_site_elements,
    COUNT(*) AS compound_count,
    COUNT(*) FILTER (WHERE es.energy_above_hull <= 0.001) AS stable_count,
    COUNT(*) FILTER (WHERE es.energy_above_hull > 0.001 AND es.energy_above_hull <= 0.05) AS metastable_count,
    COUNT(*) FILTER (WHERE es.energy_above_hull > 0.05) AS unstable_count,
    ROUND(AVG(es.energy_above_hull)::numeric, 6) AS avg_energy_above_hull_ev_per_atom,
    ROUND(
        COUNT(*) FILTER (WHERE es.energy_above_hull <= 0.001)::numeric
        / NULLIF(COUNT(*), 0),
        4
    ) AS stable_fraction
FROM site_elements se
JOIN entry_stability es
    ON se.entry_id = es.entry_id
GROUP BY
    se.a_site_elements,
    se.b_site_elements
ORDER BY
    stable_count DESC,
    stable_fraction DESC,
    compound_count DESC,
    se.a_site_elements,
    se.b_site_elements;
