WITH nacl_entries AS (
    SELECT DISTINCT s.entry_id
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON s.prototype = pd.prototype_id
    WHERE COALESCE(s.strukturbericht, pd.strukturbericht) = 'B1'
       OR s.prototype ILIKE '%NaCl%'
       OR pd.prototype_name ILIKE '%NaCl%'
),
site_pairs AS (
    SELECT
        c.entry_id,
        string_agg(DISTINCT c.element, '-' ORDER BY c.element) FILTER (WHERE c.site_label = 'A-site') AS a_site_elements,
        string_agg(DISTINCT c.element, '-' ORDER BY c.element) FILTER (WHERE c.site_label = 'B-site') AS b_site_elements
    FROM composition c
    JOIN nacl_entries ne
        ON c.entry_id = ne.entry_id
    GROUP BY c.entry_id
    HAVING COUNT(*) FILTER (WHERE c.site_label = 'A-site') > 0
       AND COUNT(*) FILTER (WHERE c.site_label = 'B-site') > 0
),
stability AS (
    SELECT
        entry_id,
        MIN(energy_above_hull) AS energy_above_hull
    FROM phase_stability
    WHERE energy_above_hull IS NOT NULL
    GROUP BY entry_id
)
SELECT
    sp.a_site_elements,
    sp.b_site_elements,
    COUNT(*) AS total_compounds,
    COUNT(*) FILTER (WHERE st.energy_above_hull <= 0.001) AS stable_count,
    COUNT(*) FILTER (WHERE st.energy_above_hull > 0.001 AND st.energy_above_hull <= 0.05) AS metastable_count,
    COUNT(*) FILTER (WHERE st.energy_above_hull > 0.05) AS unstable_count,
    COUNT(*) FILTER (WHERE st.energy_above_hull IS NULL) AS unknown_stability_count,
    ROUND(AVG(st.energy_above_hull)::numeric, 6) AS avg_energy_above_hull_ev_per_atom,
    ROUND(
        (COUNT(*) FILTER (WHERE st.energy_above_hull <= 0.001))::numeric
        / NULLIF(COUNT(*) FILTER (WHERE st.energy_above_hull IS NOT NULL), 0),
        3
    ) AS stable_fraction
FROM site_pairs sp
LEFT JOIN stability st
    ON sp.entry_id = st.entry_id
GROUP BY
    sp.a_site_elements,
    sp.b_site_elements
ORDER BY
    stable_count DESC,
    stable_fraction DESC NULLS LAST,
    total_compounds DESC,
    sp.a_site_elements,
    sp.b_site_elements;
