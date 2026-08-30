WITH nias_entries AS (
    SELECT DISTINCT s.entry_id
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON s.prototype = pd.prototype_id
    WHERE LOWER(COALESCE(pd.prototype_name, s.prototype, '')) LIKE '%nias%'
       OR LOWER(COALESCE(s.strukturbericht, pd.strukturbericht, '')) IN ('b8_1', 'b81')
),
site_elements AS (
    SELECT
        c.entry_id,
        STRING_AGG(DISTINCT c.element, '-' ORDER BY c.element) FILTER (WHERE c.site_label = 'A-site') AS a_site_elements,
        STRING_AGG(DISTINCT c.element, '-' ORDER BY c.element) FILTER (WHERE c.site_label = 'B-site') AS b_site_elements
    FROM composition c
    WHERE c.site_label IN ('A-site', 'B-site')
    GROUP BY c.entry_id
),
entry_stability AS (
    SELECT
        ps.entry_id,
        MIN(ps.energy_above_hull) AS energy_above_hull
    FROM phase_stability ps
    GROUP BY ps.entry_id
),
classified AS (
    SELECT
        ne.entry_id,
        se.a_site_elements,
        se.b_site_elements,
        es.energy_above_hull,
        CASE
            WHEN es.energy_above_hull <= 0.001 THEN 'stable'
            WHEN es.energy_above_hull <= 0.05 THEN 'metastable'
            ELSE 'unstable'
        END AS stability_class
    FROM nias_entries ne
    JOIN site_elements se
        ON ne.entry_id = se.entry_id
    JOIN entry_stability es
        ON ne.entry_id = es.entry_id
    WHERE se.a_site_elements IS NOT NULL
      AND se.b_site_elements IS NOT NULL
)
SELECT
    a_site_elements,
    b_site_elements,
    COUNT(*) AS total_count,
    COUNT(*) FILTER (WHERE stability_class = 'stable') AS stable_count,
    COUNT(*) FILTER (WHERE stability_class = 'metastable') AS metastable_count,
    COUNT(*) FILTER (WHERE stability_class = 'unstable') AS unstable_count,
    ROUND(
        COUNT(*) FILTER (WHERE stability_class = 'stable')::numeric / NULLIF(COUNT(*), 0),
        4
    ) AS stable_fraction,
    AVG(energy_above_hull) AS avg_energy_above_hull
FROM classified
GROUP BY
    a_site_elements,
    b_site_elements
ORDER BY
    stable_count DESC,
    metastable_count DESC,
    avg_energy_above_hull ASC;
