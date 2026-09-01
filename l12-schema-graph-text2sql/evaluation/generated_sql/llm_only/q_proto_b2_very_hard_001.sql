WITH b2_entries AS (
    SELECT DISTINCT
        s.entry_id
    FROM structure s
    WHERE s.strukturbericht = 'B2'
),
site_elements AS (
    SELECT
        c.entry_id,
        MAX(c.element) FILTER (WHERE c.site_label = 'A-site') AS a_site_element,
        MAX(c.element) FILTER (WHERE c.site_label = 'B-site') AS b_site_element
    FROM composition c
    WHERE c.site_label IN ('A-site', 'B-site')
    GROUP BY c.entry_id
),
stability AS (
    SELECT
        ps.entry_id,
        ps.energy_above_hull,
        CASE
            WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
            WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
            ELSE 'unstable'
        END AS stability_class
    FROM phase_stability ps
)
SELECT
    se.a_site_element,
    se.b_site_element,
    COUNT(*) AS total_count,
    COUNT(*) FILTER (WHERE st.stability_class = 'stable') AS stable_count,
    COUNT(*) FILTER (WHERE st.stability_class = 'metastable') AS metastable_count,
    COUNT(*) FILTER (WHERE st.stability_class = 'unstable') AS unstable_count,
    AVG(st.energy_above_hull) AS avg_energy_above_hull,
    MIN(st.energy_above_hull) AS min_energy_above_hull,
    MAX(st.energy_above_hull) AS max_energy_above_hull
FROM b2_entries b2
JOIN site_elements se
    ON b2.entry_id = se.entry_id
JOIN stability st
    ON b2.entry_id = st.entry_id
WHERE se.a_site_element IS NOT NULL
  AND se.b_site_element IS NOT NULL
GROUP BY
    se.a_site_element,
    se.b_site_element
ORDER BY
    stable_count DESC,
    metastable_count DESC,
    total_count DESC,
    se.a_site_element,
    se.b_site_element;
