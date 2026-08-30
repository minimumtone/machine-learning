WITH l12_entries AS (
    SELECT DISTINCT entry_id
    FROM (
        SELECT
            s.entry_id,
            concat_ws(' ', s.prototype, s.strukturbericht, pd.prototype_name, pd.strukturbericht) AS prototype_text
        FROM structure s
        LEFT JOIN prototype_definition pd
            ON pd.prototype_id = s.prototype

        UNION ALL

        SELECT
            fe.entry_id,
            concat_ws(' ', fe.prototype, fe.strukturbericht) AS prototype_text
        FROM formation_enthalpy fe
    ) p
    WHERE prototype_text ILIKE ANY (ARRAY['%L1_2%', '%L12%', '%L1₂%'])
),
site_pairs AS (
    SELECT
        le.entry_id,
        string_agg(DISTINCT c.element, '+' ORDER BY c.element)
            FILTER (WHERE c.site_label = 'A-site') AS a_site_elements,
        string_agg(DISTINCT e.category, '+' ORDER BY e.category)
            FILTER (WHERE c.site_label = 'A-site') AS a_site_categories,
        string_agg(DISTINCT c.element, '+' ORDER BY c.element)
            FILTER (WHERE c.site_label = 'B-site') AS b_site_elements,
        string_agg(DISTINCT e.category, '+' ORDER BY e.category)
            FILTER (WHERE c.site_label = 'B-site') AS b_site_categories
    FROM l12_entries le
    JOIN composition c
        ON c.entry_id = le.entry_id
    LEFT JOIN element e
        ON e.symbol = c.element
    WHERE c.site_label IN ('A-site', 'B-site')
    GROUP BY le.entry_id
    HAVING
        COUNT(DISTINCT c.element) FILTER (WHERE c.site_label = 'A-site') > 0
        AND COUNT(DISTINCT c.element) FILTER (WHERE c.site_label = 'B-site') > 0
),
stability AS (
    SELECT
        entry_id,
        MIN(energy_above_hull) AS energy_above_hull
    FROM phase_stability
    GROUP BY entry_id
)
SELECT
    sp.a_site_elements,
    sp.a_site_categories,
    sp.b_site_elements,
    sp.b_site_categories,
    COUNT(*) AS n_l12_entries,
    COUNT(st.energy_above_hull) AS n_with_stability_data,
    COUNT(*) FILTER (WHERE st.energy_above_hull <= 0.001) AS stable_count,
    COUNT(*) FILTER (
        WHERE st.energy_above_hull > 0.001
          AND st.energy_above_hull <= 0.05
    ) AS metastable_count,
    COUNT(*) FILTER (WHERE st.energy_above_hull > 0.05) AS unstable_count,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE st.energy_above_hull <= 0.001)
        / NULLIF(COUNT(st.energy_above_hull), 0),
        1
    ) AS stable_pct,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE st.energy_above_hull <= 0.05)
        / NULLIF(COUNT(st.energy_above_hull), 0),
        1
    ) AS stable_or_metastable_pct,
    ROUND(AVG(st.energy_above_hull)::numeric, 4) AS avg_energy_above_hull_ev_per_atom,
    ROUND(MIN(st.energy_above_hull)::numeric, 4) AS best_energy_above_hull_ev_per_atom,
    CASE
        WHEN COUNT(st.energy_above_hull) = 0 THEN 'no_stability_data'
        WHEN COUNT(*) FILTER (WHERE st.energy_above_hull <= 0.001)::numeric
             / COUNT(st.energy_above_hull) >= 0.5
            THEN 'promising_stable_site_combination'
        WHEN COUNT(*) FILTER (WHERE st.energy_above_hull <= 0.05)::numeric
             / COUNT(st.energy_above_hull) >= 0.5
            THEN 'metastable_candidate_site_combination'
        ELSE 'low_priority_site_combination'
    END AS design_guideline
FROM site_pairs sp
LEFT JOIN stability st
    ON st.entry_id = sp.entry_id
GROUP BY
    sp.a_site_elements,
    sp.a_site_categories,
    sp.b_site_elements,
    sp.b_site_categories
ORDER BY
    stable_pct DESC NULLS LAST,
    stable_or_metastable_pct DESC NULLS LAST,
    avg_energy_above_hull_ev_per_atom ASC NULLS LAST,
    n_l12_entries DESC;
