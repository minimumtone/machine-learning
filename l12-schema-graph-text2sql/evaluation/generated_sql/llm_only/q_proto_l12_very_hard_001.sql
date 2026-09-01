WITH l12_entries AS (
    SELECT DISTINCT s.entry_id
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON s.prototype = pd.prototype_id
    WHERE EXISTS (
        SELECT 1
        FROM unnest(ARRAY[s.strukturbericht, s.prototype, pd.strukturbericht, pd.prototype_name]) AS v(label)
        WHERE regexp_replace(
                  upper(translate(coalesce(v.label, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')),
                  '[^A-Z0-9]',
                  '',
                  'g'
              ) = 'L12'
    )

    UNION

    SELECT DISTINCT fe.entry_id
    FROM formation_enthalpy fe
    WHERE EXISTS (
        SELECT 1
        FROM unnest(ARRAY[fe.strukturbericht, fe.prototype]) AS v(label)
        WHERE regexp_replace(
                  upper(translate(coalesce(v.label, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')),
                  '[^A-Z0-9]',
                  '',
                  'g'
              ) = 'L12'
    )
),
site_composition AS (
    SELECT
        c.entry_id,
        string_agg(DISTINCT c.element, '-' ORDER BY c.element) FILTER (WHERE c.site_label = 'A-site') AS a_site_elements,
        string_agg(DISTINCT c.element, '-' ORDER BY c.element) FILTER (WHERE c.site_label = 'B-site') AS b_site_elements
    FROM composition c
    JOIN l12_entries l
        ON c.entry_id = l.entry_id
    WHERE c.site_label IN ('A-site', 'B-site')
    GROUP BY c.entry_id
),
entry_stability AS (
    SELECT
        ps.entry_id,
        MIN(ps.energy_above_hull) AS energy_above_hull
    FROM phase_stability ps
    JOIN l12_entries l
        ON ps.entry_id = l.entry_id
    GROUP BY ps.entry_id
),
classified AS (
    SELECT
        sc.a_site_elements,
        sc.b_site_elements,
        es.energy_above_hull,
        CASE
            WHEN es.energy_above_hull <= 0.001 THEN 'stable'
            WHEN es.energy_above_hull <= 0.05 THEN 'metastable'
            ELSE 'unstable'
        END AS stability_class
    FROM site_composition sc
    JOIN entry_stability es
        ON sc.entry_id = es.entry_id
    WHERE sc.a_site_elements IS NOT NULL
      AND sc.b_site_elements IS NOT NULL
)
SELECT
    a_site_elements,
    b_site_elements,
    COUNT(*) AS total_compounds,
    COUNT(*) FILTER (WHERE stability_class = 'stable') AS stable_count,
    COUNT(*) FILTER (WHERE stability_class = 'metastable') AS metastable_count,
    COUNT(*) FILTER (WHERE stability_class = 'unstable') AS unstable_count,
    AVG(energy_above_hull) AS avg_energy_above_hull_ev_per_atom,
    MIN(energy_above_hull) AS min_energy_above_hull_ev_per_atom
FROM classified
GROUP BY
    a_site_elements,
    b_site_elements
ORDER BY
    stable_count DESC,
    metastable_count DESC,
    total_compounds DESC,
    a_site_elements,
    b_site_elements;
