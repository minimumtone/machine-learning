WITH a_site_l12_entries AS (
    SELECT DISTINCT
        c.entry_id,
        CASE
            WHEN e.period_number = 4 THEN '3d_transition_metal_A_site'
            WHEN e.period_number IN (5, 6) THEN '4d_5d_transition_metal_A_site'
        END AS a_site_tm_group
    FROM composition c
    JOIN element e
        ON e.symbol = c.element
    WHERE c.site_label = 'A-site'
      AND e.category = 'transition_metal'
      AND e.block = 'd'
      AND e.period_number IN (4, 5, 6)
      AND (
          EXISTS (
              SELECT 1
              FROM structure s
              LEFT JOIN prototype_definition pd
                  ON pd.prototype_id = s.prototype
              CROSS JOIN LATERAL (
                  VALUES
                      (s.strukturbericht),
                      (pd.strukturbericht),
                      (s.prototype),
                      (pd.prototype_name)
              ) AS v(label)
              WHERE s.entry_id = c.entry_id
                AND UPPER(REPLACE(REPLACE(COALESCE(v.label, ''), '_', ''), '-', '')) IN ('L12', 'L1₂')
          )
          OR EXISTS (
              SELECT 1
              FROM formation_enthalpy fh
              CROSS JOIN LATERAL (
                  VALUES
                      (fh.strukturbericht),
                      (fh.prototype)
              ) AS v(label)
              WHERE fh.entry_id = c.entry_id
                AND UPPER(REPLACE(REPLACE(COALESCE(v.label, ''), '_', ''), '-', '')) IN ('L12', 'L1₂')
          )
      )
),
entry_stability AS (
    SELECT
        entry_id,
        AVG(energy_above_hull) AS energy_above_hull_ev_per_atom,
        BOOL_OR(is_stable) AS is_stable,
        AVG(formation_energy_per_atom) AS formation_energy_ev_per_atom
    FROM phase_stability
    GROUP BY entry_id
)
SELECT
    l.a_site_tm_group,
    COUNT(*) AS compound_count,
    AVG(es.energy_above_hull_ev_per_atom) AS avg_energy_above_hull_ev_per_atom,
    AVG(CASE WHEN es.is_stable THEN 1.0 ELSE 0.0 END) AS stable_fraction,
    AVG(es.formation_energy_ev_per_atom) AS avg_formation_energy_ev_per_atom
FROM a_site_l12_entries l
JOIN entry_stability es
    ON es.entry_id = l.entry_id
GROUP BY l.a_site_tm_group
ORDER BY
    CASE l.a_site_tm_group
        WHEN '3d_transition_metal_A_site' THEN 1
        WHEN '4d_5d_transition_metal_A_site' THEN 2
    END;
