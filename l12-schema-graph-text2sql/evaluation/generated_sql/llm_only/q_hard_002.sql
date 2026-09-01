WITH l12_entries AS (
    SELECT DISTINCT me.entry_id
    FROM material_entry me
    LEFT JOIN structure s
        ON s.entry_id = me.entry_id
    LEFT JOIN formation_enthalpy fh
        ON fh.entry_id = me.entry_id
    WHERE
        UPPER(COALESCE(s.strukturbericht, fh.strukturbericht, '')) IN ('L12', 'L1_2', 'L1₂')
        OR UPPER(COALESCE(s.prototype, fh.prototype, '')) LIKE '%L12%'
        OR UPPER(COALESCE(s.prototype, fh.prototype, '')) LIKE '%L1_2%'
        OR COALESCE(s.prototype, fh.prototype, '') LIKE '%L1₂%'
),
site_pairs AS (
    SELECT
        c.entry_id,
        STRING_AGG(DISTINCT c.element, ',' ORDER BY c.element) FILTER (WHERE c.site_label = 'A-site') AS a_site_elements,
        STRING_AGG(DISTINCT c.element, ',' ORDER BY c.element) FILTER (WHERE c.site_label = 'B-site') AS b_site_elements
    FROM composition c
    GROUP BY c.entry_id
),
entry_energies AS (
    SELECT
        le.entry_id,
        me.formula,
        sp.a_site_elements,
        sp.b_site_elements,
        COALESCE(fh.formation_enthalpy_ev_per_atom, ps.formation_energy_per_atom) AS formation_energy_ev_per_atom,
        COALESCE(fh.energy_above_hull, ps.energy_above_hull) AS energy_above_hull_ev_per_atom,
        COALESCE(fh.is_stable, ps.is_stable) AS is_stable
    FROM l12_entries le
    JOIN material_entry me
        ON me.entry_id = le.entry_id
    JOIN site_pairs sp
        ON sp.entry_id = le.entry_id
    LEFT JOIN formation_enthalpy fh
        ON fh.entry_id = le.entry_id
    LEFT JOIN phase_stability ps
        ON ps.entry_id = le.entry_id
    WHERE sp.a_site_elements IS NOT NULL
      AND sp.b_site_elements IS NOT NULL
      AND COALESCE(fh.formation_enthalpy_ev_per_atom, ps.formation_energy_per_atom) IS NOT NULL
)
SELECT
    a_site_elements,
    b_site_elements,
    COUNT(*) AS n_compounds,
    AVG(formation_energy_ev_per_atom) AS avg_formation_energy_ev_per_atom,
    MIN(formation_energy_ev_per_atom) AS min_formation_energy_ev_per_atom,
    MAX(formation_energy_ev_per_atom) AS max_formation_energy_ev_per_atom,
    STDDEV_SAMP(formation_energy_ev_per_atom) AS stddev_formation_energy_ev_per_atom,
    AVG(energy_above_hull_ev_per_atom) AS avg_energy_above_hull_ev_per_atom,
    COUNT(*) FILTER (WHERE is_stable = TRUE) AS n_stable,
    STRING_AGG(formula || ':' || formation_energy_ev_per_atom::text, '; ' ORDER BY formation_energy_ev_per_atom) AS compounds_by_formation_energy
FROM entry_energies
GROUP BY a_site_elements, b_site_elements
ORDER BY avg_formation_energy_ev_per_atom ASC, n_compounds DESC;
