WITH composition_summary AS (
    SELECT
        entry_id,
        SUM(CASE WHEN element = 'Co' THEN atomic_fraction ELSE 0 END) AS co_atomic_fraction,
        SUM(CASE WHEN element = 'Ni' THEN atomic_fraction ELSE 0 END) AS ni_atomic_fraction,
        SUM(CASE WHEN element = 'Al' THEN atomic_fraction ELSE 0 END) AS al_atomic_fraction
    FROM composition
    GROUP BY entry_id
),
l12_compounds AS (
    SELECT
        fe.entry_id,
        COALESCE(me.formula, fe.formula) AS formula,
        COALESCE(me.reduced_formula, fe.reduced_formula) AS reduced_formula,
        me.chemical_system,
        fe.reference_set,
        fe.formation_enthalpy_ev_per_atom,
        fe.energy_above_hull,
        fe.is_stable,
        COALESCE(fe.prototype, s.prototype, pd.prototype_name) AS prototype,
        COALESCE(fe.strukturbericht, s.strukturbericht, pd.strukturbericht) AS strukturbericht,
        COALESCE(fe.space_group, s.space_group) AS space_group,
        COALESCE(fe.lattice_a, s.lattice_a) AS lattice_a,
        cs.co_atomic_fraction,
        cs.ni_atomic_fraction,
        cs.al_atomic_fraction
    FROM formation_enthalpy fe
    JOIN material_entry me
        ON me.entry_id = fe.entry_id
    LEFT JOIN LATERAL (
        SELECT *
        FROM structure s
        WHERE s.entry_id = fe.entry_id
        ORDER BY s.structure_id
        LIMIT 1
    ) s ON TRUE
    LEFT JOIN prototype_definition pd
        ON pd.prototype_id = COALESCE(s.prototype, fe.prototype)
    LEFT JOIN composition_summary cs
        ON cs.entry_id = fe.entry_id
    WHERE
        COALESCE(fe.strukturbericht, s.strukturbericht, pd.strukturbericht, '') ~* 'L1([_ -]?2|₂)|L12'
        OR COALESCE(fe.prototype, s.prototype, pd.prototype_name, '') ~* 'L1([_ -]?2|₂)|L12|Cu3Au'
),
ni3al AS (
    SELECT DISTINCT ON (reference_set)
        *
    FROM l12_compounds
    WHERE
        reduced_formula IN ('Ni3Al', 'AlNi3')
        OR formula IN ('Ni3Al', 'AlNi3')
        OR (
            chemical_system = 'Al-Ni'
            AND ABS(COALESCE(ni_atomic_fraction, -1) - 0.75) <= 0.02
            AND ABS(COALESCE(al_atomic_fraction, -1) - 0.25) <= 0.02
        )
    ORDER BY
        reference_set,
        CASE WHEN is_stable THEN 0 ELSE 1 END,
        energy_above_hull NULLS LAST,
        formation_enthalpy_ev_per_atom NULLS LAST
)
SELECT
    c.entry_id,
    c.formula,
    c.reduced_formula,
    c.chemical_system,
    c.reference_set,
    c.prototype,
    c.strukturbericht,
    c.space_group,
    c.co_atomic_fraction,
    c.formation_enthalpy_ev_per_atom,
    c.energy_above_hull,
    c.lattice_a,
    n.entry_id AS ni3al_entry_id,
    n.formation_enthalpy_ev_per_atom AS ni3al_formation_enthalpy_ev_per_atom,
    n.energy_above_hull AS ni3al_energy_above_hull,
    n.lattice_a AS ni3al_lattice_a,
    c.formation_enthalpy_ev_per_atom - n.formation_enthalpy_ev_per_atom AS delta_formation_enthalpy_vs_ni3al_ev_per_atom,
    100.0 * (c.lattice_a - n.lattice_a) / NULLIF(n.lattice_a, 0) AS lattice_mismatch_vs_ni3al_pct,
    CASE
        WHEN c.lattice_a IS NULL OR n.lattice_a IS NULL OR n.lattice_a = 0
            THEN 'stable_L12_Co_containing_no_lattice_comparison'
        WHEN ABS((c.lattice_a - n.lattice_a) / n.lattice_a) <= 0.01
             AND c.formation_enthalpy_ev_per_atom <= n.formation_enthalpy_ev_per_atom
            THEN 'strong_candidate_stable_lattice_matched_and_as_or_more_exothermic_than_Ni3Al'
        WHEN ABS((c.lattice_a - n.lattice_a) / n.lattice_a) <= 0.01
            THEN 'good_candidate_stable_and_lattice_matched_to_Ni3Al'
        WHEN ABS((c.lattice_a - n.lattice_a) / n.lattice_a) <= 0.02
            THEN 'possible_candidate_stable_with_moderate_lattice_mismatch'
        ELSE 'stable_L12_Co_containing_but_large_lattice_mismatch_vs_Ni3Al'
    END AS gamma_prime_candidate_assessment
FROM l12_compounds c
JOIN ni3al n
    ON n.reference_set = c.reference_set
WHERE
    c.is_stable = TRUE
    AND (
        COALESCE(c.co_atomic_fraction, 0) > 0
        OR ('-' || c.chemical_system || '-') LIKE '%-Co-%'
    )
ORDER BY
    ABS(100.0 * (c.lattice_a - n.lattice_a) / NULLIF(n.lattice_a, 0)) NULLS LAST,
    c.energy_above_hull NULLS LAST,
    c.formation_enthalpy_ev_per_atom;
