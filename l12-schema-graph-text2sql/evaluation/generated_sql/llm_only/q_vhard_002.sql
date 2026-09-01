WITH stability AS (
    SELECT DISTINCT ON (entry_id)
        entry_id,
        reference_set,
        formation_energy_per_atom,
        energy_above_hull
    FROM phase_stability
    WHERE energy_above_hull <= 0.05
    ORDER BY entry_id, formation_energy_per_atom ASC NULLS LAST, energy_above_hull ASC NULLS LAST
),
l12_structure AS (
    SELECT DISTINCT ON (s.entry_id)
        s.entry_id,
        s.structure_id,
        s.prototype,
        COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
        s.space_group_number,
        s.lattice_a,
        ABS(s.lattice_a - 3.57) AS lattice_a_diff_from_ni3al_angstrom
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON pd.prototype_id = s.prototype
    WHERE s.lattice_a IS NOT NULL
      AND (
          regexp_replace(
              lower(translate(COALESCE(s.strukturbericht, pd.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')),
              '[^a-z0-9]', '', 'g'
          ) = 'l12'
          OR regexp_replace(lower(COALESCE(s.prototype, pd.prototype_id, '')), '[^a-z0-9]', '', 'g') LIKE '%l12%'
          OR lower(COALESCE(pd.prototype_name, '')) LIKE '%cu3au%'
          OR regexp_replace(
              lower(translate(COALESCE(pd.prototype_name, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')),
              '[^a-z0-9]', '', 'g'
          ) LIKE '%l12%'
      )
    ORDER BY s.entry_id, ABS(s.lattice_a - 3.57) ASC NULLS LAST
),
bulk_modulus AS (
    SELECT DISTINCT ON (c.entry_id)
        c.entry_id,
        c.calculation_id,
        c.method,
        c.functional,
        et.bulk_modulus_vrh
    FROM calculation c
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE et.bulk_modulus_vrh IS NOT NULL
    ORDER BY c.entry_id, et.bulk_modulus_vrh DESC NULLS LAST, c.calculation_id
)
SELECT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    st.formation_energy_per_atom AS formation_energy_ev_per_atom,
    st.energy_above_hull AS energy_above_hull_ev_per_atom,
    l12.lattice_a AS lattice_a_angstrom,
    l12.lattice_a_diff_from_ni3al_angstrom,
    bm.bulk_modulus_vrh AS bulk_modulus_vrh_gpa,
    bm.method,
    bm.functional,
    l12.prototype,
    l12.strukturbericht,
    st.reference_set
FROM material_entry me
JOIN stability st
    ON st.entry_id = me.entry_id
JOIN l12_structure l12
    ON l12.entry_id = me.entry_id
JOIN bulk_modulus bm
    ON bm.entry_id = me.entry_id
WHERE me.number_of_elements >= 2
ORDER BY
    st.formation_energy_per_atom ASC NULLS LAST,
    l12.lattice_a_diff_from_ni3al_angstrom ASC NULLS LAST;
