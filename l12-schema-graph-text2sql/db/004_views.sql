-- ============================================================
-- 004_views.sql — Derived views
-- ============================================================

-- Formation enthalpy view: corrected ΔH_f using pure-element reference
-- energies. The weighted reference energy is computed once (LATERAL) and the
-- corrected value is NULL unless a reference energy exists for every
-- constituent element (missing reference data is never silently treated as
-- zero).
CREATE OR REPLACE VIEW formation_enthalpy AS
SELECT
    m.entry_id,
    m.formula,
    m.reduced_formula,
    ps.formation_energy_per_atom AS formation_enthalpy_ev_per_atom,
    ps.energy_above_hull,
    ps.is_stable,
    s.prototype,
    s.strukturbericht,
    s.space_group,
    s.lattice_a,
    ref.weighted_ref_energy,
    CASE
        WHEN ref.n_elements = ref.n_referenced
        THEN ps.formation_energy_per_atom - ref.weighted_ref_energy
        ELSE NULL
    END AS corrected_formation_enthalpy
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
LEFT JOIN structure s ON s.entry_id = m.entry_id
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS n_elements,
        COUNT(per.element_symbol) AS n_referenced,
        SUM(c.atomic_fraction * per.energy_per_atom) AS weighted_ref_energy
    FROM composition c
    LEFT JOIN pure_element_reference per ON per.element_symbol = c.element
    WHERE c.entry_id = m.entry_id
) ref ON TRUE
WHERE m.number_of_elements >= 2;
