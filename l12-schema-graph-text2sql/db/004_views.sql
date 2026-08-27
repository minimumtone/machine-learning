-- ============================================================
-- 004_views.sql — Derived views
-- ============================================================

-- Formation enthalpy view. Defined quantity (single, explicit choice):
--
--   formation_enthalpy_ev_per_atom
--       = phase_stability.formation_energy_per_atom
--       = the fixture formation energy under ps.reference_set
--         (L12-FIXTURE-PBE-v1 in this package), relative to the fitted
--         elemental reference states of that set. Compound values are
--         synthetic fixture data, NOT compound formation energies copied
--         from OQMD/MP/AFLOW; only the pure-element delta_e values in
--         pure_element_reference are adopted OQMD DFT-PBE data.
--       This IS the formation enthalpy; no further subtraction is applied
--       to it (subtracting elemental delta_e again would double-correct).
--
--   enthalpy_vs_element_ground_states
--       = ps.formation_energy_per_atom - SUM(x_i * per.delta_e_i)
--       = the formation energy re-referenced to the stored pure-element
--         GROUND STATES instead of the fitted reference states. This is a
--         well-defined re-referencing (the fitted reference energies cancel
--         exactly) and is only computed when every constituent element has
--         a delta_e in the SAME reference_set as the material
--         (per.reference_set = ps.reference_set — never a fixed set, so
--         conventions are never mixed across source databases).
--
-- reference_status makes a NULL re-referenced value diagnosable instead of
-- silent (missing reference / composition problems / count mismatch).
-- Controlled vocabulary (exhaustive): 'ok', 'missing_composition',
-- 'element_count_mismatch', 'missing_composition_fraction',
-- 'invalid_composition', 'missing_reference_for_set'.
CREATE OR REPLACE VIEW formation_enthalpy AS
SELECT
    m.entry_id,
    m.formula,
    m.reduced_formula,
    ps.formation_energy_per_atom AS formation_enthalpy_ev_per_atom,
    ps.reference_set,
    ps.energy_above_hull,
    ps.is_stable,
    s.prototype,
    s.strukturbericht,
    s.space_group,
    s.lattice_a,
    ref.weighted_element_delta_e,
    CASE
        WHEN ref.n_elements = ref.n_referenced
         -- The declared element count must match the actual composition;
         -- the view re-checks this and the normalization itself instead of
         -- trusting that the one-time 006 assertion covered later inserts.
         AND m.number_of_elements = ref.n_elements
         AND ABS(ref.fraction_sum - 1.0) <= 1e-8
        THEN ps.formation_energy_per_atom - ref.weighted_element_delta_e
        ELSE NULL
    END AS enthalpy_vs_element_ground_states,
    -- Every NULL-able input is handled by an explicit branch BEFORE the
    -- comparisons that would go NULL on it, so no diagnosable state can
    -- fall through a NULL comparison into 'ok'.
    CASE
        WHEN ref.n_elements = 0
            THEN 'missing_composition'
        WHEN m.number_of_elements <> ref.n_elements
            THEN 'element_count_mismatch'
        WHEN ref.fraction_sum IS NULL
            THEN 'missing_composition_fraction'
        WHEN ABS(ref.fraction_sum - 1.0) > 1e-8
            THEN 'invalid_composition'
        WHEN ref.n_elements <> ref.n_referenced
            THEN 'missing_reference_for_set'
        ELSE 'ok'
    END AS reference_status
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
LEFT JOIN structure s ON s.entry_id = m.entry_id
LEFT JOIN LATERAL (
    SELECT
        -- Composition is site-resolved (the same element may appear on
        -- several sites), so element counts must be DISTINCT over elements,
        -- not raw row counts.
        COUNT(DISTINCT c.element) AS n_elements,
        -- A reference element only counts when it actually carries a
        -- delta_e (NOT NULL by DDL, but the guard stays defensive so a
        -- schema relaxation could not silently drop terms from the SUM).
        COUNT(DISTINCT c.element) FILTER (
            WHERE per.element_symbol IS NOT NULL
              AND per.delta_e IS NOT NULL
        ) AS n_referenced,
        SUM(c.atomic_fraction) AS fraction_sum,
        SUM(c.atomic_fraction * per.delta_e) AS weighted_element_delta_e
    FROM composition c
    LEFT JOIN pure_element_reference per
        ON per.element_symbol = c.element
       -- Join on the material's OWN energy convention: elemental delta_e
       -- values are only subtractable within the same reference_set.
       AND per.reference_set = ps.reference_set
    WHERE c.entry_id = m.entry_id
) ref ON TRUE
WHERE m.number_of_elements >= 2;

COMMENT ON VIEW formation_enthalpy IS
    'Formation enthalpy per compound. reference_status is a closed '
    'vocabulary: ok, missing_composition, element_count_mismatch, '
    'missing_composition_fraction, invalid_composition, '
    'missing_reference_for_set.';
