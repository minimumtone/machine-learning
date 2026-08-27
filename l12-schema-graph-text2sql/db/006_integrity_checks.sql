-- ============================================================
-- 006_integrity_checks.sql — Post-load integrity assertions
-- Cross-row invariants that a per-row CHECK cannot express.
-- All assertions live in validate_fixture_integrity(), which RAISEs
-- (aborting the load) if any invariant is violated, so a database that
-- finishes loading is guaranteed to satisfy them. This file is
-- assertion-only and idempotent: it may be re-run at any time to
-- re-validate the loaded data, and 007_initialization_marker.sql calls
-- the same function so the initialization marker can only be created on
-- a database that passes every assertion.
-- ============================================================

CREATE OR REPLACE FUNCTION validate_fixture_integrity() RETURNS void AS $$
DECLARE
    n_bad BIGINT;
BEGIN
    -- Composition must be normalized: atomic fractions sum to 1 per entry.
    -- formation_enthalpy multiplies fractions by reference energies
    -- directly, so an un-normalized composition would corrupt the
    -- corrected enthalpy.
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT entry_id
        FROM composition
        GROUP BY entry_id
        -- NULL fractions are already rejected by the DDL (NOT NULL);
        -- the FILTER term is a second line of defense against regressions.
        HAVING COUNT(*) FILTER (WHERE atomic_fraction IS NULL) > 0
            OR ABS(SUM(atomic_fraction) - 1.0) > 1e-8
    ) t;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'composition: % entries whose atomic fractions do not sum to 1',
            n_bad;
    END IF;

    -- Every material_entry must have composition rows and its declared
    -- element count must equal the distinct composition elements
    -- (site-resolved rows may repeat an element). LEFT JOIN + COALESCE
    -- makes entries with zero composition rows fail (n = 0 <> declared).
    SELECT COUNT(*) INTO n_bad
    FROM material_entry m
    LEFT JOIN (
        SELECT entry_id, COUNT(DISTINCT element) AS n
        FROM composition
        GROUP BY entry_id
    ) c ON c.entry_id = m.entry_id
    WHERE m.number_of_elements <> COALESCE(c.n, 0);
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'material_entry: % entries whose number_of_elements disagrees with distinct composition elements',
            n_bad;
    END IF;

    -- material_entry.chemical_system must be the alphabetically sorted,
    -- hyphen-joined set of the entry's composition elements (the same
    -- string convention phase_diagram_entry uses).
    SELECT COUNT(*) INTO n_bad
    FROM material_entry m
    JOIN (
        SELECT entry_id,
               STRING_AGG(DISTINCT element, '-' ORDER BY element) AS sys
        FROM composition
        GROUP BY entry_id
    ) c ON c.entry_id = m.entry_id
    WHERE m.chemical_system IS DISTINCT FROM c.sys;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'material_entry: % entries whose chemical_system disagrees with composition elements',
            n_bad;
    END IF;

    -- NULL elemental delta_e values are rejected row-by-row by the DDL
    -- (pure_element_reference.delta_e NOT NULL); this function keeps only
    -- cross-row invariants that a single-row constraint cannot express.

    -- Set-wise reference coverage: for every energy convention actually
    -- used by phase_stability, every element of every material in that
    -- convention must have an elemental delta_e in the SAME reference_set.
    -- This is a true set difference (per reviewer guidance), not a count
    -- comparison, so it also catches per-set gaps when multiple
    -- conventions coexist.
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT DISTINCT ps.reference_set, c.element
        FROM phase_stability ps
        JOIN composition c ON c.entry_id = ps.entry_id
        EXCEPT
        SELECT per.reference_set, per.element_symbol
        FROM pure_element_reference per
    ) missing;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'pure_element_reference: % (reference_set, element) pairs used by phase_stability materials have no elemental delta_e in the same set',
            n_bad;
    END IF;

    -- Source/convention mapping: every (source_db, reference_set) pair
    -- that was actually loaded must be declared in
    -- fixture_source_reference_set, so a source can never be silently
    -- assigned an energy convention it was not mapped to.
    SELECT COUNT(*) INTO n_bad
    FROM material_entry m
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    LEFT JOIN fixture_source_reference_set sec
        ON sec.source_db IS NOT DISTINCT FROM m.source_db
       AND sec.reference_set = ps.reference_set
    WHERE sec.reference_set IS NULL;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'phase_stability: % rows whose (source_db, reference_set) pair is not declared in fixture_source_reference_set',
            n_bad;
    END IF;

    -- structure copies of master attributes must match the master tables
    -- (also enforced per-row by trg_structure_master_consistency; this is
    -- the set-level assertion for databases loaded before the trigger).
    SELECT COUNT(*) INTO n_bad
    FROM structure s
    LEFT JOIN prototype_definition p ON p.prototype_id = s.prototype
    LEFT JOIN space_group g ON g.space_group_number = s.space_group_number
    WHERE (s.prototype IS NOT NULL AND
           (s.strukturbericht IS DISTINCT FROM p.strukturbericht
            OR s.formula_type IS DISTINCT FROM p.formula_type))
       OR (s.space_group_number IS NOT NULL AND
           (s.crystal_system IS DISTINCT FROM g.crystal_system
            OR s.space_group IS DISTINCT FROM g.hermann_mauguin));
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'structure: % rows whose derived columns contradict the master tables',
            n_bad;
    END IF;

    -- Electronic-gap single truth: phase_stability.band_gap is the source
    -- of truth, so a band structure of the same material must satisfy
    -- band_gap = cbm_energy - vbm_energy.
    SELECT COUNT(*) INTO n_bad
    FROM band_structure bs
    JOIN calculation cal ON cal.calculation_id = bs.calculation_id
    JOIN phase_stability ps ON ps.entry_id = cal.entry_id
    WHERE ABS((bs.cbm_energy - bs.vbm_energy) - ps.band_gap) > 1e-6;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'band_structure: % rows whose CBM-VBM gap disagrees with phase_stability.band_gap',
            n_bad;
    END IF;

    -- Metallicity single truth: where density_of_states.is_metallic is
    -- known (non-NULL), it must equal (phase_stability.band_gap = 0).
    -- IS DISTINCT FROM keeps the comparison NULL-safe even if band_gap
    -- were ever relaxed back to nullable.
    SELECT COUNT(*) INTO n_bad
    FROM density_of_states d
    JOIN calculation cal ON cal.calculation_id = d.calculation_id
    JOIN phase_stability ps ON ps.entry_id = cal.entry_id
    WHERE d.is_metallic IS NOT NULL
      AND d.is_metallic IS DISTINCT FROM (ps.band_gap = 0);
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'density_of_states: % rows whose is_metallic contradicts phase_stability.band_gap',
            n_bad;
    END IF;

    -- Elastic-modulus single truth: the scalar moduli mirrored into
    -- calculated_property must equal the elastic_tensor values of the
    -- same calculation (both are written from one generated value).
    SELECT COUNT(*) INTO n_bad
    FROM elastic_tensor et
    JOIN calculated_property cp
        ON cp.calculation_id = et.calculation_id
       AND cp.property_name IN ('bulk_modulus', 'shear_modulus', 'youngs_modulus')
    WHERE (cp.property_name = 'bulk_modulus'
           AND cp.value IS DISTINCT FROM et.bulk_modulus_vrh)
       OR (cp.property_name = 'shear_modulus'
           AND cp.value IS DISTINCT FROM et.shear_modulus_vrh)
       OR (cp.property_name = 'youngs_modulus'
           AND cp.value IS DISTINCT FROM et.youngs_modulus);
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'elastic_tensor: % scalar modulus rows in calculated_property disagree with the tensor table',
            n_bad;
    END IF;

    -- Hull-distance single truth: phase_diagram_entry.hull_distance must
    -- equal phase_stability.energy_above_hull for the same entry (both
    -- generated stability flags then agree by construction).
    SELECT COUNT(*) INTO n_bad
    FROM phase_diagram_entry pde
    JOIN phase_stability ps ON ps.entry_id = pde.entry_id
    WHERE pde.hull_distance IS DISTINCT FROM ps.energy_above_hull;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'phase_diagram_entry: % rows whose hull_distance disagrees with phase_stability.energy_above_hull',
            n_bad;
    END IF;

    -- phase_diagram_entry.chemical_system must match the material's.
    SELECT COUNT(*) INTO n_bad
    FROM phase_diagram_entry pde
    JOIN material_entry m ON m.entry_id = pde.entry_id
    WHERE pde.chemical_system IS DISTINCT FROM m.chemical_system;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'phase_diagram_entry: % rows whose chemical_system disagrees with material_entry',
            n_bad;
    END IF;

    -- Benchmark simplification the gold SQL relies on: the fixture stores
    -- exactly one calculation per material entry, so gold queries that join
    -- calculation without restricting calculation_type/method/functional
    -- cannot multiply rows. Loading a second calculation for an entry must
    -- fail here until the gold SQL is updated to select calculations.
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT entry_id FROM calculation GROUP BY entry_id HAVING COUNT(*) > 1
    ) multi;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'calculation: % entries with more than one calculation (single-calculation fixture convention violated)',
            n_bad;
    END IF;

    -- Benchmark temperature convention: every calculation with thermal data
    -- has exactly one row at the representative 300 K temperature that the
    -- gold SQL selects (tp.temperature_k = 300); additional temperatures
    -- (500 K / 800 K) exist so unfiltered joins visibly multiply rows.
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT calculation_id
        FROM thermal_property
        GROUP BY calculation_id
        HAVING COUNT(*) FILTER (WHERE temperature_k = 300.0) <> 1
    ) missing_300k;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'thermal_property: % calculations without exactly one 300 K row',
            n_bad;
    END IF;

    -- Lattice geometry consistency: hexagonal structures must have a = b,
    -- cubic structures a = b = c.
    SELECT COUNT(*) INTO n_bad
    FROM structure
    WHERE lattice_a IS NOT NULL AND lattice_b IS NOT NULL
      AND (
          (crystal_system = 'hexagonal' AND lattice_a <> lattice_b)
       OR (crystal_system = 'cubic'
           AND (lattice_a <> lattice_b OR lattice_a <> lattice_c))
      );
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'structure: % rows whose lattice parameters contradict their crystal system',
            n_bad;
    END IF;

    -- volume_per_atom must equal the conventional-cell volume divided by
    -- the prototype's conventional_cell_atoms (cubic V=a^3, hexagonal
    -- V=(sqrt(3)/2)a^2c).
    SELECT COUNT(*) INTO n_bad
    FROM structure s
    JOIN prototype_definition pd ON pd.prototype_id = s.prototype
    WHERE pd.conventional_cell_atoms IS NOT NULL
      AND s.crystal_system NOT IN ('cubic', 'hexagonal');
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'structure: % rows use a crystal system the volume consistency check has no formula for (add its cell-volume formula before loading such prototypes)',
            n_bad;
    END IF;

    SELECT COUNT(*) INTO n_bad
    FROM structure s
    JOIN prototype_definition pd ON pd.prototype_id = s.prototype
    WHERE pd.conventional_cell_atoms IS NOT NULL
      AND s.lattice_a IS NOT NULL AND s.lattice_c IS NOT NULL
      AND s.volume_per_atom IS NOT NULL
      AND ABS(
          s.volume_per_atom
          - (CASE s.crystal_system
                 WHEN 'cubic' THEN s.lattice_a ^ 3
                 WHEN 'hexagonal' THEN sqrt(3.0) / 2.0 * s.lattice_a ^ 2 * s.lattice_c
             END) / pd.conventional_cell_atoms
      ) > 1e-3;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'structure: % rows whose volume_per_atom contradicts lattice parameters and conventional_cell_atoms',
            n_bad;
    END IF;
END;
$$ LANGUAGE plpgsql;

SELECT validate_fixture_integrity();
