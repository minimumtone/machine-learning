WITH l12_structures AS (
    SELECT DISTINCT ON (s.entry_id)
        s.entry_id,
        s.prototype,
        s.strukturbericht,
        s.space_group_number,
        s.crystal_system,
        s.lattice_a,
        ABS(s.lattice_a - 3.57) AS lattice_a_diff_from_ni3al
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON s.prototype = pd.prototype_id
    WHERE s.lattice_a IS NOT NULL
      AND (
          upper(regexp_replace(translate(coalesce(s.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Za-z0-9]', '', 'g')) LIKE '%L12%'
          OR upper(regexp_replace(translate(coalesce(pd.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Za-z0-9]', '', 'g')) LIKE '%L12%'
          OR pd.prototype_name ILIKE '%Cu3Au%'
          OR s.prototype ILIKE '%Cu3Au%'
      )
    ORDER BY s.entry_id, ABS(s.lattice_a - 3.57), s.structure_id
),
hull AS (
    SELECT
        entry_id,
        MIN(energy_above_hull) AS energy_above_hull
    FROM phase_stability
    WHERE energy_above_hull <= 0.05
    GROUP BY entry_id
),
elastic AS (
    SELECT DISTINCT ON (c.entry_id)
        c.entry_id,
        et.bulk_modulus_vrh,
        c.method,
        c.functional
    FROM calculation c
    JOIN elastic_tensor et
        ON c.calculation_id = et.calculation_id
    WHERE et.bulk_modulus_vrh IS NOT NULL
    ORDER BY c.entry_id, c.calculation_id
)
SELECT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    h.energy_above_hull,
    l.prototype,
    l.strukturbericht,
    l.space_group_number,
    l.crystal_system,
    l.lattice_a,
    l.lattice_a_diff_from_ni3al,
    e.bulk_modulus_vrh,
    e.method,
    e.functional
FROM material_entry me
JOIN l12_structures l
    ON me.entry_id = l.entry_id
JOIN hull h
    ON me.entry_id = h.entry_id
JOIN elastic e
    ON me.entry_id = e.entry_id
ORDER BY
    h.energy_above_hull ASC,
    l.lattice_a_diff_from_ni3al ASC;
