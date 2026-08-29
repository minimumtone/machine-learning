WITH stable_entries AS (
    SELECT entry_id, MIN(energy_above_hull) AS energy_above_hull
    FROM (
        SELECT entry_id, energy_above_hull
        FROM phase_stability
        WHERE is_stable = TRUE
        UNION ALL
        SELECT entry_id, energy_above_hull
        FROM formation_enthalpy
        WHERE is_stable = TRUE
    ) s
    GROUP BY entry_id
),
structure_sources AS (
    SELECT
        s.entry_id,
        s.lattice_a,
        concat_ws(' ', s.prototype, s.strukturbericht, pd.prototype_name, pd.strukturbericht) AS proto_text
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON pd.prototype_id = s.prototype
    UNION ALL
    SELECT
        fe.entry_id,
        fe.lattice_a,
        concat_ws(' ', fe.prototype, fe.strukturbericht) AS proto_text
    FROM formation_enthalpy fe
),
l12_entries AS (
    SELECT DISTINCT entry_id
    FROM structure_sources
    WHERE regexp_replace(
              translate(upper(coalesce(proto_text, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'),
              '[^A-Z0-9]', '', 'g'
          ) LIKE '%L12%'
       OR proto_text ILIKE '%Cu3Au%'
),
lattice_info AS (
    SELECT entry_id, MIN(lattice_a) AS lattice_a
    FROM structure_sources
    WHERE lattice_a IS NOT NULL
    GROUP BY entry_id
),
candidates AS (
    SELECT
        me.entry_id,
        me.formula,
        me.reduced_formula,
        li.lattice_a,
        et.bulk_modulus_vrh,
        et.shear_modulus_vrh,
        et.bulk_modulus_vrh / et.shear_modulus_vrh AS bg_ratio,
        se.energy_above_hull
    FROM stable_entries se
    JOIN material_entry me
        ON me.entry_id = se.entry_id
    JOIN l12_entries l12
        ON l12.entry_id = me.entry_id
    JOIN lattice_info li
        ON li.entry_id = me.entry_id
    JOIN calculation c
        ON c.entry_id = me.entry_id
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE li.lattice_a <= 4.0
      AND et.bulk_modulus_vrh IS NOT NULL
      AND et.shear_modulus_vrh IS NOT NULL
      AND et.shear_modulus_vrh > 0
      AND et.bulk_modulus_vrh / et.shear_modulus_vrh >= 2.0
),
best_per_compound AS (
    SELECT DISTINCT ON (entry_id)
        entry_id,
        formula,
        reduced_formula,
        lattice_a,
        bulk_modulus_vrh,
        shear_modulus_vrh,
        bg_ratio,
        energy_above_hull
    FROM candidates
    ORDER BY entry_id, bg_ratio DESC, lattice_a ASC, energy_above_hull ASC
)
SELECT
    RANK() OVER (ORDER BY bg_ratio DESC) AS ranking,
    entry_id,
    formula,
    reduced_formula,
    lattice_a,
    bulk_modulus_vrh,
    shear_modulus_vrh,
    bg_ratio,
    energy_above_hull
FROM best_per_compound
ORDER BY ranking, entry_id;
