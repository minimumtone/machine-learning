WITH candidates AS (
    SELECT DISTINCT
        me.entry_id,
        COALESCE(fe.formula, me.formula) AS formula,
        me.chemical_system,
        COALESCE(ps.energy_above_hull, fe.energy_above_hull) AS energy_above_hull,
        COALESCE(ps.reference_set, fe.reference_set) AS reference_set,
        COALESCE(s.prototype, fe.prototype) AS prototype,
        COALESCE(s.strukturbericht, fe.strukturbericht, pd.strukturbericht) AS strukturbericht,
        COALESCE(s.space_group, fe.space_group) AS space_group
    FROM material_entry me
    LEFT JOIN phase_stability ps
        ON ps.entry_id = me.entry_id
    LEFT JOIN formation_enthalpy fe
        ON fe.entry_id = me.entry_id
    LEFT JOIN structure s
        ON s.entry_id = me.entry_id
    LEFT JOIN prototype_definition pd
        ON pd.prototype_id = s.prototype
    WHERE
        (ps.is_stable = TRUE OR fe.is_stable = TRUE)
        AND me.number_of_elements >= 2
        AND EXISTS (
            SELECT 1
            FROM composition c
            WHERE c.entry_id = me.entry_id
              AND c.element = 'Al'
        )
        AND NOT EXISTS (
            SELECT 1
            FROM composition c
            JOIN element e
              ON e.symbol = c.element
            WHERE c.entry_id = me.entry_id
              AND e.category NOT IN (
                  'transition_metal',
                  'post_transition_metal',
                  'lanthanide',
                  'actinide',
                  'alkali_metal',
                  'alkaline_earth_metal'
              )
        )
)
SELECT
    entry_id,
    formula,
    chemical_system,
    energy_above_hull,
    reference_set,
    prototype,
    strukturbericht,
    space_group
FROM candidates
WHERE regexp_replace(
          lower(translate(COALESCE(strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')),
          '[^a-z0-9]',
          '',
          'g'
      ) = 'l12'
ORDER BY formula, entry_id;
