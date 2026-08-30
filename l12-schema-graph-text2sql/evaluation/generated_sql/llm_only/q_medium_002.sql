WITH matching AS (
    SELECT
        fe.entry_id,
        me.chemical_system,
        COALESCE(fe.formula, me.formula) AS formula,
        COALESCE(fe.reduced_formula, me.reduced_formula) AS reduced_formula,
        fe.formation_enthalpy_ev_per_atom,
        fe.reference_set,
        fe.prototype,
        fe.strukturbericht,
        fe.space_group
    FROM formation_enthalpy fe
    JOIN material_entry me
        ON me.entry_id = fe.entry_id
    WHERE fe.formation_enthalpy_ev_per_atom < 0
      AND (
          regexp_replace(upper(translate(COALESCE(fe.prototype, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
          OR regexp_replace(upper(translate(COALESCE(fe.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
          OR EXISTS (
              SELECT 1
              FROM structure s
              LEFT JOIN prototype_definition pd
                  ON pd.prototype_id = s.prototype
              WHERE s.entry_id = fe.entry_id
                AND (
                    regexp_replace(upper(translate(COALESCE(s.prototype, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
                    OR regexp_replace(upper(translate(COALESCE(s.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
                    OR regexp_replace(upper(translate(COALESCE(pd.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
                    OR regexp_replace(upper(translate(COALESCE(pd.prototype_name, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
                )
          )
      )
),
per_compound AS (
    SELECT DISTINCT ON (entry_id)
        *
    FROM matching
    ORDER BY entry_id, formation_enthalpy_ev_per_atom ASC
)
SELECT
    chemical_system AS element_combination,
    COUNT(*) AS compound_count,
    jsonb_agg(
        jsonb_build_object(
            'entry_id', entry_id,
            'formula', formula,
            'reduced_formula', reduced_formula,
            'formation_enthalpy_ev_per_atom', formation_enthalpy_ev_per_atom,
            'reference_set', reference_set,
            'prototype', prototype,
            'strukturbericht', strukturbericht,
            'space_group', space_group
        )
        ORDER BY formation_enthalpy_ev_per_atom ASC, reduced_formula
    ) AS compounds
FROM per_compound
GROUP BY chemical_system
ORDER BY chemical_system;
