WITH top_elements AS (
    SELECT
        element_symbol,
        MAX(n_polymorphs) AS n_polymorphs
    FROM pure_element_reference
    GROUP BY element_symbol
    ORDER BY MAX(n_polymorphs) DESC NULLS LAST, element_symbol
    LIMIT 5
),
l12_entry_ids AS (
    SELECT DISTINCT entry_id
    FROM formation_enthalpy
    WHERE regexp_replace(upper(translate(COALESCE(strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
       OR regexp_replace(upper(translate(COALESCE(prototype, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'

    UNION

    SELECT DISTINCT s.entry_id
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON s.prototype = pd.prototype_id
    WHERE regexp_replace(upper(translate(COALESCE(s.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
       OR regexp_replace(upper(translate(COALESCE(s.prototype, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
       OR regexp_replace(upper(translate(COALESCE(pd.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
       OR regexp_replace(upper(translate(COALESCE(pd.prototype_name, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789')), '[^A-Z0-9]', '', 'g') = 'L12'
),
l12_entries AS (
    SELECT
        f.entry_id,
        AVG(f.formation_enthalpy_ev_per_atom) AS formation_energy_ev_per_atom
    FROM formation_enthalpy f
    JOIN l12_entry_ids l
        ON f.entry_id = l.entry_id
    GROUP BY f.entry_id
)
SELECT
    te.element_symbol,
    te.n_polymorphs,
    COUNT(DISTINCT le.entry_id) AS l12_compound_count,
    AVG(le.formation_energy_ev_per_atom) AS avg_formation_energy_ev_per_atom
FROM top_elements te
LEFT JOIN composition c
    ON c.element = te.element_symbol
LEFT JOIN l12_entries le
    ON le.entry_id = c.entry_id
GROUP BY te.element_symbol, te.n_polymorphs
ORDER BY te.n_polymorphs DESC NULLS LAST, te.element_symbol;
