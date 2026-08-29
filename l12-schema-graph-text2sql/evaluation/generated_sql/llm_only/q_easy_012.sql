SELECT COUNT(DISTINCT entry_id) AS l12_compound_count
FROM (
    SELECT s.entry_id
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON s.prototype = pd.prototype_id
    WHERE regexp_replace(translate(upper(COALESCE(s.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
       OR regexp_replace(translate(upper(COALESCE(s.prototype, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
       OR regexp_replace(translate(upper(COALESCE(pd.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
       OR regexp_replace(translate(upper(COALESCE(pd.prototype_name, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g') LIKE '%L12%'

    UNION

    SELECT fe.entry_id
    FROM formation_enthalpy fe
    WHERE regexp_replace(translate(upper(COALESCE(fe.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
       OR regexp_replace(translate(upper(COALESCE(fe.prototype, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
) AS l12_entries;
