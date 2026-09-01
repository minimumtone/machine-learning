WITH recent_refs AS (
    SELECT mr.entry_id, lr.reference_id, lr.year, lr.doi, lr.title
    FROM material_reference mr
    JOIN literature_reference lr ON lr.reference_id = mr.reference_id
    WHERE lr.year >= 2020

    UNION

    SELECT em.entry_id, lr.reference_id, lr.year, lr.doi, lr.title
    FROM experimental_measurement em
    JOIN literature_reference lr ON lr.reference_id = em.reference_id
    WHERE lr.year >= 2020

    UNION

    SELECT ms.entry_id, lr.reference_id, lr.year, lr.doi, lr.title
    FROM material_synthesis ms
    JOIN literature_reference lr ON lr.reference_id = ms.reference_id
    WHERE lr.year >= 2020
),
l12_entries AS (
    SELECT DISTINCT me.entry_id
    FROM material_entry me
    LEFT JOIN structure s ON s.entry_id = me.entry_id
    LEFT JOIN prototype_definition pd ON pd.prototype_id = s.prototype
    LEFT JOIN formation_enthalpy fh ON fh.entry_id = me.entry_id
    WHERE regexp_replace(upper(coalesce(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
       OR regexp_replace(upper(coalesce(pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
       OR regexp_replace(upper(coalesce(fh.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
)
SELECT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    MIN(rr.year) AS earliest_reference_year_since_2020,
    string_agg(DISTINCT concat_ws(' - ', rr.year::text, rr.doi, rr.title), '; ') AS references
FROM l12_entries le
JOIN material_entry me ON me.entry_id = le.entry_id
JOIN recent_refs rr ON rr.entry_id = me.entry_id
GROUP BY me.entry_id, me.formula, me.reduced_formula, me.chemical_system
ORDER BY me.chemical_system, me.reduced_formula, me.entry_id;
