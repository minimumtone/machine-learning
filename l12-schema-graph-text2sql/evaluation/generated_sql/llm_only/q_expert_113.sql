WITH high_melting_elements AS (
    SELECT
        e.symbol,
        e.name,
        MAX(ep.value) AS melting_point_k
    FROM element e
    JOIN element_property ep
        ON ep.element_id = e.element_id
    WHERE ep.property_name ILIKE '%melting%'
      AND ep.value >= 2000
    GROUP BY e.symbol, e.name
)
SELECT
    h.symbol,
    h.name,
    h.melting_point_k,
    COUNT(DISTINCT ps.entry_id) AS stable_compound_count
FROM high_melting_elements h
LEFT JOIN composition c
    ON c.element = h.symbol
LEFT JOIN material_entry me
    ON me.entry_id = c.entry_id
   AND me.number_of_elements > 1
LEFT JOIN phase_stability ps
    ON ps.entry_id = me.entry_id
   AND ps.is_stable = TRUE
GROUP BY
    h.symbol,
    h.name,
    h.melting_point_k
ORDER BY
    stable_compound_count DESC,
    h.symbol;
