SELECT stable_elements.element_symbol,
       COUNT(*) AS stable_compound_count
FROM (
    SELECT regexp_split_to_table(m.chemical_system, '-') AS element_symbol
    FROM material_entry m
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    WHERE ps.is_stable = TRUE
      AND m.number_of_elements > 1
) AS stable_elements
WHERE stable_elements.element_symbol IN ('B', 'C', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Th')
GROUP BY stable_elements.element_symbol
ORDER BY stable_elements.element_symbol
LIMIT 10000;
