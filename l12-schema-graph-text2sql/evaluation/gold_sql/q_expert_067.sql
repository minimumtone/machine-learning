SELECT m.entry_id, m.formula, e.symbol AS dopant
FROM material_entry m
JOIN material_defect md ON md.entry_id = m.entry_id
JOIN element e ON e.element_id = md.dopant_element_id
WHERE e.symbol = 'B'
ORDER BY m.formula
LIMIT 10000;
