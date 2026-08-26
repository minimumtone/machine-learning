SELECT m.entry_id, m.formula
FROM material_entry m
JOIN calculation ca ON ca.entry_id = m.entry_id
JOIN experimental_measurement em ON em.entry_id = m.entry_id
ORDER BY m.formula
LIMIT 10000;
