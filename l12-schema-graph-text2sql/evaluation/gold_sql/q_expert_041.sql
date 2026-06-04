SELECT DISTINCT m.entry_id, m.formula, ca.functional
FROM material_entry m
JOIN calculation ca ON ca.entry_id = m.entry_id
WHERE ca.functional IS NOT NULL AND ca.functional != 'GGA-PBE' AND ca.functional != 'PBE'
ORDER BY m.formula
LIMIT 10000;
