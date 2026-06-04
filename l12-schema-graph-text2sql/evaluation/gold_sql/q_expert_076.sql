SELECT DISTINCT m.entry_id, m.formula, lr.year, lr.title
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_reference mr ON mr.entry_id = m.entry_id
JOIN literature_reference lr ON lr.reference_id = mr.reference_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND lr.year >= 2020
ORDER BY lr.year DESC, m.formula
LIMIT 10000;
