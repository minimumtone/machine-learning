SELECT m.entry_id, m.formula, COUNT(mr.reference_id) AS ref_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_reference mr ON mr.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
GROUP BY m.entry_id, m.formula
HAVING COUNT(mr.reference_id) >= 3
ORDER BY ref_count DESC
LIMIT 10000;
