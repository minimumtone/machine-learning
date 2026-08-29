SELECT COUNT(*) AS nacl_type_compound_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'NaCl' OR s.strukturbericht = 'B1')
LIMIT 10000;
