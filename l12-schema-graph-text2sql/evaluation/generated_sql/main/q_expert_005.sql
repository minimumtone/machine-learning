SELECT COUNT(*) AS nacl_type_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'NaCl' OR s.prototype = 'B1' OR s.strukturbericht = 'B1')
LIMIT 10000;
