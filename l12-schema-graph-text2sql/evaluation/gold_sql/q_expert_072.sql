SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
JOIN synthesis_method sm ON sm.synthesis_id = ms.synthesis_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND sm.method_name = 'Arc Melting'
ORDER BY m.formula
LIMIT 10000;
