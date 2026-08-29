SELECT DISTINCT m.formula, s.prototype, s.strukturbericht, sm.method_name
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
JOIN synthesis_method sm ON sm.synthesis_id = ms.synthesis_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ms.success = TRUE
  AND sm.method_name ILIKE '%arc%'
  AND sm.method_name ILIKE '%melt%'
ORDER BY m.formula
LIMIT 10000;
