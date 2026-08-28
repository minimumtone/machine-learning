SELECT m.entry_id, m.formula
FROM material_entry m
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
JOIN synthesis_method sm ON sm.synthesis_id = ms.synthesis_id
WHERE sm.method_name = 'Ball Milling'
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;
