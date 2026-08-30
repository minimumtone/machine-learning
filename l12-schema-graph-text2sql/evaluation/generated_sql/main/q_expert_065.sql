SELECT m.formula, md.formation_energy, dt.defect_name
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_defect md ON md.entry_id = m.entry_id
JOIN defect_type dt ON dt.defect_type_id = md.defect_type_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND (dt.category = 'vacancy' OR dt.defect_name ILIKE '%vacancy%')
ORDER BY md.formation_energy ASC
LIMIT 1;
