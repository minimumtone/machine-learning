SELECT m.formula, sm.method_name, ms.temperature_k, ms.duration_hours, ms.atmosphere
FROM material_entry m
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
JOIN synthesis_method sm ON sm.synthesis_id = ms.synthesis_id
WHERE ms.success = TRUE
  AND (
    sm.method_name ILIKE '%ball milling%'
    OR sm.description ILIKE '%ball milling%'
    OR sm.method_name ILIKE '%ボールミリング%'
    OR sm.description ILIKE '%ボールミリング%'
  )
ORDER BY m.formula
LIMIT 10000;
