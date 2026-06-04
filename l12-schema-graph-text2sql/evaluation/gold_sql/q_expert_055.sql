SELECT m.entry_id, m.formula, mp.curie_temperature_k
FROM material_entry m
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
WHERE mp.curie_temperature_k IS NOT NULL
ORDER BY mp.curie_temperature_k DESC
LIMIT 1;
