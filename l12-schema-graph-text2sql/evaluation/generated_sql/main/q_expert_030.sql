SELECT s.prototype, COUNT(*) AS prototype_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype IS NOT NULL
GROUP BY s.prototype
ORDER BY prototype_count DESC, s.prototype
LIMIT 10000;
