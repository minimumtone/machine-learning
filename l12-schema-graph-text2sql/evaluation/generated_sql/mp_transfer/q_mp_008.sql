SELECT COUNT(*) FILTER (WHERE band_gap > 0) * 100.0 / COUNT(*) AS band_gap_material_percentage
FROM mp_entries
LIMIT 10000;
