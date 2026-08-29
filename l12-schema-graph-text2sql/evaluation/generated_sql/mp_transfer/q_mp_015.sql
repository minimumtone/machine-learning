SELECT crystal_system, AVG(band_gap) AS average_band_gap
FROM mp_entries
GROUP BY crystal_system
ORDER BY AVG(band_gap) DESC
LIMIT 10000;
