SELECT formula
FROM mp_entries
WHERE chemsys = 'Co-Ti'
ORDER BY band_gap DESC NULLS LAST
LIMIT 1;
