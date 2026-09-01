SELECT crystal_system, AVG(band_gap) FROM mp_entries GROUP BY crystal_system ORDER BY AVG(band_gap) DESC NULLS LAST, crystal_system ASC;
