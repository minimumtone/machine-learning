SELECT crystal_system, COUNT(*) FROM mp_entries GROUP BY crystal_system ORDER BY COUNT(*) DESC, crystal_system ASC;
