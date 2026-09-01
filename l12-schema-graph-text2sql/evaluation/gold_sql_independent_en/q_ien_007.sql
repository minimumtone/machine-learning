SELECT crystal_system, COUNT(*) AS n_structures FROM structure GROUP BY crystal_system ORDER BY n_structures DESC, crystal_system;
