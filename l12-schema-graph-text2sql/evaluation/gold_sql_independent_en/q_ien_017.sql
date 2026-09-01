SELECT sigma_value, COUNT(*) AS n_boundaries, AVG(gb_energy_j_m2) AS avg_gb_energy FROM grain_boundary GROUP BY sigma_value ORDER BY sigma_value;
