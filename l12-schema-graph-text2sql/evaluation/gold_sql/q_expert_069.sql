SELECT m.entry_id, m.formula, se100.surface_energy_j_m2 AS se_100, se110.surface_energy_j_m2 AS se_110
FROM material_entry m
JOIN surface_energy se100 ON se100.entry_id = m.entry_id AND se100.miller_index = '100'
JOIN surface_energy se110 ON se110.entry_id = m.entry_id AND se110.miller_index = '110'
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;
