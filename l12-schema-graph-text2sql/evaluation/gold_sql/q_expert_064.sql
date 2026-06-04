SELECT m.entry_id, m.formula, gb.gb_energy_j_m2, gb.rotation_axis
FROM material_entry m
JOIN grain_boundary gb ON gb.entry_id = m.entry_id
WHERE gb.sigma_value = 5
ORDER BY gb.gb_energy_j_m2
LIMIT 10000;
