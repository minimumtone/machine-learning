SELECT m.formula, gb.sigma_value, gb.gb_energy_j_m2
FROM material_entry m
JOIN grain_boundary gb ON gb.entry_id = m.entry_id
WHERE gb.sigma_value = 5
  AND gb.gb_energy_j_m2 IS NOT NULL
ORDER BY m.formula
LIMIT 10000;
