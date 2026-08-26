SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, ps.formation_energy_per_atom
FROM material_entry m
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE m.formula = 'Ni3Al'
LIMIT 10000;
