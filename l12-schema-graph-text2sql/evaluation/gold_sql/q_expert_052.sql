SELECT m.entry_id, m.formula, bs.band_gap_type, bs.cbm_energy, bs.vbm_energy
FROM material_entry m
JOIN calculation cal_bs ON cal_bs.entry_id = m.entry_id AND cal_bs.calculation_type = 'relaxation'
JOIN band_structure bs ON bs.calculation_id = cal_bs.calculation_id
WHERE bs.is_direct_gap = TRUE
ORDER BY m.formula, m.entry_id
LIMIT 10000;
