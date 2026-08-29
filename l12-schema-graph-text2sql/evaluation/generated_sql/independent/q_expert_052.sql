SELECT m.formula, ps.band_gap, bs.is_direct_gap
FROM material_entry m
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN band_structure bs ON bs.calculation_id = calc.calculation_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE bs.is_direct_gap = TRUE
ORDER BY ps.band_gap DESC
LIMIT 10000;
