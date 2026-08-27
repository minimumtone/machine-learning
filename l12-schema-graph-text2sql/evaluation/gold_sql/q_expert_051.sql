SELECT m.entry_id, m.formula, dos.total_dos_at_fermi
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation cal_dos ON cal_dos.entry_id = m.entry_id AND cal_dos.calculation_type = 'relaxation'
JOIN density_of_states dos ON dos.calculation_id = cal_dos.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY dos.total_dos_at_fermi DESC
LIMIT 1;
