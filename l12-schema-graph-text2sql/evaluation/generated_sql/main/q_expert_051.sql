SELECT m.formula, dos.total_dos_at_fermi
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN density_of_states dos ON dos.calculation_id = calc.calculation_id
WHERE s.prototype = 'L12'
   OR s.strukturbericht = 'L12'
ORDER BY dos.total_dos_at_fermi DESC
LIMIT 1;
