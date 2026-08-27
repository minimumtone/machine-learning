SELECT m.formula, ps.energy_above_hull, ps.is_stable,
       cp.property_name, cp.value
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE c.element = 'Fe'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula, cp.property_name
LIMIT 10000;