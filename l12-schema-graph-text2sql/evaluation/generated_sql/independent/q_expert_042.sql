SELECT DISTINCT
    m.entry_id, m.formula, s.prototype, s.lattice_a, s.space_group, cp.property_name, cp.value, cp.unit
FROM material_entry m
    JOIN calculation calc ON calc.entry_id = m.entry_id
    JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
    JOIN structure s ON s.entry_id = m.entry_id
WHERE
    (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    AND cp.value >= 200.0
    AND cp.property_name = 'bulk_modulus'

LIMIT 10000;
