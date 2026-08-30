SELECT DISTINCT
    m.entry_id, m.formula, cp.property_name, cp.value, cp.unit
FROM material_entry m
    JOIN calculation calc ON calc.entry_id = m.entry_id
    JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE
    cp.property_name = 'youngs_modulus'

LIMIT 10000;
