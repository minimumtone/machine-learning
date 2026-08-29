SELECT DISTINCT
    m.entry_id, m.formula
FROM material_entry m
WHERE
    (m.formula = 'Ni3Al' OR m.reduced_formula = 'Ni3Al')

LIMIT 10000;
