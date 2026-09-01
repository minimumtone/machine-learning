SELECT DISTINCT
    m.entry_id, m.formula
FROM material_entry m
WHERE
    (m.formula = 'Co3Ti' OR m.reduced_formula = 'Co3Ti')

LIMIT 10000;
