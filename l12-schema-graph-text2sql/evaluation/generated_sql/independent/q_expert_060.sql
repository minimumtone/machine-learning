SELECT DISTINCT
    m.formula,
    s.prototype,
    s.strukturbericht,
    dos.is_metallic,
    mp.magnetic_ordering,
    mp.total_magnetization
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN magnetic_property mp ON mp.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN density_of_states dos ON dos.calculation_id = calc.calculation_id
WHERE
    (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    AND dos.is_metallic = TRUE
    AND mp.magnetic_ordering ILIKE 'ferro%'
ORDER BY m.formula
LIMIT 10000;
