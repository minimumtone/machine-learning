SELECT DISTINCT
    m.formula,
    s.prototype,
    s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_reference mr1 ON mr1.entry_id = m.entry_id
JOIN material_reference mr2 ON mr2.entry_id = m.entry_id
JOIN material_reference mr3 ON mr3.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND mr1.reference_id < mr2.reference_id
  AND mr2.reference_id < mr3.reference_id
ORDER BY m.formula
LIMIT 10000;
