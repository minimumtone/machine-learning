SELECT DISTINCT
    m.entry_id, m.formula
FROM material_entry m
WHERE
    EXISTS (SELECT 1 FROM composition c_ni WHERE c_ni.entry_id = m.entry_id AND c_ni.element = 'Ni')
    AND EXISTS (SELECT 1 FROM composition c_al WHERE c_al.entry_id = m.entry_id AND c_al.element = 'Al')

LIMIT 10000;
