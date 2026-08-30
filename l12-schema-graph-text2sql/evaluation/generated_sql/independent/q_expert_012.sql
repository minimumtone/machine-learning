SELECT DISTINCT
    m.entry_id, m.formula
FROM material_entry m
WHERE
    EXISTS (SELECT 1 FROM composition c_al WHERE c_al.entry_id = m.entry_id AND c_al.element = 'Al')
    AND EXISTS (SELECT 1 FROM composition c_rh WHERE c_rh.entry_id = m.entry_id AND c_rh.element = 'Rh')

LIMIT 10000;
