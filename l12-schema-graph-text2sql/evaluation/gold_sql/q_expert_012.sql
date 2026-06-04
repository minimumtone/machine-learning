SELECT DISTINCT m.entry_id, m.formula, s.prototype, s.crystal_system
FROM material_entry m
JOIN composition c_rh ON c_rh.entry_id = m.entry_id AND c_rh.element = 'Rh'
JOIN composition c_al ON c_al.entry_id = m.entry_id AND c_al.element = 'Al'
JOIN structure s ON s.entry_id = m.entry_id
ORDER BY m.formula
LIMIT 10000;
