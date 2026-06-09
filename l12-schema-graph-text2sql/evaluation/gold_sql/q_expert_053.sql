SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN density_of_states dos ON dos.entry_id = m.entry_id
WHERE dos.spin_polarized = TRUE
ORDER BY m.formula
LIMIT 10000;
