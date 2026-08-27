SELECT m.entry_id, m.formula, se.miller_index, se.is_reconstructed
FROM material_entry m
JOIN surface_energy se ON se.entry_id = m.entry_id
WHERE se.is_reconstructed = TRUE
ORDER BY m.formula, m.entry_id, se.miller_index
LIMIT 10000;
