SELECT DISTINCT chemical_system
FROM material_entry
WHERE number_of_elements > 1
ORDER BY chemical_system;
