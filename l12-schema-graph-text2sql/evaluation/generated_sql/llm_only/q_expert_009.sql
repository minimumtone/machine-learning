SELECT DISTINCT formula
FROM material_entry
WHERE chemical_system ~ '(^|-)Ti(-|$)';
