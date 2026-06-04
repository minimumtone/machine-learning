SELECT DISTINCT m.entry_id, m.formula, ad.domain_name
FROM material_entry m
JOIN material_application ma ON ma.entry_id = m.entry_id
JOIN application_domain ad ON ad.domain_id = ma.domain_id
WHERE ad.domain_name LIKE '%Heat%' OR ad.domain_name LIKE '%Thermal%' OR ad.domain_name LIKE '%High-Temperature%'
ORDER BY m.formula
LIMIT 10000;
