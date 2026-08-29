SELECT
    m.formula,
    dt.defect_name,
    dt.category,
    md.site,
    md.concentration,
    md.formation_energy
FROM material_entry m
JOIN material_defect md ON md.entry_id = m.entry_id
JOIN defect_type dt ON dt.defect_type_id = md.defect_type_id
WHERE dt.category = 'interstitial'
   OR dt.defect_name ILIKE '%interstitial%'
ORDER BY m.formula, dt.defect_name
LIMIT 10000;
