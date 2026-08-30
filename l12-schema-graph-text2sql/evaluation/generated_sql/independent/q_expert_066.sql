SELECT
    m.formula,
    s.prototype,
    s.strukturbericht,
    dt.defect_name,
    dt.category,
    md.site,
    md.concentration,
    md.formation_energy
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_defect md ON md.entry_id = m.entry_id
JOIN defect_type dt ON dt.defect_type_id = md.defect_type_id
WHERE
    (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    AND (dt.category ILIKE '%antisite%' OR dt.defect_name ILIKE '%antisite%')
ORDER BY
    m.formula,
    dt.defect_name,
    md.site
LIMIT 10000;
