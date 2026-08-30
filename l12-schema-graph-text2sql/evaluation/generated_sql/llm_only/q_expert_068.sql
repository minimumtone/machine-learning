SELECT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    dt.defect_name,
    dt.category AS defect_category,
    md.formation_energy,
    md.concentration,
    md.site
FROM material_entry AS me
JOIN material_defect AS md
    ON me.entry_id = md.entry_id
JOIN defect_type AS dt
    ON md.defect_type_id = dt.defect_type_id
WHERE LOWER(dt.defect_name) LIKE '%interstitial%'
   OR LOWER(dt.category) LIKE '%interstitial%'
ORDER BY me.chemical_system, me.reduced_formula, me.entry_id;
