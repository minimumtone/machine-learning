SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    s.strukturbericht,
    dt.defect_name,
    dt.category AS defect_category,
    md.formation_energy,
    md.concentration,
    md.site,
    e.symbol AS dopant_element
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN material_defect AS md
    ON md.entry_id = me.entry_id
JOIN defect_type AS dt
    ON dt.defect_type_id = md.defect_type_id
LEFT JOIN element AS e
    ON e.element_id = md.dopant_element_id
WHERE (
        regexp_replace(upper(COALESCE(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
        OR regexp_replace(upper(COALESCE(s.prototype, '')), '[^A-Z0-9]', '', 'g') = 'L12'
      )
  AND (
        dt.defect_name ILIKE '%antisite%'
        OR dt.category ILIKE '%antisite%'
      )
ORDER BY
    me.chemical_system,
    me.reduced_formula,
    me.entry_id;
