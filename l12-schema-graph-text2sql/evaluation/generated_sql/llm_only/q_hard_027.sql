SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    ca.element AS a_site_element,
    cb.element AS b_site_element
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN composition AS ca
    ON ca.entry_id = me.entry_id
   AND ca.site_label = 'A-site'
JOIN composition AS cb
    ON cb.entry_id = me.entry_id
   AND cb.site_label = 'B-site'
WHERE ca.element IN ('Ni', 'Co')
  AND cb.element IN ('Al', 'Ti')
  AND s.strukturbericht IN ('L1_2', 'L1₂', 'L12');
