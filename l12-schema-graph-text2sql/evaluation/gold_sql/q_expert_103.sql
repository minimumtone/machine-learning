-- VH: L1₂型化合物をA-site元素とB-site元素で分類し、各組合せの平均バルクモジュラスと安定相の数を集計して
-- Tables: material_entry, composition(×2 self-join), structure, phase_stability, calculation, calculated_property (6 distinct)
SELECT ca.element AS a_site, cb.element AS b_site,
       AVG(cp_bm.value) AS avg_bulk_modulus,
       COUNT(*) AS total_count,
       SUM(CASE WHEN ps.energy_above_hull <= 0.001 THEN 1 ELSE 0 END) AS stable_count
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A-site'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B-site'
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp_bm.property_name = 'bulk_modulus'
GROUP BY ca.element, cb.element
ORDER BY avg_bulk_modulus DESC
LIMIT 10000;
