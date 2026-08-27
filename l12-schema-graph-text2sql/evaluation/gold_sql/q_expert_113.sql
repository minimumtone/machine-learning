-- VH: 融点が2000K以上の元素を、その元素を含む安定化合物の件数とともに一覧して
-- Tables: element, element_property, composition, material_entry, phase_stability (5)
-- Exercises the element_property dictionary (element-level measured data).
SELECT e.symbol, ep.value AS melting_point_k,
       COUNT(DISTINCT m.entry_id) FILTER (WHERE ps.is_stable) AS stable_compound_count
FROM element e
JOIN element_property ep ON ep.element_id = e.element_id
LEFT JOIN composition c ON c.element = e.symbol
LEFT JOIN material_entry m ON m.entry_id = c.entry_id
LEFT JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE ep.property_name = 'melting_point'
  AND ep.value >= 2000
GROUP BY e.symbol, ep.value
ORDER BY ep.value DESC, e.symbol
LIMIT 10000;
