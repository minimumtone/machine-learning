WITH top_pure_elements AS (
  SELECT
    m.chemical_system AS element_symbol,
    COUNT(DISTINCT s.structure_id) AS polymorph_count
  FROM material_entry m
  JOIN structure s ON s.entry_id = m.entry_id
  WHERE m.number_of_elements = 1
  GROUP BY m.chemical_system
  ORDER BY COUNT(DISTINCT s.structure_id) DESC
  LIMIT 5
)
SELECT
  t.element_symbol,
  t.polymorph_count,
  COUNT(DISTINCT m.entry_id) AS l12_compound_count,
  AVG(ps.formation_energy_per_atom) AS avg_eform
FROM top_pure_elements t
JOIN material_entry m
  ON (
    m.chemical_system = t.element_symbol
    OR m.chemical_system LIKE t.element_symbol || '-%'
    OR m.chemical_system LIKE '%-' || t.element_symbol || '-%'
    OR m.chemical_system LIKE '%-' || t.element_symbol
  )
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
GROUP BY t.element_symbol, t.polymorph_count
ORDER BY t.polymorph_count DESC
LIMIT 10000;
