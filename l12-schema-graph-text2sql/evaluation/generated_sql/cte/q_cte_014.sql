SELECT
  l12_elements.element_symbol,
  ROUND(
    (
      COUNT(DISTINCT l12_elements.entry_id) FILTER (
        WHERE l12_elements.formation_energy_per_atom < 0
      ) * 100.0 / COUNT(DISTINCT l12_elements.entry_id)
    )::numeric,
    2
  ) AS negative_formation_energy_percentage
FROM (
  SELECT
    m.entry_id,
    (regexp_matches(m.formula, '([A-Z][a-z]?)', 'g'))[1] AS element_symbol,
    ps.formation_energy_per_atom
  FROM material_entry m
  JOIN structure s ON s.entry_id = m.entry_id
  JOIN phase_stability ps ON ps.entry_id = m.entry_id
  WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
) AS l12_elements
GROUP BY l12_elements.element_symbol
ORDER BY l12_elements.element_symbol
LIMIT 10000;
