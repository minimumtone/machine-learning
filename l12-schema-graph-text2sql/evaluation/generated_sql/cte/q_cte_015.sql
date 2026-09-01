SELECT c.element AS a_site,
       AVG(ps.formation_energy_per_atom) AS avg_eform
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'A-site'
  AND ps.is_stable = TRUE
  AND c.element IN (
      SELECT cp.element
      FROM material_entry mp
      JOIN composition cp ON cp.entry_id = mp.entry_id
      JOIN structure sp ON sp.entry_id = mp.entry_id
      JOIN phase_stability psp ON psp.entry_id = mp.entry_id
      WHERE mp.number_of_elements = 1
        AND cp.atomic_fraction = 1
        AND sp.volume_per_atom >= 15
        AND psp.is_stable = TRUE
  )
GROUP BY c.element
ORDER BY c.element ASC
LIMIT 10000;
