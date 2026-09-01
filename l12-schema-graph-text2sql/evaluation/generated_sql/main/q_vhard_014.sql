SELECT CASE
         WHEN c.element IN ('Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn') THEN '3d_transition_metals'
         WHEN c.element IN ('Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg') THEN '4d_5d_transition_metals'
       END AS a_site_element_period_group,
       COUNT(*) AS count,
       AVG(ps.energy_above_hull) AS avg_energy_above_hull,
       COUNT(*) FILTER (WHERE ps.is_stable = TRUE) * 100.0 / COUNT(*) AS stable_percentage,
       AVG(ps.formation_energy_per_atom) AS avg_eform
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'A-site'
  AND c.element IN ('Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg')
GROUP BY CASE
           WHEN c.element IN ('Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn') THEN '3d_transition_metals'
           WHEN c.element IN ('Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                              'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg') THEN '4d_5d_transition_metals'
         END
ORDER BY a_site_element_period_group
LIMIT 10000;
