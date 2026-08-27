SELECT m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       CASE WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
            WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
            ELSE 'unstable' END AS stability_class
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY ps.energy_above_hull ASC, m.entry_id ASC
LIMIT 10000;