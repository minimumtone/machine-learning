SELECT m.formula,
       s.lattice_a,
       cp.value AS shear_modulus,
       ps.energy_above_hull,
       ps.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.lattice_a BETWEEN 3.53 AND 3.59
  AND cp.property_name = 'shear_modulus'
  AND cp.unit = 'GPa'
  AND cp.value >= 70
  AND ps.is_stable = TRUE
ORDER BY s.lattice_a ASC, cp.value DESC
LIMIT 10000;
