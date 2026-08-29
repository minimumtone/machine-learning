SELECT
    m.formula,
    s.lattice_a,
    ps.formation_energy_per_atom,
    cp.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE
    (s.prototype = 'L12' OR s.strukturbericht = 'L12')
    AND cp.property_name = 'bulk_modulus'
    AND ps.formation_energy_per_atom < (
        SELECT ps.formation_energy_per_atom
        FROM material_entry m
        JOIN phase_stability ps ON ps.entry_id = m.entry_id
        JOIN structure s ON s.entry_id = m.entry_id
        WHERE
            m.formula = 'Ni3Al'
            AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
        ORDER BY ps.formation_energy_per_atom ASC
        LIMIT 1
    )
    AND cp.value > (
        SELECT cp.value
        FROM material_entry m
        JOIN structure s ON s.entry_id = m.entry_id
        JOIN calculation calc ON calc.entry_id = m.entry_id
        JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
        WHERE
            m.formula = 'Ni3Al'
            AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
            AND cp.property_name = 'bulk_modulus'
        ORDER BY cp.value DESC
        LIMIT 1
    )
ORDER BY ps.formation_energy_per_atom ASC, cp.value DESC
LIMIT 10000;
