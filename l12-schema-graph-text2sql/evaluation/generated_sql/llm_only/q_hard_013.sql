SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.chemical_system,
    et.bulk_modulus_vrh AS bulk_modulus_gpa,
    ps.formation_energy_per_atom AS formation_energy_ev_per_atom,
    ps.energy_above_hull,
    ps.reference_set
FROM material_entry AS me
JOIN composition AS comp
    ON comp.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN calculation AS calc
    ON calc.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = calc.calculation_id
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
WHERE comp.element = 'Ni'
  AND REPLACE(REPLACE(UPPER(COALESCE(s.strukturbericht, '')), '_', ''), '₂', '2') = 'L12'
  AND et.bulk_modulus_vrh IS NOT NULL
  AND ps.formation_energy_per_atom IS NOT NULL
ORDER BY ps.formation_energy_per_atom, et.bulk_modulus_vrh;
