WITH compound AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom, ps.reference_set
    FROM material_entry m
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    WHERE m.formula = 'Co3Ti'
    LIMIT 1
),
ref_energies AS (
    SELECT compound.entry_id,
           SUM(c.atomic_fraction * per.delta_e) AS weighted_ref
    FROM compound
    JOIN composition c ON c.entry_id = compound.entry_id
    JOIN pure_element_reference per ON per.element_symbol = c.element
        AND per.reference_set = compound.reference_set
    GROUP BY compound.entry_id
)
SELECT compound.formula,
       compound.formation_energy_per_atom,
       ref_energies.weighted_ref AS ref_energy,
       compound.formation_energy_per_atom - ref_energies.weighted_ref AS corrected_enthalpy
FROM compound
JOIN ref_energies ON ref_energies.entry_id = compound.entry_id
LIMIT 10000;
