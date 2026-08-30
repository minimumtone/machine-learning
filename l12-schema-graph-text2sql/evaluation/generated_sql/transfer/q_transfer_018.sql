WITH ref_energy AS (
    SELECT
        oqmd_entries.entry_key,
        oqmd_entries.composition_formula,
        oqmd_formation_energies.delta_e,
        SUM(oqmd_element_ratios.atomic_ratio * oqmd_reference_states.reference_delta_e) AS weighted_ref_energy
    FROM oqmd_entries
    JOIN oqmd_formation_energies
        ON oqmd_formation_energies.entry_key = oqmd_entries.entry_key
    JOIN oqmd_element_ratios
        ON oqmd_element_ratios.entry_key = oqmd_entries.entry_key
    JOIN oqmd_elements
        ON oqmd_elements.symbol = oqmd_element_ratios.symbol
    JOIN oqmd_reference_states
        ON oqmd_reference_states.symbol = oqmd_elements.symbol
    WHERE oqmd_entries.composition_formula = 'Co3Ti'
    GROUP BY
        oqmd_entries.entry_key,
        oqmd_entries.composition_formula,
        oqmd_formation_energies.delta_e
)
SELECT
    composition_formula,
    delta_e,
    weighted_ref_energy,
    delta_e - weighted_ref_energy AS formation_enthalpy_from_pure_reference
FROM ref_energy
ORDER BY formation_enthalpy_from_pure_reference ASC
LIMIT 10000;
