-- very_hard: CTE 凸包上L12化合物の補正生成エンタルピー -0.3未満
WITH enthalpy AS (
    SELECT e.entry_key, e.composition_formula, f.delta_e,
           SUM(r.atomic_ratio * rs.reference_delta_e) AS weighted_ref
    FROM oqmd_entries e
    JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
    JOIN oqmd_element_ratios r ON r.entry_key = e.entry_key
    JOIN oqmd_reference_states rs ON rs.symbol = r.symbol
    WHERE e.prototype_label = 'L12' AND f.on_hull = true
    GROUP BY e.entry_key, e.composition_formula, f.delta_e
)
SELECT composition_formula,
       ROUND((delta_e - weighted_ref)::numeric, 4) AS enthalpy_vs_element_ground_states
FROM enthalpy
WHERE delta_e - weighted_ref < -0.3
ORDER BY enthalpy_vs_element_ground_states ASC;
