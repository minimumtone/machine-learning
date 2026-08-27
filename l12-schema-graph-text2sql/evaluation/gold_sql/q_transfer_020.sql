-- very_hard: CTE A-site元素ごとの平均補正生成エンタルピー
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
SELECT ra.symbol AS a_site,
       COUNT(*) AS n_compounds,
       ROUND(AVG(en.delta_e - en.weighted_ref)::numeric, 4) AS avg_enthalpy
FROM enthalpy en
JOIN oqmd_element_ratios ra
    ON ra.entry_key = en.entry_key AND ra.wyckoff_site = 'A-site'
GROUP BY ra.symbol
HAVING COUNT(*) >= 2
ORDER BY avg_enthalpy ASC;
