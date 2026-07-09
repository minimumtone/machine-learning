-- very_hard: CTE 生成エンタルピー計算（Co3Ti相当）
WITH target AS (
    SELECT e.entry_key, e.composition_formula, f.delta_e
    FROM oqmd_entries e
    JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
    WHERE e.composition_formula = 'Co3Ti'
    LIMIT 1
),
ref AS (
    SELECT t.entry_key,
           SUM(r.atomic_ratio * rs.energy_pa) AS weighted_ref
    FROM target t
    JOIN oqmd_element_ratios r ON r.entry_key = t.entry_key
    JOIN oqmd_reference_states rs ON rs.symbol = r.symbol
    GROUP BY t.entry_key
)
SELECT t.composition_formula, t.delta_e, ref.weighted_ref,
       t.delta_e - ref.weighted_ref AS corrected_enthalpy
FROM target t
JOIN ref ON ref.entry_key = t.entry_key;
