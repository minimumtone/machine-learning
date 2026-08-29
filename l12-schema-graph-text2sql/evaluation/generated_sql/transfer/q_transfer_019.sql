WITH corrected_rows AS (
    SELECT
        e.entry_key,
        fe.fe_key,
        e.composition_formula,
        fe.delta_e,
        fe.delta_e - (
            SUM(er.atomic_ratio * rs.reference_delta_e) OVER (PARTITION BY e.entry_key, fe.fe_key)
            / NULLIF(SUM(er.atomic_ratio) OVER (PARTITION BY e.entry_key, fe.fe_key), 0)
        ) AS rebased_delta_e
    FROM oqmd_entries e
    JOIN oqmd_formation_energies fe ON fe.entry_key = e.entry_key
    JOIN oqmd_element_ratios er ON er.entry_key = e.entry_key
    JOIN oqmd_elements el ON el.symbol = er.symbol
    JOIN oqmd_reference_states rs ON rs.symbol = el.symbol
    WHERE fe.on_hull = TRUE
      AND e.prototype_label IN ('L12', 'L1_2', 'L1₂')
),
per_entry AS (
    SELECT DISTINCT ON (entry_key, fe_key)
        entry_key,
        fe_key,
        composition_formula,
        delta_e,
        rebased_delta_e
    FROM corrected_rows
    WHERE rebased_delta_e < -0.3
    ORDER BY entry_key, fe_key
)
SELECT
    composition_formula,
    delta_e,
    rebased_delta_e
FROM per_entry
ORDER BY rebased_delta_e ASC
LIMIT 10000;
