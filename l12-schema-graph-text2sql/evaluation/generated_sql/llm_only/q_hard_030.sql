SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.prototype,
    s.strukturbericht
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE EXISTS (
    SELECT 1
    FROM phase_stability AS ps
    WHERE ps.entry_id = me.entry_id
      AND ps.is_stable = TRUE
)
AND EXISTS (
    SELECT 1
    FROM composition AS c
    JOIN element AS e
        ON e.symbol = c.element
    WHERE c.entry_id = me.entry_id
      AND e.category = 'transition_metal'
)
AND (
    s.strukturbericht IN ('L1_2', 'L1₂')
    OR pd.strukturbericht IN ('L1_2', 'L1₂')
    OR s.prototype ILIKE '%L1_2%'
    OR s.prototype ILIKE '%L1₂%'
    OR pd.prototype_name ILIKE '%L1_2%'
    OR pd.prototype_name ILIKE '%L1₂%'
)
ORDER BY me.reduced_formula, me.entry_id;
