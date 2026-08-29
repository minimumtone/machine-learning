SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.prototype,
    s.strukturbericht,
    ps.energy_above_hull,
    ps.is_stable
FROM material_entry AS me
JOIN composition AS c
    ON c.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
WHERE c.element = 'Co'
  AND s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
ORDER BY ps.energy_above_hull ASC NULLS LAST, me.formula;
