SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    ps.energy_above_hull,
    ps.band_gap,
    s.strukturbericht
FROM material_entry AS me
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
WHERE ps.is_stable = TRUE
  AND ps.band_gap = 0
  AND REPLACE(UPPER(s.strukturbericht), '_', '') = 'L12';
