SELECT
  me.entry_id,
  me.formula,
  gb.gb_energy_j_m2
FROM material_entry AS me
JOIN grain_boundary AS gb
  ON gb.entry_id = me.entry_id
WHERE gb.sigma_value = 5
  AND gb.gb_energy_j_m2 IS NOT NULL;
