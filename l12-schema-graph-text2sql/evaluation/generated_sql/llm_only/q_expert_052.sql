SELECT EXISTS (
  SELECT 1
  FROM material_entry me
  JOIN calculation c ON c.entry_id = me.entry_id
  JOIN band_structure bs ON bs.calculation_id = c.calculation_id
  WHERE me.number_of_elements > 1
    AND bs.is_direct_gap = TRUE
    AND (bs.cbm_energy - bs.vbm_energy) > 0
) AS has_direct_band_gap_compounds;
