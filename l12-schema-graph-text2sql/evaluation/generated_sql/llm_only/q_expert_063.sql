SELECT EXISTS (
  SELECT 1
  FROM surface_energy se
  JOIN material_entry me ON me.entry_id = se.entry_id
  WHERE se.is_reconstructed = TRUE
    AND me.number_of_elements > 1
) AS has_surface_reconstructed_compounds;
