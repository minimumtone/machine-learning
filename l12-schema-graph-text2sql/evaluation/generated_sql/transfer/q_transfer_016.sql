SELECT
  oqmd_reference_states.symbol,
  oqmd_elements.element_name,
  oqmd_reference_states.polymorph_count,
  oqmd_reference_states.gs_spacegroup
FROM oqmd_reference_states
JOIN oqmd_elements
  ON oqmd_elements.symbol = oqmd_reference_states.symbol
ORDER BY oqmd_reference_states.polymorph_count DESC
LIMIT 5;
