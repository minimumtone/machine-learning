SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  mp.curie_temperature_k
FROM magnetic_property mp
JOIN material_entry me
  ON mp.entry_id = me.entry_id
WHERE mp.curie_temperature_k IS NOT NULL
  AND me.number_of_elements > 1
ORDER BY mp.curie_temperature_k DESC
LIMIT 1;
