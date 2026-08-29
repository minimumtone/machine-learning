SELECT DISTINCT me.*
FROM material_entry AS me
JOIN composition AS c
  ON me.entry_id = c.entry_id
WHERE c.element = 'Al'
  AND me.number_of_elements > 1;
