SELECT entry_id, formula, number_of_elements FROM material_entry WHERE number_of_elements >= 3 ORDER BY formula LIMIT 10000;
