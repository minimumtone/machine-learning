SELECT EXISTS (
  SELECT 1
  FROM material_entry
  WHERE number_of_elements >= 3
) AS has_compounds_with_three_or_more_elements;
