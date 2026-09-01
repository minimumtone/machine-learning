SELECT EXISTS (
    SELECT 1
    FROM thermal_property tp
    JOIN calculation c ON tp.calculation_id = c.calculation_id
    JOIN material_entry me ON c.entry_id = me.entry_id
    WHERE tp.gruneisen_parameter >= 2
      AND me.number_of_elements > 1
) AS has_compounds;
