SELECT EXISTS (
    SELECT 1
    FROM material_defect md
    JOIN element e ON md.dopant_element_id = e.element_id
    WHERE e.symbol = 'B'
) AS has_b_dopant_compounds;
