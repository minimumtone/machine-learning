SELECT EXISTS (
    SELECT 1
    FROM material_synthesis ms
    JOIN synthesis_method sm ON ms.synthesis_id = sm.synthesis_id
    JOIN material_entry me ON ms.entry_id = me.entry_id
    WHERE sm.method_name ILIKE '%ball%milling%'
      AND ms.success = TRUE
      AND me.number_of_elements > 1
) AS has_ball_milling_synthesized_compounds;
