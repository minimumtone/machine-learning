SELECT sm.method_name, COUNT(*) AS n_success FROM material_synthesis ms JOIN synthesis_method sm ON sm.synthesis_id = ms.synthesis_id WHERE ms.success GROUP BY sm.method_name ORDER BY sm.method_name;
