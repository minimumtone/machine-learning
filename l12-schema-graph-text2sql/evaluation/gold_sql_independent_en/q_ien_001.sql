SELECT source_db, COUNT(*) AS n_entries FROM material_entry GROUP BY source_db ORDER BY source_db;
