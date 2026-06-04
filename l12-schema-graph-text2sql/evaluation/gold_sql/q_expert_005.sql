SELECT COUNT(DISTINCT entry_id) AS nacl_count FROM structure WHERE prototype = 'NaCl' OR strukturbericht = 'B1' LIMIT 10000;
