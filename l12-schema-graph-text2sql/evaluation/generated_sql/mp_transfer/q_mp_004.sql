SELECT COUNT(*) AS stable_entry_count
FROM mp_entries
WHERE energy_above_hull = 0
LIMIT 10000;
