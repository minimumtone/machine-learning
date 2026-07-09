-- medium: 結晶系ごとの平均形成エネルギー
SELECT e.crystal_system,
       COUNT(*) AS n,
       ROUND(AVG(f.delta_e)::numeric, 4) AS avg_delta_e
FROM oqmd_entries e
JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
WHERE e.crystal_system IS NOT NULL
GROUP BY e.crystal_system
ORDER BY avg_delta_e ASC;
