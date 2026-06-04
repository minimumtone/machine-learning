SELECT s.prototype, AVG(ps.energy_above_hull) AS avg_ehull, COUNT(*) AS cnt
FROM structure s
JOIN phase_stability ps ON ps.entry_id = s.entry_id
GROUP BY s.prototype
ORDER BY avg_ehull
LIMIT 10000;
