SELECT DISTINCT ON (chemsys) chemsys, formula, energy_above_hull
FROM mp_entries
ORDER BY chemsys, energy_above_hull ASC NULLS LAST, formula
LIMIT 10000;
