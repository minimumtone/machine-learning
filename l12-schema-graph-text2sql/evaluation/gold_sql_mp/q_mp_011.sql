SELECT DISTINCT ON (chemsys) chemsys, formula, energy_above_hull, entry_id FROM mp_entries ORDER BY chemsys, energy_above_hull ASC, formula, entry_id;
