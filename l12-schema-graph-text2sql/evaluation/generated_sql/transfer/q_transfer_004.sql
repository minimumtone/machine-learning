SELECT entry_key, composition_formula, lattice_param_a
FROM oqmd_entries
WHERE lattice_param_a < 4
ORDER BY lattice_param_a ASC
LIMIT 10000;
