-- easy: 格子定数4Å未満
SELECT entry_key, composition_formula, lattice_param_a
FROM oqmd_entries
WHERE lattice_param_a < 4.0
LIMIT 10000;
