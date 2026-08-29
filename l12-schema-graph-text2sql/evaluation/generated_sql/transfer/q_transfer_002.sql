SELECT composition_formula, prototype_label
FROM oqmd_entries
WHERE prototype_label ILIKE '%L12%'
   OR prototype_label ILIKE '%L1₂%'
   OR prototype_label ILIKE '%L1\_2%' ESCAPE '\'
ORDER BY composition_formula, prototype_label
LIMIT 10000;
