-- easy: L12プロトタイプ一覧
SELECT entry_key, composition_formula
FROM oqmd_entries
WHERE prototype_label = 'L12'
LIMIT 10000;
