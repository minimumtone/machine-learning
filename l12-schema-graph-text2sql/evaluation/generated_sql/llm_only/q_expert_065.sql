SELECT
  me.entry_id,
  me.formula,
  md.formation_energy AS vacancy_formation_energy
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN material_defect AS md
  ON md.entry_id = me.entry_id
JOIN defect_type AS dt
  ON dt.defect_type_id = md.defect_type_id
WHERE
  (
    REPLACE(UPPER(COALESCE(s.strukturbericht, '')), '_', '') = 'L12'
    OR REPLACE(UPPER(COALESCE(pd.strukturbericht, '')), '_', '') = 'L12'
    OR REPLACE(UPPER(COALESCE(s.prototype, '')), '_', '') = 'L12'
    OR REPLACE(UPPER(COALESCE(pd.prototype_name, '')), '_', '') LIKE '%L12%'
  )
  AND (
    LOWER(dt.defect_name) LIKE '%vacanc%'
    OR LOWER(dt.category) LIKE '%vacanc%'
  )
  AND md.formation_energy IS NOT NULL
ORDER BY md.formation_energy ASC
FETCH FIRST 1 ROW WITH TIES;
