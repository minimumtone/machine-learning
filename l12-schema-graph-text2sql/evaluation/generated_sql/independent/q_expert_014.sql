SELECT DISTINCT m.formula
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND EXISTS (
    SELECT 1
    FROM composition c
    WHERE c.entry_id = m.entry_id
  )
  AND NOT EXISTS (
    SELECT 1
    FROM composition c
    WHERE c.entry_id = m.entry_id
      AND c.element NOT IN (
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
      )
  )
ORDER BY m.formula ASC
LIMIT 10000;
