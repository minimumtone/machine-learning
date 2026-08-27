SELECT m.entry_id, m.formula, c.element
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element IN ('La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Sc','Y')
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;
