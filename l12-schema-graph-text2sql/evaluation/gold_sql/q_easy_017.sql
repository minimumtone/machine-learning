SELECT DISTINCT c.element, c.site_label
FROM composition c
WHERE c.site_label = 'A'
ORDER BY c.element
LIMIT 10000;