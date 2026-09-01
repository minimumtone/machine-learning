SELECT DISTINCT c.element, c.site_label
FROM composition c
WHERE c.site_label = 'A-site'
ORDER BY c.element
LIMIT 10000;
