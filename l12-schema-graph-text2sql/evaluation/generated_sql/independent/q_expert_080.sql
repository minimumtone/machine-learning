SELECT lr.title, lr.authors, lr.journal, lr.year, lr.doi
FROM literature_reference lr
WHERE lr.doi IS NOT NULL
  AND lr.doi <> ''
ORDER BY lr.year DESC, lr.title
LIMIT 10000;
