SELECT
  reference_id,
  doi,
  title,
  authors,
  journal,
  year,
  volume,
  pages
FROM literature_reference
WHERE doi IS NOT NULL
  AND doi <> ''
ORDER BY year DESC, reference_id;
