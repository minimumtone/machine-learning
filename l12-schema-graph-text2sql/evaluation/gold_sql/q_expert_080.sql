SELECT reference_id, doi, title, authors, journal, year
FROM literature_reference
WHERE doi IS NOT NULL
ORDER BY year DESC, "reference_id", "doi", "title", "authors", "journal", "year" ASC
LIMIT 10000;
