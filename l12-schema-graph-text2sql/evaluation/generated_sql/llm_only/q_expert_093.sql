WITH a_site_counts AS (
  SELECT
    c.element AS a_site_element,
    COUNT(DISTINCT c.entry_id) AS l12_compound_count
  FROM composition c
  JOIN structure s
    ON s.entry_id = c.entry_id
  WHERE c.site_label = 'A-site'
    AND upper(replace(coalesce(s.strukturbericht, ''), '_', '')) = 'L12'
  GROUP BY c.element
),
ranked AS (
  SELECT
    a_site_element,
    l12_compound_count,
    RANK() OVER (ORDER BY l12_compound_count DESC) AS rnk
  FROM a_site_counts
)
SELECT
  a_site_element,
  l12_compound_count
FROM ranked
WHERE rnk = 1
ORDER BY a_site_element;
