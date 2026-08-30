SELECT COUNT(*) AS metastable_compound_count
FROM tbl_delta
WHERE col_hotel = false
  AND col_xenon < 0.05
LIMIT 10000;
