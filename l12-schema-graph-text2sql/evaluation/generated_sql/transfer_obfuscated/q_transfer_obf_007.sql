SELECT COUNT(*) AS convex_hull_compound_count
FROM tbl_delta
WHERE col_hotel = TRUE
LIMIT 10000;
