-- medium: hull距離0.05未満の準安定化合物数
SELECT COUNT(*) AS n_near_hull
FROM tbl_delta
WHERE col_xenon < 0.05 AND col_hotel = false;
