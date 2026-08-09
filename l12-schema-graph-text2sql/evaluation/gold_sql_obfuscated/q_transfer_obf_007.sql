-- medium: 凸包上の化合物数
SELECT COUNT(*) AS n_on_hull
FROM tbl_delta
WHERE col_hotel = true;
