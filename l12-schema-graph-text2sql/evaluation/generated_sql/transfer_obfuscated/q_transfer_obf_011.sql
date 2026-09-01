SELECT
  tbl_zulu.col_halo AS crystal_system,
  AVG(tbl_delta.col_luna) AS avg_eform
FROM tbl_zulu
JOIN tbl_delta ON tbl_delta.col_rhea = tbl_zulu.col_rhea
GROUP BY tbl_zulu.col_halo
ORDER BY avg_formation_energy ASC
LIMIT 10000;
