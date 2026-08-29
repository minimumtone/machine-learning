SELECT z.col_quebec AS formula,
       j.col_papa AS a_site,
       d.col_luna AS formation_energy
FROM tbl_zulu z
JOIN tbl_juliet j ON j.col_rhea = z.col_rhea
JOIN tbl_delta d ON d.col_rhea = z.col_rhea
WHERE j.col_zulu = 'A-site'
  AND j.col_papa = 'Co'
ORDER BY d.col_luna ASC
LIMIT 5;
