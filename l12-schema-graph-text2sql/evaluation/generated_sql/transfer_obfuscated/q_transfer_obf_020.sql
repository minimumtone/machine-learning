SELECT entry_energies.a_site_element,
       AVG(entry_energies.ground_state_referenced_formation_energy) AS avg_ground_state_referenced_formation_energy
FROM (
    SELECT z.col_rhea,
           ja.col_papa AS a_site,
           d.col_luna - SUM(jc.col_juliet * v.col_tango) AS ground_state_referenced_formation_energy
    FROM tbl_zulu z
    JOIN tbl_delta d ON d.col_rhea = z.col_rhea
    JOIN tbl_juliet ja ON ja.col_rhea = z.col_rhea
    JOIN tbl_juliet jc ON jc.col_rhea = z.col_rhea
    JOIN tbl_xray x ON x.col_papa = jc.col_papa
    JOIN tbl_victor v ON v.col_papa = x.col_papa
    WHERE z.col_delta = 'L12'
      AND d.col_hotel = TRUE
      AND ja.col_zulu = 'a'
    GROUP BY z.col_rhea, ja.col_papa, d.col_luna
) entry_energies
GROUP BY entry_energies.a_site_element
ORDER BY avg_ground_state_referenced_formation_energy ASC
LIMIT 10000;
