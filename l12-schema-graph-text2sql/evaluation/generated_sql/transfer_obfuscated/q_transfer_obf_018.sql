WITH ref_energies AS (
    SELECT
        z.col_rhea,
        SUM(j.col_juliet * v.col_tango) AS weighted_ref_energy
    FROM tbl_zulu z
    JOIN tbl_juliet j ON j.col_rhea = z.col_rhea
    JOIN tbl_xray x ON j.col_papa = x.col_papa
    JOIN tbl_victor v ON v.col_papa = x.col_papa
    WHERE z.col_quebec = 'Co3Ti'
    GROUP BY z.col_rhea
)
SELECT
    z.col_quebec,
    d.col_luna,
    r.weighted_ref_energy,
    d.col_luna - r.weighted_ref_energy AS formation_enthalpy_from_pure_elements
FROM tbl_zulu z
JOIN tbl_delta d ON d.col_rhea = z.col_rhea
JOIN ref_energies r ON r.col_rhea = z.col_rhea
WHERE z.col_quebec = 'Co3Ti'
ORDER BY formation_enthalpy_from_pure_elements ASC
LIMIT 10000;
