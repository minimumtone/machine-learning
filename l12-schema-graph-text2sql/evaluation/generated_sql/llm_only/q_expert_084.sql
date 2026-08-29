SELECT DISTINCT
    me.entry_id,
    me.formula,
    s.lattice_a,
    et.shear_modulus_vrh,
    ps.energy_above_hull
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE
    ps.is_stable = TRUE
    AND s.lattice_a BETWEEN 3.56 - 0.03 AND 3.56 + 0.03
    AND et.shear_modulus_vrh >= 70
    AND (
        LOWER(REPLACE(COALESCE(s.strukturbericht, ''), '_', '')) = 'l12'
        OR LOWER(REPLACE(COALESCE(pd.strukturbericht, ''), '_', '')) = 'l12'
        OR pd.prototype_name ILIKE '%L12%'
    )
ORDER BY
    et.shear_modulus_vrh DESC;
