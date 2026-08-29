SELECT
    me.entry_id,
    me.formula,
    me.chemical_system,
    COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
    s.prototype,
    ps.reference_set,
    ps.formation_energy_per_atom,
    ps.energy_above_hull,
    CASE
        WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
        WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
        ELSE 'unstable'
    END AS stability_class,
    ps.is_stable AS phase_is_stable,
    et.bulk_modulus_vrh,
    et.shear_modulus_vrh,
    et.youngs_modulus,
    et.poisson_ratio,
    et.is_stable AS elastic_is_stable
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE EXISTS (
    SELECT 1
    FROM composition AS comp
    WHERE comp.entry_id = me.entry_id
      AND comp.element = 'Fe'
)
AND (
    COALESCE(s.strukturbericht, pd.strukturbericht) IN ('L1₂', 'L1_2', 'L12')
    OR pd.prototype_name ILIKE '%L1₂%'
    OR pd.prototype_name ILIKE '%L1_2%'
    OR pd.prototype_name ILIKE '%L12%'
)
ORDER BY ps.energy_above_hull ASC, me.entry_id;
