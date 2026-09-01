WITH ni3al AS (
    SELECT AVG(s.lattice_a) AS ni3al_lattice_a
    FROM material_entry me
    JOIN structure s ON s.entry_id = me.entry_id
    JOIN composition c_ni ON c_ni.entry_id = me.entry_id
    JOIN composition c_al ON c_al.entry_id = me.entry_id
    WHERE s.strukturbericht = 'L1_2'
      AND c_ni.element = 'Ni'
      AND ABS(c_ni.atomic_fraction - 0.75) < 1e-6
      AND c_al.element = 'Al'
      AND ABS(c_al.atomic_fraction - 0.25) < 1e-6
)
SELECT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.lattice_a,
    ni3al.ni3al_lattice_a AS ni3al_lattice_a,
    ABS(s.lattice_a - ni3al.ni3al_lattice_a) AS lattice_a_difference,
    ps.energy_above_hull,
    CASE
        WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
        WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
        ELSE 'unstable'
    END AS stability_class
FROM material_entry me
JOIN structure s ON s.entry_id = me.entry_id
JOIN phase_stability ps ON ps.entry_id = me.entry_id
JOIN composition c_co ON c_co.entry_id = me.entry_id
CROSS JOIN ni3al
WHERE s.strukturbericht = 'L1_2'
  AND c_co.element = 'Co'
  AND ABS(c_co.atomic_fraction - 0.75) < 1e-6
ORDER BY
    ps.energy_above_hull ASC,
    ABS(s.lattice_a - ni3al.ni3al_lattice_a) ASC;
