WITH ni3al AS (
  SELECT s.lattice_a AS ref_lattice_a
  FROM structure s
  JOIN material_entry me ON me.entry_id = s.entry_id
  WHERE s.lattice_a IS NOT NULL
    AND (
      me.reduced_formula IN ('AlNi3', 'Ni3Al')
      OR (
        me.chemical_system = 'Al-Ni'
        AND EXISTS (
          SELECT 1
          FROM composition c
          WHERE c.entry_id = me.entry_id
            AND c.element = 'Ni'
            AND ABS(c.atomic_fraction - 0.75) < 0.01
        )
        AND EXISTS (
          SELECT 1
          FROM composition c
          WHERE c.entry_id = me.entry_id
            AND c.element = 'Al'
            AND ABS(c.atomic_fraction - 0.25) < 0.01
        )
      )
    )
    AND (
      s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
      OR s.prototype ILIKE '%L1%'
      OR s.prototype ILIKE '%Cu3Au%'
    )
  ORDER BY s.lattice_a
  LIMIT 1
),
stability AS (
  SELECT entry_id, MIN(energy_above_hull) AS energy_above_hull
  FROM phase_stability
  GROUP BY entry_id
)
SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  s.prototype,
  s.strukturbericht,
  s.space_group,
  s.lattice_a,
  ni3al.ref_lattice_a AS ni3al_lattice_a,
  ABS(s.lattice_a - ni3al.ref_lattice_a) AS lattice_a_difference,
  100.0 * ABS(s.lattice_a - ni3al.ref_lattice_a) / ni3al.ref_lattice_a AS percent_difference,
  stability.energy_above_hull
FROM structure s
JOIN material_entry me ON me.entry_id = s.entry_id
CROSS JOIN ni3al
LEFT JOIN stability ON stability.entry_id = me.entry_id
WHERE s.lattice_a IS NOT NULL
  AND (
    s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR s.prototype ILIKE '%L1%'
    OR s.prototype ILIKE '%Cu3Au%'
  )
ORDER BY lattice_a_difference ASC
LIMIT 25;
