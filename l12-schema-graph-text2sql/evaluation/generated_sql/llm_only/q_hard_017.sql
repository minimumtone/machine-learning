WITH candidates AS (
  SELECT
    CASE
      WHEN formula IN ('Co3Ti', 'TiCo3') OR reduced_formula IN ('Co3Ti', 'TiCo3') THEN 'Co3Ti'
      WHEN formula IN ('Ni3Al', 'AlNi3') OR reduced_formula IN ('Ni3Al', 'AlNi3') THEN 'Ni3Al'
    END AS material,
    lattice_a,
    is_stable,
    energy_above_hull,
    formation_enthalpy_ev_per_atom
  FROM formation_enthalpy
  WHERE lattice_a IS NOT NULL
    AND (
      formula IN ('Co3Ti', 'TiCo3', 'Ni3Al', 'AlNi3')
      OR reduced_formula IN ('Co3Ti', 'TiCo3', 'Ni3Al', 'AlNi3')
    )
),
ranked AS (
  SELECT
    material,
    lattice_a,
    ROW_NUMBER() OVER (
      PARTITION BY material
      ORDER BY
        CASE WHEN is_stable THEN 0 ELSE 1 END,
        energy_above_hull NULLS LAST,
        formation_enthalpy_ev_per_atom
    ) AS rn
  FROM candidates
  WHERE material IS NOT NULL
)
SELECT
  MAX(lattice_a) FILTER (WHERE material = 'Co3Ti') AS co3ti_lattice_a,
  MAX(lattice_a) FILTER (WHERE material = 'Ni3Al') AS ni3al_lattice_a,
  MAX(lattice_a) FILTER (WHERE material = 'Co3Ti')
    - MAX(lattice_a) FILTER (WHERE material = 'Ni3Al') AS lattice_a_difference
FROM ranked
WHERE rn = 1;
