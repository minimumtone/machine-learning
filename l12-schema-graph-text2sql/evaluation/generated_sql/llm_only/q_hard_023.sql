WITH ni3al_ref AS (
    SELECT AVG(s.lattice_a) AS ni3al_lattice_a
    FROM structure s
    JOIN material_entry me ON me.entry_id = s.entry_id
    WHERE s.lattice_a IS NOT NULL
      AND (s.strukturbericht ILIKE 'L1_2' OR s.strukturbericht ILIKE 'L12')
      AND (
          me.formula IN ('Ni3Al', 'AlNi3')
          OR me.reduced_formula IN ('Ni3Al', 'AlNi3')
      )
)
SELECT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.lattice_a,
    r.ni3al_lattice_a,
    s.lattice_a - r.ni3al_lattice_a AS lattice_difference,
    ABS(s.lattice_a - r.ni3al_lattice_a) / r.ni3al_lattice_a AS lattice_mismatch,
    100.0 * ABS(s.lattice_a - r.ni3al_lattice_a) / r.ni3al_lattice_a AS lattice_mismatch_percent
FROM structure s
JOIN material_entry me ON me.entry_id = s.entry_id
CROSS JOIN ni3al_ref r
WHERE s.lattice_a IS NOT NULL
  AND (s.strukturbericht ILIKE 'L1_2' OR s.strukturbericht ILIKE 'L12')
ORDER BY lattice_mismatch ASC, me.formula, me.entry_id;
