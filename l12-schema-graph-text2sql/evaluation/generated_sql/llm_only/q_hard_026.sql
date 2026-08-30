WITH base AS (
  SELECT DISTINCT ON (f.entry_id, f.reference_set)
    f.entry_id,
    COALESCE(me.formula, f.formula) AS formula,
    COALESCE(me.reduced_formula, f.reduced_formula) AS reduced_formula,
    me.chemical_system,
    f.reference_set,
    f.energy_above_hull,
    f.is_stable,
    COALESCE(f.lattice_a, s.lattice_a) AS lattice_a,
    COALESCE(f.prototype, s.prototype, pd.prototype_name) AS prototype,
    COALESCE(f.strukturbericht, s.strukturbericht, pd.strukturbericht) AS strukturbericht
  FROM formation_enthalpy f
  JOIN material_entry me ON me.entry_id = f.entry_id
  LEFT JOIN structure s ON s.entry_id = f.entry_id
  LEFT JOIN prototype_definition pd ON pd.prototype_id = s.prototype
  WHERE f.energy_above_hull IS NOT NULL
    AND COALESCE(f.lattice_a, s.lattice_a) IS NOT NULL
    AND (
      regexp_replace(translate(lower(COALESCE(f.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^a-z0-9]', '', 'g') IN ('l12', 'cu3au')
      OR regexp_replace(translate(lower(COALESCE(s.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^a-z0-9]', '', 'g') IN ('l12', 'cu3au')
      OR regexp_replace(translate(lower(COALESCE(pd.strukturbericht, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^a-z0-9]', '', 'g') IN ('l12', 'cu3au')
      OR regexp_replace(translate(lower(COALESCE(f.prototype, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^a-z0-9]', '', 'g') IN ('l12', 'cu3au')
      OR regexp_replace(translate(lower(COALESCE(s.prototype, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^a-z0-9]', '', 'g') IN ('l12', 'cu3au')
      OR regexp_replace(translate(lower(COALESCE(pd.prototype_name, '')), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^a-z0-9]', '', 'g') IN ('l12', 'cu3au')
    )
  ORDER BY f.entry_id, f.reference_set, COALESCE(f.lattice_a, s.lattice_a)
),
ni3al AS (
  SELECT *
  FROM base
  WHERE reduced_formula IN ('Ni3Al', 'AlNi3')
     OR formula IN ('Ni3Al', 'AlNi3')
  ORDER BY is_stable DESC, energy_above_hull ASC, lattice_a
  LIMIT 1
)
SELECT
  b.entry_id,
  b.formula,
  b.reduced_formula,
  b.chemical_system,
  b.reference_set,
  b.prototype,
  b.strukturbericht,
  b.energy_above_hull,
  CASE
    WHEN b.energy_above_hull <= 0.001 THEN 'stable'
    WHEN b.energy_above_hull <= 0.05 THEN 'metastable'
    ELSE 'unstable'
  END AS stability_class,
  b.lattice_a,
  ABS(b.energy_above_hull - n.energy_above_hull) AS energy_above_hull_delta_ev_per_atom,
  ABS(b.lattice_a - n.lattice_a) AS lattice_a_delta,
  100.0 * ABS(b.lattice_a - n.lattice_a) / NULLIF(n.lattice_a, 0) AS lattice_a_mismatch_percent,
  sqrt(
    power((b.energy_above_hull - n.energy_above_hull) / 0.05, 2)
    + power((ABS(b.lattice_a - n.lattice_a) / NULLIF(n.lattice_a, 0)) / 0.05, 2)
  ) AS similarity_score
FROM base b
JOIN ni3al n
  ON b.reference_set IS NOT DISTINCT FROM n.reference_set
WHERE NOT (
    b.reduced_formula IN ('Ni3Al', 'AlNi3')
    OR b.formula IN ('Ni3Al', 'AlNi3')
  )
  AND b.energy_above_hull <= 0.05
  AND ABS(b.lattice_a - n.lattice_a) / NULLIF(n.lattice_a, 0) <= 0.05
ORDER BY similarity_score, energy_above_hull_delta_ev_per_atom, lattice_a_delta
LIMIT 20;
