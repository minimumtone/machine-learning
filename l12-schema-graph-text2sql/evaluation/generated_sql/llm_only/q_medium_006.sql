WITH l12_structures AS (
  SELECT DISTINCT
    s.entry_id,
    COALESCE(pd.prototype_name, s.prototype) AS prototype,
    COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
    s.space_group_number,
    s.crystal_system
  FROM structure s
  LEFT JOIN prototype_definition pd
    ON s.prototype = pd.prototype_id
  WHERE
    upper(regexp_replace(translate(COALESCE(s.strukturbericht, pd.strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g')) = 'L12'
    OR upper(regexp_replace(translate(COALESCE(s.prototype, pd.prototype_name, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789'), '[^A-Z0-9]', '', 'g')) LIKE '%L12%'
)
SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  l12.prototype,
  l12.strukturbericht,
  l12.space_group_number,
  l12.crystal_system,
  ps.reference_set,
  ps.energy_above_hull
FROM material_entry me
JOIN phase_stability ps
  ON me.entry_id = ps.entry_id
JOIN l12_structures l12
  ON me.entry_id = l12.entry_id
WHERE ps.energy_above_hull > 0.001
  AND ps.energy_above_hull <= 0.05
ORDER BY ps.energy_above_hull, me.entry_id;
