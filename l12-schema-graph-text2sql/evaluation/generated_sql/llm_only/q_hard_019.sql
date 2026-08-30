WITH candidates AS (
  SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    ps.reference_set,
    ps.energy_above_hull
  FROM phase_stability ps
  JOIN material_entry me
    ON me.entry_id = ps.entry_id
  WHERE ps.energy_above_hull > 0.001
    AND ps.energy_above_hull <= 0.05
    AND (
      EXISTS (
        SELECT 1
        FROM structure s
        WHERE s.entry_id = me.entry_id
          AND (
            s.strukturbericht IN ('L1_2', 'L1₂', 'L12')
            OR s.prototype ILIKE '%L1_2%'
            OR s.prototype ILIKE '%L1₂%'
            OR s.prototype ILIKE '%L12%'
          )
      )
      OR EXISTS (
        SELECT 1
        FROM formation_enthalpy fe
        WHERE fe.entry_id = me.entry_id
          AND (
            fe.strukturbericht IN ('L1_2', 'L1₂', 'L12')
            OR fe.prototype ILIKE '%L1_2%'
            OR fe.prototype ILIKE '%L1₂%'
            OR fe.prototype ILIKE '%L12%'
          )
      )
    )
)
SELECT
  RANK() OVER (ORDER BY energy_above_hull ASC) AS stability_rank,
  entry_id,
  formula,
  reduced_formula,
  reference_set,
  energy_above_hull
FROM candidates
ORDER BY energy_above_hull ASC, entry_id;
