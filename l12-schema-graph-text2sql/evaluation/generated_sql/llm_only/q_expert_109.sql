SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  als.system_name AS alloy_system_name,
  MAX(et.bulk_modulus_vrh) AS bulk_modulus_vrh
FROM material_entry AS me
JOIN material_alloy_system AS mas
  ON mas.entry_id = me.entry_id
JOIN alloy_system AS als
  ON als.alloy_system_id = mas.alloy_system_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
WHERE
  EXISTS (
    SELECT 1
    FROM phase_stability AS ps
    WHERE ps.entry_id = me.entry_id
      AND ps.is_stable = TRUE
  )
  AND et.bulk_modulus_vrh IS NOT NULL
  AND (
    s.strukturbericht IN ('L12', 'L1_2', 'L1₂')
    OR pd.strukturbericht IN ('L12', 'L1_2', 'L1₂')
    OR s.prototype ILIKE '%L1%2%'
    OR pd.prototype_name ILIKE '%L1%2%'
  )
  AND (
    me.chemical_system ~ '(^|-)Ni(-|$)'
    OR als.system_name ~* '(^|[^A-Za-z])Ni([^A-Za-z]|$)'
    OR als.system_name ILIKE '%nickel%'
  )
GROUP BY
  me.entry_id,
  me.formula,
  me.reduced_formula,
  als.system_name
ORDER BY
  MAX(et.bulk_modulus_vrh) DESC
LIMIT 10;
