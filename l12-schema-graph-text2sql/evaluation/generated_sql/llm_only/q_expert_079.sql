WITH RECURSIVE heat_domains AS (
  SELECT domain_id
  FROM application_domain
  WHERE domain_name ILIKE '%heat-resistant%'
     OR domain_name ILIKE '%heat resistant%'
     OR domain_name ILIKE '%high-temperature%'
     OR domain_name ILIKE '%refractory%'
  UNION
  SELECT ad.domain_id
  FROM application_domain ad
  JOIN heat_domains hd ON ad.parent_domain_id = hd.domain_id
)
SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system
FROM material_entry me
JOIN material_application ma ON ma.entry_id = me.entry_id
JOIN heat_domains hd ON hd.domain_id = ma.domain_id
WHERE me.number_of_elements > 1
ORDER BY me.reduced_formula, me.entry_id;
