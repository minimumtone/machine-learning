WITH RECURSIVE target_domains AS (
    SELECT domain_id
    FROM application_domain
    WHERE domain_name ILIKE '%high%temperature%superalloy%'

    UNION

    SELECT ad.domain_id
    FROM application_domain ad
    JOIN target_domains td
      ON ad.parent_domain_id = td.domain_id
)
SELECT DISTINCT me.*
FROM material_entry me
JOIN material_application ma
  ON me.entry_id = ma.entry_id
WHERE ma.domain_id IN (SELECT domain_id FROM target_domains)
ORDER BY me.entry_id;
