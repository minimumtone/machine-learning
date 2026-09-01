SELECT EXISTS (
  SELECT 1
  FROM structure s
  LEFT JOIN prototype_definition pd
    ON s.prototype = pd.prototype_id
  WHERE (
      UPPER(s.strukturbericht) = 'B2'
      OR UPPER(pd.strukturbericht) = 'B2'
      OR UPPER(s.prototype) = 'B2'
    )
    AND LOWER(s.crystal_system) <> 'cubic'
) AS has_b2_non_cubic;
