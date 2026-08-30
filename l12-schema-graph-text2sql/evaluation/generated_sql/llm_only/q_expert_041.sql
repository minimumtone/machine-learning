SELECT EXISTS (
  SELECT 1
  FROM calculation
  WHERE functional IS NOT NULL
    AND functional <> 'GGA-PBE'
) AS has_non_gga_pbe_entries;
