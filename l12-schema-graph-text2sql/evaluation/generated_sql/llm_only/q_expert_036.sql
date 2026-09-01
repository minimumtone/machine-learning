SELECT EXISTS (
  SELECT 1
  FROM structure s
  LEFT JOIN prototype_definition pd ON pd.prototype_id = s.prototype
  JOIN phase_stability ps ON ps.entry_id = s.entry_id
  WHERE ps.band_gap > 0
    AND (
      s.strukturbericht ILIKE 'L12'
      OR s.strukturbericht ILIKE 'L1_2'
      OR pd.strukturbericht ILIKE 'L12'
      OR pd.strukturbericht ILIKE 'L1_2'
      OR s.prototype ILIKE '%L12%'
      OR s.prototype ILIKE '%L1_2%'
      OR pd.prototype_name ILIKE '%L12%'
      OR pd.prototype_name ILIKE '%L1_2%'
      OR pd.prototype_name ILIKE '%Cu3Au%'
    )
) AS has_l12_compounds_with_positive_band_gap;
