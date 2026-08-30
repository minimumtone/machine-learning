WITH l12_band_gaps AS (
  SELECT DISTINCT
    ps.entry_id,
    ps.reference_set,
    ps.band_gap
  FROM phase_stability ps
  JOIN structure s
    ON s.entry_id = ps.entry_id
  LEFT JOIN prototype_definition pd
    ON pd.prototype_id = s.prototype
  WHERE ps.band_gap IS NOT NULL
    AND (
      s.strukturbericht IN ('L12', 'L1_2', 'L1₂')
      OR pd.strukturbericht IN ('L12', 'L1_2', 'L1₂')
      OR s.prototype ILIKE '%L12%'
      OR s.prototype ILIKE '%L1_2%'
      OR pd.prototype_name ILIKE '%L12%'
      OR pd.prototype_name ILIKE '%L1_2%'
    )
)
SELECT
  CASE
    WHEN band_gap = 0 THEN 'zero_band_gap'
    ELSE 'nonzero_band_gap'
  END AS band_gap_group,
  COUNT(*) AS n_compounds,
  MIN(band_gap) AS min_band_gap_ev,
  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY band_gap) AS p25_band_gap_ev,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY band_gap) AS median_band_gap_ev,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY band_gap) AS p75_band_gap_ev,
  MAX(band_gap) AS max_band_gap_ev,
  AVG(band_gap) AS avg_band_gap_ev
FROM l12_band_gaps
GROUP BY
  CASE
    WHEN band_gap = 0 THEN 'zero_band_gap'
    ELSE 'nonzero_band_gap'
  END
ORDER BY
  CASE
    WHEN CASE WHEN band_gap = 0 THEN 'zero_band_gap' ELSE 'nonzero_band_gap' END = 'zero_band_gap' THEN 0
    ELSE 1
  END;
