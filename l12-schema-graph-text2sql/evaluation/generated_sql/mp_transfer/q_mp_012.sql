SELECT COUNT(*) AS count_cubic_band_gap_gt_1ev
FROM mp_entries
WHERE crystal_system = 'Cubic'
  AND band_gap > 1
LIMIT 10000;
