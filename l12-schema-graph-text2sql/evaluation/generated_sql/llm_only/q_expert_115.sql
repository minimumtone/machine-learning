SELECT
  fsrs.source_db,
  fsrs.reference_set,
  res.method,
  res.functional,
  res.source
FROM fixture_source_reference_set AS fsrs
JOIN reference_energy_set AS res
  ON res.reference_set = fsrs.reference_set
ORDER BY
  fsrs.source_db,
  fsrs.reference_set;
