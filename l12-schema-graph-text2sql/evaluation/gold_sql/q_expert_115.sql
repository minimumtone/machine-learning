-- VH: 各材料ソース(source_db)が宣言を許可されているエネルギー規約(reference_set)と、その規約のmethod・functional・sourceを一覧して
-- Tables: fixture_source_reference_set, reference_energy_set (2)
-- Exercises the fixture_source_reference_set map and the reference_energy_set master.
SELECT fsr.source_db, fsr.reference_set,
       res.method, res.functional, res.source
FROM fixture_source_reference_set fsr
JOIN reference_energy_set res
  ON res.reference_set = fsr.reference_set
ORDER BY fsr.source_db, fsr.reference_set
LIMIT 10000;
