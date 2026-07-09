-- hard: 多形数上位5元素の基底状態情報
SELECT symbol, gs_spacegroup, polymorph_count
FROM oqmd_reference_states
ORDER BY polymorph_count DESC, symbol ASC
LIMIT 5;
