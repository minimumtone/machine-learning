-- hard: 多形数上位5元素の基底状態情報
SELECT col_papa, col_gemini, col_uniform
FROM tbl_victor
ORDER BY col_uniform DESC, col_papa ASC
LIMIT 5;
