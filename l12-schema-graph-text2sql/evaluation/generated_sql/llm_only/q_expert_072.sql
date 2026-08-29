SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.prototype,
    COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
    sm.method_name AS synthesis_method
FROM material_entry AS me
JOIN material_synthesis AS ms
    ON ms.entry_id = me.entry_id
JOIN synthesis_method AS sm
    ON sm.synthesis_id = ms.synthesis_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE ms.success = TRUE
  AND sm.method_name ILIKE '%arc%melting%'
  AND regexp_replace(lower(COALESCE(s.strukturbericht, pd.strukturbericht, s.prototype, pd.prototype_name, '')), '[^a-z0-9]', '', 'g') = 'l12'
ORDER BY me.formula, me.entry_id;
