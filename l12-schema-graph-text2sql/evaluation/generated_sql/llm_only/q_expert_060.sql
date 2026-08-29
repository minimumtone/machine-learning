SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    s.strukturbericht,
    mp.total_magnetization,
    mp.magnetic_ordering
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN magnetic_property AS mp
    ON mp.entry_id = me.entry_id
WHERE regexp_replace(upper(coalesce(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
  AND (
      lower(mp.magnetic_ordering) IN ('ferromagnetic', 'fm')
      OR lower(mp.magnetic_ordering) LIKE 'ferro%'
  )
  AND EXISTS (
      SELECT 1
      FROM calculation AS c
      JOIN density_of_states AS dos
          ON dos.calculation_id = c.calculation_id
      WHERE c.entry_id = me.entry_id
        AND dos.is_metallic = TRUE
  )
ORDER BY me.entry_id;
