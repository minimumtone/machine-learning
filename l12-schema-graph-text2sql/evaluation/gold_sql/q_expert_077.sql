-- "Both experimental and calculated values exist": requires an actual
-- measured_property row and an actual calculated_property row, deduplicated
-- via EXISTS so multiple measurements/calculations cannot multiply rows.
SELECT m.entry_id, m.formula
FROM material_entry m
WHERE EXISTS (
    SELECT 1
    FROM experimental_measurement em
    JOIN measured_property mp ON mp.measurement_id = em.measurement_id
    WHERE em.entry_id = m.entry_id
)
AND EXISTS (
    SELECT 1
    FROM calculation ca
    JOIN calculated_property cp ON cp.calculation_id = ca.calculation_id
    WHERE ca.entry_id = m.entry_id
)
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;
