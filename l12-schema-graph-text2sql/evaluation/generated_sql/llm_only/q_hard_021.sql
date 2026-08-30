WITH l12_info AS (
    SELECT
        entry_id,
        MAX(prototype) AS prototype,
        MAX(strukturbericht) AS strukturbericht
    FROM (
        SELECT
            s.entry_id,
            s.prototype,
            s.strukturbericht
        FROM structure s

        UNION ALL

        SELECT
            s.entry_id,
            s.prototype,
            pd.strukturbericht
        FROM structure s
        JOIN prototype_definition pd
            ON s.prototype = pd.prototype_id

        UNION ALL

        SELECT
            fe.entry_id,
            fe.prototype,
            fe.strukturbericht
        FROM formation_enthalpy fe
    ) x
    WHERE lower(
        regexp_replace(
            translate(COALESCE(strukturbericht, ''), '₀₁₂₃₄₅₆₇₈₉', '0123456789'),
            '[^a-z0-9]',
            '',
            'g'
        )
    ) = 'l12'
    GROUP BY entry_id
)
SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    li.prototype,
    li.strukturbericht,
    c.calculation_id,
    c.method,
    c.functional,
    c.calculation_type,
    cp.property_name,
    cp.tensor_component,
    cp.value,
    cp.unit
FROM l12_info li
JOIN material_entry me
    ON me.entry_id = li.entry_id
JOIN calculation c
    ON c.entry_id = me.entry_id
JOIN calculated_property cp
    ON cp.calculation_id = c.calculation_id
WHERE c.method ILIKE '%DFT%'
ORDER BY
    me.chemical_system,
    me.reduced_formula,
    cp.property_name,
    cp.tensor_component,
    c.calculation_id;
