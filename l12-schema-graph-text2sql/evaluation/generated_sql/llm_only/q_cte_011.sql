WITH l12_entries AS (
    SELECT DISTINCT s.entry_id
    FROM structure s
    LEFT JOIN prototype_definition pd
        ON pd.prototype_id = s.prototype
    WHERE upper(replace(replace(coalesce(s.strukturbericht, ''), '₂', '2'), '_', '')) = 'L12'
       OR upper(replace(replace(coalesce(pd.strukturbericht, ''), '₂', '2'), '_', '')) = 'L12'
       OR upper(replace(replace(coalesce(s.prototype, ''), '₂', '2'), '_', '')) LIKE '%L12%'
       OR upper(replace(replace(coalesce(pd.prototype_name, ''), '₂', '2'), '_', '')) LIKE '%L12%'
),
best_debye AS (
    SELECT
        c.entry_id,
        c.calculation_id,
        tp.debye_temperature_k,
        row_number() OVER (
            PARTITION BY c.entry_id
            ORDER BY tp.debye_temperature_k DESC NULLS LAST, c.calculation_id
        ) AS rn
    FROM l12_entries l
    JOIN calculation c
        ON c.entry_id = l.entry_id
    JOIN thermal_property tp
        ON tp.calculation_id = c.calculation_id
    WHERE tp.debye_temperature_k IS NOT NULL
),
top10 AS (
    SELECT entry_id, calculation_id, debye_temperature_k
    FROM best_debye
    WHERE rn = 1
    ORDER BY debye_temperature_k DESC
    LIMIT 10
),
weighted_delta AS (
    SELECT
        comp.entry_id,
        per.reference_set,
        SUM(comp.atomic_fraction * per.delta_e) AS weighted_element_delta_e_calc
    FROM composition comp
    JOIN pure_element_reference per
        ON per.element_symbol = comp.element
    GROUP BY comp.entry_id, per.reference_set
)
SELECT
    me.entry_id,
    me.formula,
    me.chemical_system,
    t.calculation_id,
    t.debye_temperature_k,
    fh.reference_set,
    fh.formation_enthalpy_ev_per_atom,
    COALESCE(fh.weighted_element_delta_e, wd.weighted_element_delta_e_calc) AS weighted_element_delta_e,
    fh.formation_enthalpy_ev_per_atom
        + COALESCE(fh.weighted_element_delta_e, wd.weighted_element_delta_e_calc)
        AS formation_energy_vs_pure_element_ground_states_ev_per_atom
FROM top10 t
JOIN material_entry me
    ON me.entry_id = t.entry_id
LEFT JOIN LATERAL (
    SELECT *
    FROM formation_enthalpy fh
    WHERE fh.entry_id = t.entry_id
    ORDER BY fh.reference_set
    LIMIT 1
) fh ON TRUE
LEFT JOIN weighted_delta wd
    ON wd.entry_id = t.entry_id
   AND wd.reference_set = fh.reference_set
ORDER BY t.debye_temperature_k DESC;
