-- CTE K: CTE + 熱物性テーブル到達
-- 「デバイ温度が高い上位10件のL1₂化合物について生成エンタルピーも計算して出して」
WITH thermal_l12 AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom,
           tp.debye_temperature_k
    FROM material_entry m
    JOIN structure s ON s.entry_id = m.entry_id
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    JOIN calculation cal_tp ON cal_tp.entry_id = m.entry_id
JOIN thermal_property tp ON tp.calculation_id = cal_tp.calculation_id
    WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
      AND tp.debye_temperature_k IS NOT NULL
),
enthalpy AS (
    SELECT tl.entry_id, tl.formula, tl.formation_energy_per_atom,
           tl.debye_temperature_k,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM thermal_l12 tl
    JOIN composition comp ON comp.entry_id = tl.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    GROUP BY tl.entry_id, tl.formula, tl.formation_energy_per_atom,
             tl.debye_temperature_k
)
SELECT formula,
       ROUND(debye_temperature_k::numeric, 1) AS debye_temperature_k,
       ROUND((formation_energy_per_atom - weighted_ref)::numeric, 4) AS delta_h_f
FROM enthalpy
ORDER BY debye_temperature_k DESC
LIMIT 10;
