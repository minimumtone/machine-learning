-- CTE A: 単純CTE（導出量計算）
-- 「Co3Tiの生成エンタルピーを純物質基準エネルギーから計算して」
WITH compound AS (
    SELECT m.entry_id, m.formula, ps.formation_energy_per_atom
    FROM material_entry m
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    WHERE m.formula = 'Co3Ti'
    LIMIT 1
),
ref_energies AS (
    SELECT c.entry_id,
           SUM(comp.atomic_fraction * per.delta_e) AS weighted_ref
    FROM compound c
    JOIN composition comp ON comp.entry_id = c.entry_id
    JOIN pure_element_reference per ON per.element_symbol = comp.element
    GROUP BY c.entry_id
)
SELECT c.formula,
       c.formation_energy_per_atom,
       r.weighted_ref AS ref_energy,
       c.formation_energy_per_atom - r.weighted_ref AS corrected_enthalpy
FROM compound c
JOIN ref_energies r ON r.entry_id = c.entry_id;
