WITH l12_props AS (
    SELECT
        me.entry_id,
        me.formula,
        me.reduced_formula,
        me.chemical_system,
        MIN(fh.formation_enthalpy_ev_per_atom) AS formation_energy_ev_per_atom,
        MAX(et.bulk_modulus_vrh) AS bulk_modulus_gpa
    FROM material_entry me
    JOIN formation_enthalpy fh
        ON fh.entry_id = me.entry_id
    JOIN calculation c
        ON c.entry_id = me.entry_id
    JOIN elastic_tensor et
        ON et.calculation_id = c.calculation_id
    WHERE fh.formation_enthalpy_ev_per_atom IS NOT NULL
      AND et.bulk_modulus_vrh IS NOT NULL
      AND (
          LOWER(REPLACE(REPLACE(COALESCE(fh.strukturbericht, ''), '_', ''), '₂', '2')) = 'l12'
          OR EXISTS (
              SELECT 1
              FROM structure s
              WHERE s.entry_id = me.entry_id
                AND LOWER(REPLACE(REPLACE(COALESCE(s.strukturbericht, ''), '_', ''), '₂', '2')) = 'l12'
          )
      )
    GROUP BY
        me.entry_id,
        me.formula,
        me.reduced_formula,
        me.chemical_system
),
ni3al_ref AS (
    SELECT
        MIN(lp.formation_energy_ev_per_atom) AS formation_energy_ev_per_atom,
        MAX(lp.bulk_modulus_gpa) AS bulk_modulus_gpa
    FROM l12_props lp
    WHERE lp.formula IN ('Ni3Al', 'AlNi3')
       OR lp.reduced_formula IN ('Ni3Al', 'AlNi3')
       OR (
           lp.chemical_system = 'Al-Ni'
           AND EXISTS (
               SELECT 1
               FROM composition cni
               WHERE cni.entry_id = lp.entry_id
                 AND cni.element = 'Ni'
                 AND ABS(cni.atomic_fraction - 0.75) < 1e-6
           )
           AND EXISTS (
               SELECT 1
               FROM composition cal
               WHERE cal.entry_id = lp.entry_id
                 AND cal.element = 'Al'
                 AND ABS(cal.atomic_fraction - 0.25) < 1e-6
           )
       )
)
SELECT
    lp.entry_id,
    lp.formula,
    lp.reduced_formula,
    lp.chemical_system,
    lp.formation_energy_ev_per_atom,
    lp.bulk_modulus_gpa
FROM l12_props lp
CROSS JOIN ni3al_ref ref
WHERE lp.formation_energy_ev_per_atom < ref.formation_energy_ev_per_atom
  AND lp.bulk_modulus_gpa > ref.bulk_modulus_gpa
ORDER BY
    lp.formation_energy_ev_per_atom ASC,
    lp.bulk_modulus_gpa DESC;
